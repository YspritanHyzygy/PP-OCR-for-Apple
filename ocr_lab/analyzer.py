from __future__ import annotations

import io
import math
import os
import time
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from shapely.geometry import MultiPoint

from .apple_vision import AppleVisionOCR
from .contracts import AnalysisResult, DetectedTextRegion, Point, StageTiming, TextLayout
from .model_registry import ModelRegistry
from .onnx_detector import ONNXProbabilityMapDetector
from .onnx_recognizer import ONNXCTCRecognizer
from .region_grouping import group_regions


_UNRECOGNIZED_PLACEHOLDERS = {
    "未识别",
    "无法识别",
    "unrecognized",
    "notrecognized",
    "unknown",
    "n/a",
    "<unk>",
    "[unk]",
}


def usable_recognition_result(
    text: str,
    confidence: float,
    recognizer: str,
    vision_identifier: str,
    minimum_model_confidence: float = 0.75,
) -> bool:
    """Return whether one recognizer result is safe to translate or erase.

    This is deliberately a provider-output gate, not a language whitelist.  A
    whitelist may choose a recognizer before inference, but it must never be
    used after inference to decide whether the returned text is garbage.
    """

    trimmed = text.strip()
    if not trimmed or "\ufffd" in trimmed:
        return False
    if any(unicodedata.category(character) == "Cc" for character in trimmed):
        return False
    normalized = "".join(trimmed.lower().split())
    if normalized in _UNRECOGNIZED_PLACEHOLDERS:
        return False
    if recognizer not in {"none", vision_identifier} and confidence < minimum_model_confidence:
        return False
    return any(
        unicodedata.category(character)[0] in {"L", "N"}
        for character in trimmed
    )


@dataclass(frozen=True)
class ModelSelection:
    detector: str
    classifier: str
    recognizer: str
    threshold: float = 0.35


class ReferenceGeometryDetector:
    """Deterministic local baseline, intentionally not presented as production OCR.

    It groups dark connected strokes into candidate text lines and returns minimum
    rotated rectangles.  The lab uses it when no ONNX detector has been prepared,
    so uploads remain inspectable without inventing recognition results.
    """

    identifier = "reference-geometry"

    def detect(self, image: Image.Image, threshold: float) -> list[DetectedTextRegion]:
        gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
        if gray.size == 0:
            return []
        cutoff = min(185, int(np.percentile(gray, 28)))
        ink = gray <= cutoff

        # Join characters into horizontal lines and genuine vertical columns. The
        # two masks stay separate to avoid turning a dense menu into one giant box.
        horizontal = ndimage.maximum_filter(ink, size=(5, 27))
        vertical = ndimage.maximum_filter(ink, size=(27, 5))
        candidates = self._components(horizontal, ink, image.width, image.height, "horizontal")
        candidates += self._components(vertical, ink, image.width, image.height, "vertical")
        return self._deduplicate(candidates, minimum_confidence=max(0.12, threshold * 0.45))

    @staticmethod
    def _components(
        mask: np.ndarray,
        original_ink: np.ndarray,
        width: int,
        height: int,
        mode: str,
    ) -> list[DetectedTextRegion]:
        labels, count = ndimage.label(mask)
        output: list[DetectedTextRegion] = []
        minimum_area = max(24, int(width * height * 0.000025))
        for label in range(1, count + 1):
            ys, xs = np.nonzero(labels == label)
            if xs.size < minimum_area:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            box_width, box_height = x1 - x0 + 1, y1 - y0 + 1
            if box_width < 10 or box_height < 8:
                continue
            ratio = box_width / max(box_height, 1)
            if mode == "horizontal" and ratio < 1.35:
                continue
            if mode == "vertical" and ratio > 0.78:
                continue

            ink_y, ink_x = np.nonzero(original_ink[y0 : y1 + 1, x0 : x1 + 1])
            if ink_x.size < 8:
                continue
            points = np.column_stack((ink_x + x0, ink_y + y0))
            if points.shape[0] > 1600:
                stride = max(1, points.shape[0] // 1600)
                points = points[::stride]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rectangle = MultiPoint(points).minimum_rotated_rectangle
            if rectangle.is_empty or not hasattr(rectangle, "exterior"):
                coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            else:
                coords = list(rectangle.exterior.coords)[:4]
                if len(coords) != 4 or not np.isfinite(np.asarray(coords)).all():
                    coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            ordered = ReferenceGeometryDetector._clockwise_from_top_left(coords)
            density = float(ink_x.size) / float(max(1, box_width * box_height))
            confidence = min(0.92, 0.38 + math.sqrt(min(1, density)) * 0.52)
            output.append(
                DetectedTextRegion(
                    id=0,
                    quad=[Point(x=float(x), y=float(y)) for x, y in ordered],
                    confidence=confidence,
                    layout=TextLayout(
                        writing_mode=mode,
                        reading_direction="left-to-right" if mode == "horizontal" else "top-to-bottom",
                        confidence=0.35,
                    ),
                )
            )
        return output

    @staticmethod
    def _clockwise_from_top_left(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        center_x = sum(point[0] for point in coords) / len(coords)
        center_y = sum(point[1] for point in coords) / len(coords)
        clockwise = sorted(coords, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
        start = min(range(4), key=lambda index: clockwise[index][0] + clockwise[index][1])
        ordered = clockwise[start:] + clockwise[:start]
        # Image coordinates use y down. Ensure TL -> TR -> BR -> BL.
        cross = (
            (ordered[1][0] - ordered[0][0]) * (ordered[2][1] - ordered[1][1])
            - (ordered[1][1] - ordered[0][1]) * (ordered[2][0] - ordered[1][0])
        )
        if cross < 0:
            ordered = [ordered[0], ordered[3], ordered[2], ordered[1]]
        return ordered

    @staticmethod
    def _bounds(region: DetectedTextRegion) -> tuple[float, float, float, float]:
        xs = [point.x for point in region.quad]
        ys = [point.y for point in region.quad]
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def _deduplicate(
        cls, regions: list[DetectedTextRegion], minimum_confidence: float
    ) -> list[DetectedTextRegion]:
        kept: list[DetectedTextRegion] = []
        for candidate in sorted(regions, key=lambda item: item.confidence, reverse=True):
            if candidate.confidence < minimum_confidence:
                continue
            a = cls._bounds(candidate)
            duplicate = False
            for existing in kept:
                b = cls._bounds(existing)
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                    0, min(a[3], b[3]) - max(a[1], b[1])
                )
                area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
                area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
                if intersection / min(area_a, area_b) > 0.72:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        kept.sort(key=lambda item: (min(point.y for point in item.quad), min(point.x for point in item.quad)))
        for index, region in enumerate(kept, start=1):
            region.id = index
        return kept[:120]


class OCRLabAnalyzer:
    auto_recognizer_id = "verto-auto-recognizer"
    auto_primary_recognizer_id = "pp-ocrv5-mobile-rec"
    auto_fallback_threshold = 0.75

    def __init__(self, model_root: Path | None = None) -> None:
        self.reference = ReferenceGeometryDetector()
        self.apple_vision = AppleVisionOCR()
        self.registry = ModelRegistry(model_root or configured_model_root() or Path(__file__).parent / "models")
        self._onnx_detectors: dict[str, ONNXProbabilityMapDetector] = {}
        self._onnx_recognizers: dict[str, ONNXCTCRecognizer] = {}

    def default_detector_id(self) -> str:
        return (
            "pp-ocrv6-tiny-det"
            if self.registry.detector("pp-ocrv6-tiny-det") is not None
            else self.reference.identifier
        )

    def default_recognizer_id(self) -> str:
        if self.registry.recognizer(self.auto_primary_recognizer_id) is not None:
            return self.auto_recognizer_id
        for identifier in ("pp-ocrv6-small-rec", "pp-ocrv6-tiny-rec", "pp-ocrv6-medium-rec"):
            if self.registry.recognizer(identifier) is not None:
                return identifier
        return "none"

    def analyze(self, payload: bytes, name: str, selection: ModelSelection) -> AnalysisResult:
        total_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception as error:  # Pillow raises several format-specific exceptions.
            raise ValueError("无法读取图片，请使用 JPG、PNG 或 WebP。") from error
        image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
        preprocessing = self._elapsed(preprocess_started)

        detection_started = time.perf_counter()
        warnings: list[str] = []
        detector_version = "baseline-1"
        if selection.detector == self.apple_vision.identifier:
            regions = self.apple_vision.recognize(image)
            detector_version = self.apple_vision.version
            warnings.append("区域坐标与文字均来自 Apple Vision OCR。")
        elif selection.detector == self.reference.identifier:
            regions = self.reference.detect(image, selection.threshold)
            warnings.append("reference-geometry 只用于测试界面和几何基线，不代表生产 detector。")
        else:
            manifest = self.registry.detector(selection.detector)
            if manifest is None:
                raise ValueError(f"detector {selection.detector!r} 尚未准备或未通过 manifest 校验。")
            detector_version = manifest.version
            detector = self._onnx_detectors.get(selection.detector)
            try:
                if detector is None:
                    detector = ONNXProbabilityMapDetector(manifest)
                    self._onnx_detectors[selection.detector] = detector
                regions = detector.detect(image, selection.threshold)
            except Exception as error:
                raise ValueError(f"ONNX detector 推理失败：{error}") from error
        detection = self._elapsed(detection_started)

        layout_started = time.perf_counter()
        if selection.classifier == "geometry-fallback":
            for region in regions:
                self._apply_geometry_layout(region)
        elif selection.classifier != "layout-pending":
            raise ValueError(f"classifier {selection.classifier!r} 尚未准备。")
        layout = self._elapsed(layout_started)

        recognition_started = time.perf_counter()
        recognizer_version = "none"
        region_recognizers: dict[int, str] = {}
        if selection.recognizer == self.auto_recognizer_id:
            manifest = self.registry.recognizer(self.auto_primary_recognizer_id)
            if manifest is None:
                raise ValueError("Verto Auto 缺少 PP-OCRv5 Mobile recognizer。")
            recognizer_version = f"{manifest.version} + Vision low-confidence fallback"
            recognizer = self._onnx_recognizers.get(self.auto_primary_recognizer_id)
            try:
                if recognizer is None:
                    recognizer = ONNXCTCRecognizer(manifest)
                    self._onnx_recognizers[self.auto_primary_recognizer_id] = recognizer
                reads = recognizer.recognize(image, regions)
            except Exception as error:
                raise ValueError(f"Verto Auto 主 recognizer 推理失败：{error}") from error
            reads_by_id = {read.region_id: read for read in reads}
            fallback_regions = []
            for region in regions:
                read = reads_by_id.get(region.id)
                if read is None:
                    fallback_regions.append(region)
                    continue
                region.text = read.text
                region.recognition_confidence = read.confidence
                region_recognizers[region.id] = "onnx"
                if not read.text or read.confidence < self.auto_fallback_threshold:
                    fallback_regions.append(region)
            if fallback_regions and self.apple_vision.available:
                fallback_reads = self.apple_vision.recognize_regions(image, fallback_regions)
                for region, read in zip(fallback_regions, fallback_reads):
                    if read.text and read.confidence >= region.recognition_confidence:
                        region.text = read.text
                        region.recognition_confidence = read.confidence
                        region_recognizers[region.id] = self.apple_vision.identifier
            for region in regions:
                region.layout.script_family = self._script_family(region.text)
        elif selection.recognizer == self.apple_vision.identifier:
            recognizer_version = self.apple_vision.version
            if selection.detector != self.apple_vision.identifier:
                reads = self.apple_vision.recognize_regions(image, regions)
                reads_by_id = {read.region_id: read for read in reads}
                for region in regions:
                    read = reads_by_id.get(region.id)
                    if read is None:
                        continue
                    region.text = read.text
                    region.recognition_confidence = read.confidence
                    region_recognizers[region.id] = self.apple_vision.identifier
            for region in regions:
                region.layout.script_family = self._script_family(region.text)
        elif selection.recognizer != "none":
            manifest = self.registry.recognizer(selection.recognizer)
            if manifest is None:
                raise ValueError(
                    f"recognizer {selection.recognizer!r} 尚未准备或未通过 manifest 校验。"
                )
            recognizer_version = manifest.version
            recognizer = self._onnx_recognizers.get(selection.recognizer)
            try:
                if recognizer is None:
                    recognizer = ONNXCTCRecognizer(manifest)
                    self._onnx_recognizers[selection.recognizer] = recognizer
                reads = recognizer.recognize(image, regions)
            except Exception as error:
                raise ValueError(f"ONNX recognizer 推理失败：{error}") from error
            reads_by_id = {read.region_id: read for read in reads}
            for region in regions:
                read = reads_by_id.get(region.id)
                if read is None:
                    continue
                region.text = read.text
                region.recognition_confidence = read.confidence
                region_recognizers[region.id] = "onnx"
                region.layout.script_family = self._script_family(read.text)
        for region in regions:
            actual_recognizer = region_recognizers.get(region.id, selection.recognizer)
            if not usable_recognition_result(
                region.text,
                region.recognition_confidence,
                actual_recognizer,
                self.apple_vision.identifier,
            ):
                if region.text.strip():
                    warnings.append(f"区域 {region.id} 的 OCR 返回未通过质量门；跳过翻译与背景处理。")
                region.text = ""
                region.recognition_confidence = 0
        recognition = self._elapsed(recognition_started)
        groups = group_regions(regions)

        total = self._elapsed(total_started)
        return AnalysisResult(
            image_name=name,
            width=image.width,
            height=image.height,
            detector=selection.detector,
            classifier=selection.classifier,
            recognizer=selection.recognizer,
            model_versions={
                "detector": detector_version,
                "classifier": (
                    "geometry-1" if selection.classifier == "geometry-fallback" else "pending"
                ),
                "recognizer": recognizer_version,
            },
            regions=regions,
            groups=groups,
            timings_ms=StageTiming(
                preprocessing=preprocessing,
                detection=detection,
                layout=layout,
                recognition=recognition,
                total=total,
            ),
            warnings=warnings,
        )

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    @staticmethod
    def _apply_geometry_layout(region: DetectedTextRegion) -> None:
        start, end = region.quad[0], region.quad[1]
        dx, dy = end.x - start.x, end.y - start.y
        if abs(dx) >= abs(dy):
            region.layout.writing_mode = "horizontal"
            region.layout.reading_direction = "left-to-right"
        else:
            region.layout.writing_mode = "vertical"
            region.layout.reading_direction = "top-to-bottom"
        region.layout.confidence = 0.5

    @staticmethod
    def _script_family(text: str) -> str:
        values = [ord(character) for character in text]
        if any(
            0x3040 <= value <= 0x30FF
            or 0x31F0 <= value <= 0x31FF
            or 0xFF66 <= value <= 0xFF9D
            for value in values
        ):
            return "kana"
        if any(
            0x1100 <= value <= 0x11FF
            or 0x3130 <= value <= 0x318F
            or 0xAC00 <= value <= 0xD7AF
            for value in values
        ):
            return "hangul"
        has_han = any(
            0x3400 <= value <= 0x4DBF
            or 0x4E00 <= value <= 0x9FFF
            or 0xF900 <= value <= 0xFAFF
            for value in values
        )
        has_latin = any(
            0x0041 <= value <= 0x024F
            or 0x1E00 <= value <= 0x1EFF
            for value in values
        )
        if has_han and not has_latin:
            return "han"
        if has_latin and not has_han:
            return "latin"
        return "other"

    @classmethod
    def _merge_vision_text(
        cls,
        regions: list[DetectedTextRegion],
        vision_regions: list[DetectedTextRegion],
    ) -> None:
        candidates = sorted(
            (
                (cls._bbox_iou(region, vision), region_index, vision_index)
                for region_index, region in enumerate(regions)
                for vision_index, vision in enumerate(vision_regions)
            ),
            reverse=True,
        )
        used_regions: set[int] = set()
        used_vision: set[int] = set()
        for score, region_index, vision_index in candidates:
            if score < 0.1:
                break
            if region_index in used_regions or vision_index in used_vision:
                continue
            region = regions[region_index]
            vision = vision_regions[vision_index]
            region.text = vision.text
            region.recognition_confidence = vision.recognition_confidence
            used_regions.add(region_index)
            used_vision.add(vision_index)

    @staticmethod
    def _bbox_iou(lhs: DetectedTextRegion, rhs: DetectedTextRegion) -> float:
        def bounds(region: DetectedTextRegion) -> tuple[float, float, float, float]:
            xs = [point.x for point in region.quad]
            ys = [point.y for point in region.quad]
            return min(xs), min(ys), max(xs), max(ys)

        a, b = bounds(lhs), bounds(rhs)
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1])
        )
        area_a = max(0, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(0, (b[2] - b[0]) * (b[3] - b[1]))
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0


def configured_model_root() -> Path | None:
    value = os.environ.get("OCR_LAB_MODEL_ROOT")
    return Path(value).expanduser().resolve() if value else None
