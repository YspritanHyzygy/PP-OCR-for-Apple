from __future__ import annotations

import math
import warnings

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy import ndimage
from shapely.geometry import MultiPoint

from .contracts import DetectedTextRegion, Point, TextLayout
from .model_registry import ONNXDetectorManifest


class ONNXProbabilityMapDetector:
    """Runs a DB-style probability-map detector from a verified manifest."""

    def __init__(self, manifest: ONNXDetectorManifest) -> None:
        self.manifest = manifest
        available = set(ort.get_available_providers())
        providers = [
            provider
            for provider in ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            if provider in available
        ]
        self.session = ort.InferenceSession(
            str(manifest.artifact),
            providers=providers or None,
        )

    def detect(self, image: Image.Image, threshold: float) -> list[DetectedTextRegion]:
        tensor, valid_width, valid_height = self._prepare(image)
        outputs = self.session.run(
            [self.manifest.output_name] if self.manifest.output_name else None,
            {self.manifest.input_name: tensor},
        )
        if not outputs:
            raise ValueError("ONNX detector 没有输出")
        probability = np.asarray(outputs[0], dtype=np.float32).squeeze()
        if probability.ndim != 2:
            raise ValueError(f"概率图必须是二维，实际为 {list(probability.shape)}")
        probability = np.asarray(
            Image.fromarray(probability, mode="F").resize(
                (self.manifest.input_width, self.manifest.input_height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        cutoff = max(float(threshold), self.manifest.probability_threshold)
        mask = probability >= cutoff
        mask[valid_height:, :] = False
        mask[:, valid_width:] = False
        return self._regions(
            probability,
            mask,
            valid_width,
            valid_height,
            image.width,
            image.height,
            self.manifest.box_score_threshold,
            self.manifest.unclip_ratio,
            self.manifest.maximum_candidates,
        )

    def _prepare(self, image: Image.Image) -> tuple[np.ndarray, int, int]:
        width, height = self.manifest.input_width, self.manifest.input_height
        scale = min(width / image.width, height / image.height)
        valid_width = max(1, min(width, round(image.width * scale)))
        valid_height = max(1, min(height, round(image.height * scale)))
        resized = image.resize((valid_width, valid_height), Image.Resampling.LANCZOS)
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        canvas[:valid_height, :valid_width] = np.asarray(resized, dtype=np.float32) / 255
        if self.manifest.color_order == "BGR":
            canvas = canvas[:, :, ::-1]
        mean = np.asarray(self.manifest.mean, dtype=np.float32)
        std = np.asarray(self.manifest.std, dtype=np.float32)
        canvas = (canvas - mean) / std
        return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), valid_width, valid_height

    @staticmethod
    def _regions(
        probability: np.ndarray,
        mask: np.ndarray,
        valid_width: int,
        valid_height: int,
        image_width: int,
        image_height: int,
        box_score_threshold: float,
        unclip_ratio: float,
        maximum_candidates: int,
    ) -> list[DetectedTextRegion]:
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        scale_x = image_width / valid_width
        scale_y = image_height / valid_height
        regions: list[DetectedTextRegion] = []
        component_sizes = np.bincount(labels.ravel())
        labels_by_size = np.argsort(component_sizes[1:])[::-1] + 1
        for label in labels_by_size[:maximum_candidates]:
            ys, xs = np.nonzero(labels == label)
            if xs.size < 6:
                continue
            score = float(probability[ys, xs].mean())
            if score < box_score_threshold:
                continue
            points = np.column_stack((xs * scale_x, ys * scale_y))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rectangle = MultiPoint(points).minimum_rotated_rectangle
            if rectangle.is_empty or not hasattr(rectangle, "exterior"):
                continue
            coords = list(rectangle.exterior.coords)[:4]
            if len(coords) != 4 or not np.isfinite(np.asarray(coords)).all():
                continue
            coords = ONNXProbabilityMapDetector._long_axis_order(coords)
            width = math.dist(coords[0], coords[1])
            height = math.dist(coords[0], coords[3])
            if width < 3 or height < 3:
                continue
            coords = ONNXProbabilityMapDetector._expand_rectangle(
                coords,
                unclip_ratio,
                image_width,
                image_height,
            )
            regions.append(
                DetectedTextRegion(
                    id=len(regions) + 1,
                    quad=[Point(x=float(x), y=float(y)) for x, y in coords],
                    confidence=min(max(score, 0), 1),
                    layout=TextLayout(),
                )
            )
        regions.sort(
            key=lambda region: (
                min(point.y for point in region.quad), min(point.x for point in region.quad)
            )
        )
        for index, region in enumerate(regions, start=1):
            region.id = index
        return regions[:3000]

    @staticmethod
    def _expand_rectangle(
        coords: list[tuple[float, float]],
        ratio: float,
        image_width: int,
        image_height: int,
    ) -> list[tuple[float, float]]:
        width = math.dist(coords[0], coords[1])
        height = math.dist(coords[0], coords[3])
        perimeter = 2 * (width + height)
        if width <= 0 or height <= 0 or perimeter <= 0:
            return coords
        distance = width * height * ratio / perimeter
        ux = (coords[1][0] - coords[0][0]) / width
        uy = (coords[1][1] - coords[0][1]) / width
        vx = (coords[3][0] - coords[0][0]) / height
        vy = (coords[3][1] - coords[0][1]) / height
        signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        expanded: list[tuple[float, float]] = []
        for point, (u_sign, v_sign) in zip(coords, signs):
            x = point[0] + u_sign * distance * ux + v_sign * distance * vx
            y = point[1] + u_sign * distance * uy + v_sign * distance * vy
            expanded.append(
                (
                    min(max(x, 0), image_width - 1),
                    min(max(y, 0), image_height - 1),
                )
            )
        return expanded

    @staticmethod
    def _long_axis_order(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
        center_x = sum(point[0] for point in coords) / 4
        center_y = sum(point[1] for point in coords) / 4
        cyclic = sorted(coords, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
        long_edge = max(
            range(4),
            key=lambda index: math.dist(cyclic[index], cyclic[(index + 1) % 4]),
        )
        opposite_edge = (long_edge + 2) % 4
        candidates = [
            [cyclic[long_edge], cyclic[(long_edge + 1) % 4]],
            [cyclic[opposite_edge], cyclic[(opposite_edge + 1) % 4]],
        ]
        dx = candidates[0][1][0] - candidates[0][0][0]
        dy = candidates[0][1][1] - candidates[0][0][1]
        horizontal = abs(dx) >= abs(dy)

        for edge in candidates:
            forward = edge[1][0] >= edge[0][0] if horizontal else edge[1][1] >= edge[0][1]
            if not forward:
                edge.reverse()

        # Both long edges have the same reading direction after normalization.
        # Use the upper edge for horizontal text and the right edge for vertical
        # text so recognizer crops never flip when min-area rectangles differ by
        # tiny floating-point amounts between model tiers.
        if horizontal:
            leading = min(candidates, key=lambda edge: (edge[0][1] + edge[1][1]) / 2)
        else:
            leading = max(candidates, key=lambda edge: (edge[0][0] + edge[1][0]) / 2)
        trailing = candidates[1] if leading is candidates[0] else candidates[0]
        return [leading[0], leading[1], trailing[1], trailing[0]]
