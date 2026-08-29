from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass

from PIL import Image

from .analyzer import ModelSelection, OCRLabAnalyzer
from .cloud_services import GoogleCloudServices, cloud_document_to_analysis
from .contracts import AnalysisResult, PipelineResult, TranslationRecord
from .migan import MIGANInpainting
from .reconstruction import compose_final_image, encode_png


@dataclass(frozen=True)
class PipelineSelection:
    ocr_provider: str = "local"
    translation_provider: str = "google-cloud-translation"
    reconstructor: str = "migan-512"
    detector: str = "auto"
    classifier: str = "geometry-fallback"
    recognizer: str = "auto"
    threshold: float = 0.35
    source_language: str = "auto"
    target_language: str = "zh-Hans"
    cloud_feature: str = "DOCUMENT_TEXT_DETECTION"
    cloud_grouping: str = "google-paragraph"


class OCRTranslationPipeline:
    def __init__(
        self,
        analyzer: OCRLabAnalyzer,
        cloud: GoogleCloudServices,
        migan: MIGANInpainting,
    ) -> None:
        self.analyzer = analyzer
        self.cloud = cloud
        self.migan = migan

    def run_cloud_ocr(
        self,
        payload: bytes,
        name: str,
        source_language: str,
        feature_type: str = "DOCUMENT_TEXT_DETECTION",
        grouping: str = "google-paragraph",
    ) -> AnalysisResult:
        total_started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        image = self._working_image(payload)
        normalized_payload = self._encode_working_image(image)
        preprocessing_ms = self._elapsed(preprocessing_started)

        ocr_started = time.perf_counter()
        document, _ = self.cloud.document_text_detection(
            normalized_payload,
            image.width,
            image.height,
            source_language,
            feature_type,
        )
        ocr_ms = self._elapsed(ocr_started)
        return cloud_document_to_analysis(
            document,
            name,
            {
                "preprocessing": preprocessing_ms,
                "recognition": ocr_ms,
                "total": self._elapsed(total_started),
            },
            grouping=grouping,
        )

    def run(self, payload: bytes, name: str, selection: PipelineSelection) -> PipelineResult:
        total_started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        image = self._working_image(payload)
        normalized_payload = self._encode_working_image(image)
        preprocessing_ms = self._elapsed(preprocessing_started)

        ocr_started = time.perf_counter()
        if selection.ocr_provider == "google-cloud-vision":
            document, _ = self.cloud.document_text_detection(
                normalized_payload,
                image.width,
                image.height,
                selection.source_language,
                selection.cloud_feature,
            )
            analysis = cloud_document_to_analysis(document, name, grouping=selection.cloud_grouping)
        elif selection.ocr_provider == "local":
            analysis = self.analyzer.analyze(
                normalized_payload,
                name,
                ModelSelection(
                    self.analyzer_default_detector(selection.detector),
                    selection.classifier,
                    self.analyzer_default_recognizer(selection.recognizer),
                    min(max(selection.threshold, 0.05), 0.95),
                ),
            )
        else:
            raise ValueError(f"未知 OCR 提供方：{selection.ocr_provider}")
        ocr_ms = self._elapsed(ocr_started)

        translation_started = time.perf_counter()
        texts = [group.text for group in analysis.groups]
        if selection.translation_provider == "google-cloud-translation":
            records = self.cloud.translate(texts, selection.target_language, selection.source_language)
            records = [
                record.model_copy(update={"group_id": group.id})
                for group, record in zip(analysis.groups, records)
            ]
        elif selection.translation_provider == "none":
            records = [
                TranslationRecord(group_id=group.id, source_text=group.text, translated_text=group.text)
                for group in analysis.groups
            ]
        else:
            raise ValueError(f"未知翻译提供方：{selection.translation_provider}")
        translation_ms = self._elapsed(translation_started)

        reconstruction_started = time.perf_counter()
        translated_by_group = {record.group_id: record.translated_text for record in records}
        final_image, reconstruction_warnings = compose_final_image(
            image,
            analysis,
            translated_by_group,
            selection.reconstructor,
            self.migan,
        )
        reconstruction_ms = self._elapsed(reconstruction_started)

        encode_started = time.perf_counter()
        encoded = base64.b64encode(encode_png(final_image)).decode("ascii")
        encode_ms = self._elapsed(encode_started)
        total_ms = self._elapsed(total_started)
        analysis.warnings.extend(reconstruction_warnings)
        return PipelineResult(
            image_name=name,
            width=image.width,
            height=image.height,
            ocr_provider=selection.ocr_provider,
            translation_provider=selection.translation_provider,
            reconstructor=selection.reconstructor,
            source_language=selection.source_language,
            target_language=selection.target_language,
            analysis=analysis,
            translations=records,
            final_image_base64=encoded,
            final_image_media_type="image/png",
            timings_ms={
                "preprocessing": preprocessing_ms,
                "ocr": ocr_ms,
                "translation": translation_ms,
                "reconstruction": reconstruction_ms,
                "encode": encode_ms,
                "total": total_ms,
            },
            atomic_publish=True,
            warnings=[
                "最终图片在 OCR、翻译、背景修复和文字排版全部完成后一次性返回。",
                *reconstruction_warnings,
            ],
        )

    def analyzer_default_detector(self, value: str) -> str:
        return self.analyzer_default(value, self.analyzer.default_detector_id())

    def analyzer_default_recognizer(self, value: str) -> str:
        return self.analyzer_default(value, self.analyzer.default_recognizer_id())

    @staticmethod
    def analyzer_default(value: str, fallback: str) -> str:
        return fallback if value == "auto" else value

    @staticmethod
    def _working_image(payload: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception as error:
            raise ValueError("无法读取图片，请使用 JPG、PNG 或 WebP。") from error
        image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def _encode_working_image(image: Image.Image) -> bytes:
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=94, optimize=True)
        return output.getvalue()

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
