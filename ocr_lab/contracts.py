from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class TextLayout(BaseModel):
    writing_mode: Literal["horizontal", "vertical", "unknown"] = "unknown"
    reading_direction: Literal[
        "left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top", "unknown"
    ] = "unknown"
    script_family: Literal["latin", "han", "kana", "hangul", "other", "unknown"] = "unknown"
    confidence: float = Field(default=0, ge=0, le=1)


class DetectedTextRegion(BaseModel):
    id: int
    quad: list[Point] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0, le=1)
    recognition_confidence: float = Field(default=0, ge=0, le=1)
    text: str = ""
    layout: TextLayout = Field(default_factory=TextLayout)
    group_id: int | None = None
    line_index: int | None = None


class TextFlowLine(BaseModel):
    index: int
    region_ids: list[int]
    text: str


class TextFlowGroup(BaseModel):
    id: int
    writing_mode: Literal["horizontal", "vertical"]
    lines: list[TextFlowLine]
    text: str


class StageTiming(BaseModel):
    preprocessing: float = 0
    detection: float = 0
    layout: float = 0
    recognition: float = 0
    total: float = 0


class AnalysisResult(BaseModel):
    image_name: str
    width: int
    height: int
    detector: str
    classifier: str
    recognizer: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    regions: list[DetectedTextRegion]
    groups: list[TextFlowGroup] = Field(default_factory=list)
    timings_ms: StageTiming
    warnings: list[str] = Field(default_factory=list)


class ModelOption(BaseModel):
    id: str
    label: str
    kind: Literal["detector", "classifier", "recognizer"]
    available: bool
    version: str = ""
    detail: str = ""


class ModelCatalogResponse(BaseModel):
    detectors: list[ModelOption]
    classifiers: list[ModelOption]
    recognizers: list[ModelOption]


class CloudOCRToken(BaseModel):
    text: str
    quad: list[Point] = Field(min_length=4, max_length=4)
    confidence: float = Field(default=0, ge=0, le=1)


class CloudOCRLine(BaseModel):
    id: str
    text: str
    quad: list[Point] = Field(min_length=4, max_length=4)
    tokens: list[CloudOCRToken] = Field(default_factory=list)


class CloudOCRParagraph(BaseModel):
    id: str
    detected_language: str | None = None
    confidence: float = Field(default=0, ge=0, le=1)
    lines: list[CloudOCRLine] = Field(default_factory=list)


class CloudOCRDocument(BaseModel):
    provider: str
    width: int
    height: int
    feature_type: Literal["DOCUMENT_TEXT_DETECTION", "TEXT_DETECTION"] = "DOCUMENT_TEXT_DETECTION"
    paragraphs: list[CloudOCRParagraph] = Field(default_factory=list)


class TranslationRecord(BaseModel):
    group_id: int
    source_text: str
    translated_text: str
    detected_language: str | None = None


class PipelineResult(BaseModel):
    image_name: str
    width: int
    height: int
    ocr_provider: str
    translation_provider: str
    reconstructor: str
    source_language: str
    target_language: str
    analysis: AnalysisResult
    translations: list[TranslationRecord] = Field(default_factory=list)
    final_image_base64: str
    final_image_media_type: str = "image/jpeg"
    timings_ms: dict[str, float] = Field(default_factory=dict)
    atomic_publish: bool = True
    warnings: list[str] = Field(default_factory=list)
