from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analyzer import ModelSelection, OCRLabAnalyzer, configured_model_root
from .cloud_services import GoogleCloudServices
from .contracts import AnalysisResult, ModelCatalogResponse, ModelOption, PipelineResult
from .migan import MIGANInpainting
from .model_registry import RegistryEntry
from .pipeline import OCRTranslationPipeline, PipelineSelection


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "ocr_lab" / "public"
FRONTEND_DIST = ROOT / "ocr_lab" / "frontend" / "dist"
SAMPLE_RESULT = PUBLIC / "sample-menu.json"

app = FastAPI(title="Verto OCR Lab", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
analyzer = OCRLabAnalyzer()
cloud_services = GoogleCloudServices()
migan = MIGANInpainting()
pipeline = OCRTranslationPipeline(analyzer, cloud_services, migan)


def default_detector_id() -> str:
    return analyzer.default_detector_id()


def default_recognizer_id() -> str:
    return analyzer.default_recognizer_id()


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "connected",
        "runtime": "ONNX Runtime reference host",
        "modelRoot": str(configured_model_root()) if configured_model_root() else None,
        "defaultDetector": default_detector_id(),
        "defaultRecognizer": default_recognizer_id(),
        "atomicFinalPublish": True,
    }


@app.get("/api/integrations")
def integrations() -> dict[str, object]:
    return {
        "cloud": cloud_services.status(),
        "migan": migan.status(),
        "atomicFinalPublish": True,
        "privacy": "图片只在本次请求的 FastAPI 内存中处理；Google 请求由服务器端适配器发出。",
    }


@app.get("/api/models", response_model=ModelCatalogResponse)
def models() -> ModelCatalogResponse:
    tier_order = {"tiny": 0, "small": 1, "medium": 2}

    def sort_tier(entry: RegistryEntry) -> int:
        identifier = entry.option.id.split("-")
        return next(
            (rank for tier, rank in tier_order.items() if tier in identifier),
            99,
        )

    discovered = sorted(analyzer.registry.detector_entries(), key=sort_tier)
    discovered_recognizers = sorted(analyzer.registry.recognizer_entries(), key=sort_tier)
    discovered_options = [entry.option for entry in discovered]
    detector_options = [
        ModelOption(
            id="reference-geometry",
            label="Reference Geometry",
            kind="detector",
            available=True,
            version="baseline-1",
            detail="本地确定性几何基线",
        )
    ]
    if analyzer.apple_vision.available:
        detector_options.append(
            ModelOption(
                id=analyzer.apple_vision.identifier,
                label=analyzer.apple_vision.label,
                kind="detector",
                available=True,
                version=analyzer.apple_vision.version,
                detail="macOS VNRecognizeTextRequest 坐标与文字",
            )
        )
    if not any(option.id == "verto-text-detector" for option in discovered_options):
        detector_options.append(
            ModelOption(
                id="verto-text-detector",
                label="VertoTextDetector (ONNX)",
                kind="detector",
                available=False,
                version="pending",
                detail="等待训练或导入模型资产",
            )
        )
    detector_options.extend(discovered_options)
    return ModelCatalogResponse(
        detectors=detector_options,
        classifiers=[
            ModelOption(
                id="geometry-fallback",
                label="Geometry Layout Fallback",
                kind="classifier",
                available=True,
                version="geometry-1",
                detail="按 detector 长轴判断横排或竖排",
            ),
            ModelOption(
                id="layout-pending",
                label="TextLayoutClassifier",
                kind="classifier",
                available=False,
                version="pending",
                detail="四方向与文字体系双 head 契约已保留",
            )
        ],
        recognizers=[
            ModelOption(
                id=analyzer.auto_recognizer_id,
                label="Verto Auto (PP-OCRv5 中英日 + Vision 低置信度兜底)",
                kind="recognizer",
                available=(
                    analyzer.registry.recognizer(analyzer.auto_primary_recognizer_id)
                    is not None
                ),
                version="auto-1",
                detail="Tiny/Small detector 裁片；中英日走 PP-OCRv5，低置信度局部走 Vision",
            ),
            ModelOption(
                id="none",
                label="Detection only",
                kind="recognizer",
                available=True,
                version="none",
                detail="首版只报告 detector 坐标，不伪造 OCR",
            ),
            ModelOption(
                id=analyzer.apple_vision.identifier,
                label=analyzer.apple_vision.label,
                kind="recognizer",
                available=analyzer.apple_vision.available,
                version=analyzer.apple_vision.version,
                detail="macOS VNRecognizeTextRequest",
            ),
        ] + [entry.option for entry in discovered_recognizers],
    )


@app.get("/api/sample", response_model=AnalysisResult)
def sample() -> AnalysisResult:
    return AnalysisResult.model_validate_json(SAMPLE_RESULT.read_text(encoding="utf-8"))


@app.get("/api/sample-image")
def sample_image() -> FileResponse:
    return FileResponse(PUBLIC / "sample-menu.png", media_type="image/png")


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(
    image: UploadFile = File(...),
    detector: str = Form("auto"),
    classifier: str = Form("geometry-fallback"),
    recognizer: str = Form("auto"),
    threshold: float = Form(0.35),
) -> AnalysisResult:
    payload = await image.read()
    if len(payload) > 24 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="单张图片不能超过 24 MB。")
    try:
        return analyzer.analyze(
            payload,
            image.filename or "upload",
            ModelSelection(
                default_detector_id() if detector == "auto" else detector,
                classifier,
                default_recognizer_id() if recognizer == "auto" else recognizer,
                min(max(threshold, 0.05), 0.95),
            ),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/cloud-ocr", response_model=AnalysisResult)
async def cloud_ocr(
    image: UploadFile = File(...),
    source_language: str = Form("auto"),
    cloud_feature: str = Form("DOCUMENT_TEXT_DETECTION"),
    cloud_grouping: str = Form("google-paragraph"),
) -> AnalysisResult:
    payload = await image.read()
    if len(payload) > 24 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="单张图片不能超过 24 MB。")
    try:
        return pipeline.run_cloud_ocr(
            payload,
            image.filename or "upload",
            source_language,
            cloud_feature,
            cloud_grouping,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/pipeline", response_model=PipelineResult)
async def run_pipeline(
    image: UploadFile = File(...),
    ocr_provider: str = Form("local"),
    translation_provider: str = Form("google-cloud-translation"),
    reconstructor: str = Form("migan-512"),
    detector: str = Form("auto"),
    classifier: str = Form("geometry-fallback"),
    recognizer: str = Form("auto"),
    threshold: float = Form(0.35),
    source_language: str = Form("auto"),
    target_language: str = Form("zh-Hans"),
    cloud_feature: str = Form("DOCUMENT_TEXT_DETECTION"),
    cloud_grouping: str = Form("google-paragraph"),
) -> PipelineResult:
    payload = await image.read()
    if len(payload) > 24 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="单张图片不能超过 24 MB。")
    try:
        return pipeline.run(
            payload,
            image.filename or "upload",
            PipelineSelection(
                ocr_provider=ocr_provider,
                translation_provider=translation_provider,
                reconstructor=reconstructor,
                detector=detector,
                classifier=classifier,
                recognizer=recognizer,
                threshold=min(max(threshold, 0.05), 0.95),
                source_language=source_language,
                target_language=target_language,
                cloud_feature=cloud_feature,
                cloud_grouping=cloud_grouping,
            ),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str) -> FileResponse:
        candidate = FRONTEND_DIST / path
        return FileResponse(candidate if candidate.is_file() else FRONTEND_DIST / "index.html")
