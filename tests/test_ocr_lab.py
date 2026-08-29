from __future__ import annotations

import io
import json
import hashlib
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from ocr_lab.apple_vision import AppleVisionOCR
from ocr_lab.app import app
from ocr_lab.analyzer import ModelSelection, OCRLabAnalyzer, usable_recognition_result
from ocr_lab.cloud_services import GoogleCloudServices, cloud_document_to_analysis, parse_cloud_document, parse_text_annotations
from ocr_lab.contracts import DetectedTextRegion, Point
from ocr_lab.migan import MIGANInpainting
from ocr_lab.model_registry import ModelRegistry
from ocr_lab.onnx_detector import ONNXProbabilityMapDetector
from ocr_lab.onnx_recognizer import ONNXCTCRecognizer
from ocr_lab.pipeline import OCRTranslationPipeline, PipelineSelection
from ocr_lab.reconstruction import (
    _line_quad,
    build_text_mask,
    compose_final_image,
    estimate_foreground_color,
    expanded_quad,
    render_translation,
    split_translation_lines,
)
from ocr_lab.region_grouping import group_regions


client = TestClient(app)


def test_catalog_distinguishes_available_reference_from_pending_models() -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    detectors = {option["id"]: option for option in payload["detectors"]}
    assert detectors["reference-geometry"]["available"] is True
    assert detectors["verto-text-detector"]["available"] is False


def test_health_reports_the_available_default_detector() -> None:
    catalog = client.get("/api/models").json()
    available = {
        option["id"] for option in catalog["detectors"] if option["available"]
    }
    health = client.get("/api/health").json()

    assert health["defaultDetector"] in available
    if "pp-ocrv6-tiny-det" in available:
        assert health["defaultDetector"] == "pp-ocrv6-tiny-det"

    available_recognizers = {
        option["id"] for option in catalog["recognizers"] if option["available"]
    }
    if "verto-auto-recognizer" in available_recognizers:
        assert health["defaultRecognizer"] == "verto-auto-recognizer"


def test_recognizer_character_sets_make_script_coverage_explicit() -> None:
    model_root = Path("ocr_lab/models")
    tiny = set((model_root / "pp-ocrv6-tiny-rec-charset.txt").read_text(encoding="utf-8"))
    multilingual = set(
        (model_root / "pp-ocrv5-mobile-rec-charset.txt").read_text(encoding="utf-8")
    )
    thai = set(
        (model_root / "pp-ocrv5-thai-mobile-rec-charset.txt").read_text(encoding="utf-8")
    )

    assert not set("のおすすめ").issubset(tiny)
    assert set("本日のおすすめ").issubset(multilingual)
    assert set("ภาษาไทย").issubset(thai)


def test_recognition_quality_gate_does_not_use_language_whitelist() -> None:
    assert usable_recognition_result("ภาษาไทย", 0.8, "some-latin-only-model", "apple-vision-ocr") is True
    assert usable_recognition_result("本日のおすすめ", 0.8, "pp-ocrv6-small-rec", "apple-vision-ocr") is True
    assert usable_recognition_result("本日0书寸寸", 0.8, "pp-ocrv6-small-rec", "apple-vision-ocr") is True
    assert usable_recognition_result("未识别", 0.99, "apple-vision-ocr", "apple-vision-ocr") is False
    assert usable_recognition_result("abc\ufffddef", 0.99, "pp-ocrv6-small-rec", "apple-vision-ocr") is False
    assert usable_recognition_result("abc", 0.4, "pp-ocrv6-small-rec", "apple-vision-ocr") is False


def test_sample_preserves_vertical_region_contract() -> None:
    response = client.get("/api/sample")
    assert response.status_code == 200
    vertical = [
        region for region in response.json()["regions"]
        if region["layout"]["writing_mode"] == "vertical"
    ]
    assert len(vertical) == 1
    assert vertical[0]["layout"]["reading_direction"] == "top-to-bottom"
    assert len(vertical[0]["quad"]) == 4


def test_uploaded_image_runs_real_reference_geometry() -> None:
    image = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 70, 510, 104), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/api/analyze",
        files={"image": ("line.png", buffer.getvalue(), "image/png")},
        data={
            "detector": "reference-geometry",
            "classifier": "layout-pending",
            "recognizer": "none",
            "threshold": "0.35",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["image_name"] == "line.png"
    assert payload["detector"] == "reference-geometry"
    assert payload["timings_ms"]["total"] >= payload["timings_ms"]["detection"]
    assert payload["regions"]


def test_unavailable_detector_fails_instead_of_returning_fixture_data() -> None:
    image = Image.new("RGB", (20, 20), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = client.post(
        "/api/analyze",
        files={"image": ("blank.png", buffer.getvalue(), "image/png")},
        data={"detector": "verto-text-detector"},
    )
    assert response.status_code == 422
    assert "尚未准备" in response.json()["detail"]


def test_manifest_with_wrong_checksum_is_not_selectable(tmp_path) -> None:
    artifact = tmp_path / "detector.onnx"
    artifact.write_bytes(b"not-a-real-model")
    manifest = {
        "id": "broken-detector",
        "label": "Broken detector",
        "kind": "detector",
        "version": "1.0.0",
        "source": "fixture",
        "license": "Apache-2.0",
        "artifact": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": "0" * 64,
        "output_contract": "db-probability-map-v1",
        "input": {
            "name": "x",
            "shape": [1, 3, 64, 64],
            "color_order": "RGB",
            "mean": [0, 0, 0],
            "std": [1, 1, 1],
        },
    }
    (tmp_path / "broken.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    entry = ModelRegistry(tmp_path).detector_entries()[0]

    assert entry.option.id == "broken-detector"
    assert entry.option.available is False
    assert "SHA-256" in entry.option.detail


def test_verified_onnx_probability_map_runs_through_runtime(tmp_path) -> None:
    probability = np.zeros((1, 1, 64, 64), dtype=np.float32)
    probability[:, :, 16:28, 10:48] = 0.92
    output_tensor = helper.make_tensor(
        "probability",
        TensorProto.FLOAT,
        probability.shape,
        probability.flatten(),
    )
    graph = helper.make_graph(
        [helper.make_node("Constant", inputs=[], outputs=["probability"], value=output_tensor)],
        "fixture-detector",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("probability", TensorProto.FLOAT, [1, 1, 64, 64])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    artifact = tmp_path / "fixture.onnx"
    onnx.save(model, artifact)
    manifest = {
        "id": "fixture-onnx",
        "label": "Fixture ONNX",
        "kind": "detector",
        "version": "test-1",
        "source": "generated-test-fixture",
        "license": "Apache-2.0",
        "artifact": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "output_contract": "db-probability-map-v1",
        "output_name": "probability",
        "input": {
            "name": "x",
            "shape": [1, 3, 64, 64],
            "color_order": "RGB",
            "mean": [0, 0, 0],
            "std": [1, 1, 1],
        },
    }
    (tmp_path / "fixture.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    image = Image.new("RGB", (128, 64), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    result = OCRLabAnalyzer(tmp_path).analyze(
        buffer.getvalue(),
        "fixture.png",
        ModelSelection("fixture-onnx", "layout-pending", "none", 0.35),
    )

    assert result.model_versions["detector"] == "test-1"
    assert len(result.regions) == 1
    assert result.regions[0].confidence > 0.9


def test_detector_quad_order_is_stable_across_opposite_long_edges() -> None:
    horizontal = ONNXProbabilityMapDetector._long_axis_order(
        [(168, 311), (479, 248), (465, 178), (154, 241)]
    )
    vertical = ONNXProbabilityMapDetector._long_axis_order(
        [(1238, 229), (1250, 889), (1187, 891), (1175, 230)]
    )

    assert horizontal == [(154, 241), (465, 178), (479, 248), (168, 311)]
    assert vertical == [(1238, 229), (1250, 889), (1187, 891), (1175, 230)]


def test_region_grouping_preserves_boxes_but_links_fragments_and_lines() -> None:
    def region(identifier: int, text: str, x: float, y: float, width: float) -> DetectedTextRegion:
        return DetectedTextRegion(
            id=identifier,
            text=text,
            confidence=1,
            quad=[
                Point(x=x, y=y), Point(x=x + width, y=y),
                Point(x=x + width, y=y + 20), Point(x=x, y=y + 20),
            ],
            layout={
                "writing_mode": "horizontal",
                "reading_direction": "left-to-right",
                "script_family": "latin",
                "confidence": 1,
            },
        )

    regions = [
        region(1, "Hello", 10, 10, 55),
        region(2, "world", 72, 10, 55),
        region(3, "Second line", 10, 38, 117),
        region(4, "Separate", 300, 10, 90),
    ]

    groups = group_regions(regions)

    assert [line.region_ids for line in groups[0].lines] == [[1, 2], [3]]
    assert groups[0].text == "Hello world\nSecond line"
    assert regions[0].group_id == regions[1].group_id == regions[2].group_id
    assert regions[0].line_index == regions[1].line_index == 1
    assert regions[2].line_index == 2
    assert regions[3].group_id != regions[0].group_id


def test_reconstruction_keeps_line_tilt_and_separates_erase_geometry() -> None:
    region = DetectedTextRegion(
        id=1,
        text="Tilted",
        confidence=1,
        quad=[
            Point(x=30, y=50), Point(x=130, y=70),
            Point(x=126, y=90), Point(x=26, y=70),
        ],
        layout={
            "writing_mode": "horizontal",
            "reading_direction": "left-to-right",
            "script_family": "latin",
            "confidence": 1,
        },
    )

    line_quad = _line_quad([region])
    erase = expanded_quad(region.quad, region.layout.writing_mode)

    assert line_quad[1].y > line_quad[0].y, "译文不能退化成轴对齐框"
    original_area = abs(sum(
        region.quad[index].x * region.quad[(index + 1) % 4].y
        - region.quad[(index + 1) % 4].x * region.quad[index].y
        for index in range(4)
    ))
    erase_area = abs(sum(
        erase[index].x * erase[(index + 1) % 4].y
        - erase[(index + 1) % 4].x * erase[index].y
        for index in range(4)
    ))
    assert erase_area > original_area


def test_loose_local_box_only_masks_detected_strokes() -> None:
    image = Image.new("RGB", (160, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.line((38, 48, 122, 48), fill="black", width=4)
    region = DetectedTextRegion(
        id=1,
        text="text",
        confidence=1,
        quad=[
            Point(x=25, y=30), Point(x=135, y=30),
            Point(x=135, y=70), Point(x=25, y=70),
        ],
        layout={
            "writing_mode": "horizontal",
            "reading_direction": "left-to-right",
            "script_family": "latin",
            "confidence": 1,
        },
    )
    analysis = type("Analysis", (), {"detector": "local", "regions": [region]})()

    mask = build_text_mask(image, analysis)

    assert mask.getpixel((80, 48)) == 0
    assert mask.getpixel((80, 35)) == 255


def test_translation_uses_original_ink_colour() -> None:
    image = Image.new("RGB", (160, 100), (245, 230, 200))
    draw = ImageDraw.Draw(image)
    quad = [Point(x=20, y=35), Point(x=140, y=35), Point(x=140, y=65), Point(x=20, y=65)]
    draw.rectangle((45, 43, 110, 57), fill=(24, 110, 90))

    colour = estimate_foreground_color(image, quad)
    render_translation(image, quad, "OK", "horizontal", colour)

    assert colour[1] > colour[0], "译文颜色没有继承原文字的绿色倾向"
    pixels = image.load()
    assert any(
        pixels[x, y][1] > pixels[x, y][0] and pixels[x, y][1] > pixels[x, y][2]
        for y in range(image.height)
        for x in range(image.width)
    )


def test_translation_line_mapping_never_repeats_a_last_line() -> None:
    assert split_translation_lines("one\ntwo", 2) == ["one", "two"]
    assert split_translation_lines("one two three", 2) == ["one two", "three"]
    assert split_translation_lines("仅", 3) is None


def test_verified_ctc_recognizer_crops_and_decodes(tmp_path) -> None:
    probabilities = np.full((1, 6, 4), 0.01, dtype=np.float32)
    for step, index in enumerate([1, 0, 3, 0, 2, 0]):
        probabilities[0, step, index] = 0.97
    output_tensor = helper.make_tensor(
        "probabilities",
        TensorProto.FLOAT,
        probabilities.shape,
        probabilities.flatten(),
    )
    graph = helper.make_graph(
        [helper.make_node("Constant", inputs=[], outputs=["probabilities"], value=output_tensor)],
        "fixture-recognizer",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 48, 64])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 6, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    artifact = tmp_path / "fixture-rec.onnx"
    onnx.save(model, artifact)
    characters = tmp_path / "fixture-charset.txt"
    characters.write_text("A\nB\n", encoding="utf-8")
    manifest = {
        "id": "fixture-rec",
        "label": "Fixture recognizer",
        "kind": "recognizer",
        "version": "test-1",
        "source": "generated-test-fixture",
        "license": "Apache-2.0",
        "artifact": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "output_contract": "ctc-probabilities-v1",
        "output_name": "probabilities",
        "characters_file": characters.name,
        "characters_sha256": hashlib.sha256(characters.read_bytes()).hexdigest(),
        "characters_count": 2,
        "class_count": 4,
        "input": {
            "name": "x",
            "shape": [1, 3, 48, 64],
            "color_order": "RGB",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
    }
    (tmp_path / "fixture-rec.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    recognizer_manifest = ModelRegistry(tmp_path).recognizer("fixture-rec")
    assert recognizer_manifest is not None
    region = DetectedTextRegion(
        id=1,
        quad=[
            Point(x=0, y=0), Point(x=127, y=0),
            Point(x=127, y=31), Point(x=0, y=31),
        ],
        confidence=1,
    )

    result = ONNXCTCRecognizer(recognizer_manifest).recognize(
        Image.new("RGB", (128, 32), "white"),
        [region],
    )[0]

    assert result.text == "A B"
    assert result.confidence == pytest.approx(0.97)


@pytest.mark.skipif(not AppleVisionOCR().available, reason="Apple Vision requires macOS")
def test_apple_vision_reads_the_bundled_menu() -> None:
    regions = AppleVisionOCR().recognize(
        Image.open("ocr_lab/public/sample-menu.png").convert("RGB")
    )

    assert len(regions) >= 5
    assert any(region.text for region in regions)


def test_cloud_parser_rebuilds_physical_lines_and_break_spaces() -> None:
    raw = {
        "pages": [{
            "blocks": [{
                "paragraphs": [{
                    "property": {"detectedLanguages": [{"languageCode": "ja", "confidence": 0.98}]},
                    "words": [
                        {"confidence": 0.9, "boundingBox": {"vertices": [{"x": 10, "y": 10}, {"x": 30, "y": 10}, {"x": 30, "y": 30}, {"x": 10, "y": 30}]}, "symbols": [{"text": "本"}, {"text": "日", "property": {"detectedBreak": {"type": "SPACE"}}}]},
                        {"confidence": 0.8, "boundingBox": {"vertices": [{"x": 34, "y": 10}, {"x": 54, "y": 10}, {"x": 54, "y": 30}, {"x": 34, "y": 30}]}, "symbols": [{"text": "の", "property": {"detectedBreak": {"type": "EOL_SURE_SPACE"}}}]},
                    ],
                }],
            }],
        }],
    }

    document = parse_cloud_document(raw, 100, 100)
    analysis = cloud_document_to_analysis(document, "cloud.png")

    assert document.paragraphs[0].detected_language == "ja"
    assert document.paragraphs[0].lines[0].text == "本日 の"
    assert analysis.groups[0].text == "本日 の"
    assert len(analysis.regions[0].quad) == 4


def test_text_detection_fallback_keeps_annotations_selectable() -> None:
    annotations = [
        {"description": "Hello world"},
        {"description": "Hello", "boundingPoly": {"vertices": [{"x": 5, "y": 5}, {"x": 35, "y": 5}, {"x": 35, "y": 20}, {"x": 5, "y": 20}]}},
        {"description": "world", "boundingPoly": {"vertices": [{"x": 40, "y": 5}, {"x": 75, "y": 5}, {"x": 75, "y": 20}, {"x": 40, "y": 20}]}},
    ]

    document = parse_text_annotations(annotations, 100, 100)
    analysis = cloud_document_to_analysis(document, "text.png", grouping="google-paragraph")

    assert document.feature_type == "TEXT_DETECTION"
    assert analysis.model_versions["detector"] == "TEXT_DETECTION"
    assert analysis.regions[0].text == "Hello world"


class _FakeMIGAN:
    def inpaint(self, image, hole_mask):
        return Image.new("RGB", (512, 512), (240, 40, 40))


def test_full_pipeline_runs_one_repair_and_returns_only_final_image() -> None:
    image = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.line((28, 39, 92, 39), fill="black", width=4)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    analysis = OCRLabAnalyzer().analyze(
        buffer.getvalue(),
        "pipeline.png",
        ModelSelection("reference-geometry", "geometry-fallback", "none", 0.35),
    )
    # Keep the detector fixture deliberately loose; the regression is that
    # MI-GAN must still receive only the stroke mask, not this whole rectangle.
    analysis.regions[0].quad = [
        Point(x=20, y=25), Point(x=100, y=25),
        Point(x=100, y=55), Point(x=20, y=55),
    ]
    for region in analysis.regions:
        region.text = "hello"
    analysis.groups = group_regions(analysis.regions)

    repaired, warnings = compose_final_image(
        image,
        analysis,
        {group.id: "你好" for group in analysis.groups},
        "migan-512",
        _FakeMIGAN(),
    )

    assert repaired.getpixel((0, 0)) == (255, 255, 255)
    assert repaired.getpixel((50, 39)) != image.getpixel((50, 39))
    assert repaired.getpixel((50, 30)) == (255, 255, 255)
    assert warnings == []


def test_migan_status_is_explicit_when_no_local_model_is_configured() -> None:
    status = MIGANInpainting(model_path=None).status()

    assert status["available"] is False
    assert "OCR_LAB_MIGAN_MODEL" in status["detail"]


def test_google_cloud_adapter_keeps_credentials_server_side(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_OCR_ACCESS_TOKEN", "local-test-token")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "verto-test-project")
    calls = []

    def fake_post(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        request = httpx.Request("POST", endpoint)
        if "vision.googleapis.com" in endpoint:
            return httpx.Response(200, request=request, json={"responses": [{"fullTextAnnotation": {"pages": []}}]})
        return httpx.Response(200, request=request, json={"translations": [{"translatedText": "你好", "detectedLanguageCode": "en"}]})

    monkeypatch.setattr("ocr_lab.cloud_services.httpx.post", fake_post)
    service = GoogleCloudServices()
    document, _ = service.document_text_detection(b"image", 20, 20, "auto")
    records = service.translate(["hello"], "zh-Hans", "auto")

    assert document.provider == "google-cloud-vision"
    assert records[0].translated_text == "你好"
    assert all(kwargs["headers"]["Authorization"] == "Bearer local-test-token" for _, kwargs in calls)
    assert all(kwargs["headers"]["x-goog-user-project"] == "verto-test-project" for _, kwargs in calls)
    assert calls[1][0].endswith("projects/verto-test-project/locations/global:translateText")
