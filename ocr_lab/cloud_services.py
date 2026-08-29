from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

import httpx
from shapely.geometry import MultiPoint

from .contracts import (
    AnalysisResult,
    CloudOCRDocument,
    CloudOCRLine,
    CloudOCRParagraph,
    CloudOCRToken,
    DetectedTextRegion,
    Point,
    StageTiming,
    TextFlowGroup,
    TextFlowLine,
    TextLayout,
    TranslationRecord,
)
from .region_grouping import group_regions


VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
TRANSLATION_ENDPOINT = "https://translation.googleapis.com/v3"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class GoogleCloudServices:
    """Small, server-side Google Cloud adapter for the localhost lab.

    Credentials never enter the browser.  The adapter accepts an explicitly
    supplied development bearer token or Application Default Credentials.  It
    does not write images, OCR text, or translations to disk.
    """

    provider = "google-cloud"

    def __init__(self) -> None:
        self._credentials = None
        self._project_id: str | None = None

    def status(self) -> dict[str, object]:
        token = os.environ.get("GOOGLE_OCR_ACCESS_TOKEN") or os.environ.get("GOOGLE_TRANSLATE_ACCESS_TOKEN")
        project = self._configured_project()
        auth_installed = self._google_auth_installed()
        adc_available = auth_installed and self._adc_available()
        gcloud_available = self._gcloud_authenticated()
        can_authenticate = bool(token or adc_available or gcloud_available)
        ocr_available = can_authenticate
        translation_available = can_authenticate and bool(project)
        return {
            "ocr": {
                "id": "google-cloud-vision",
                "label": "Google Cloud Vision",
                "available": ocr_available,
                "detail": self._detail(can_authenticate, needs_project=False, project=project),
            },
            "translation": {
                "id": "google-cloud-translation",
                "label": "Google Cloud Translation",
                "available": translation_available,
                "detail": self._detail(can_authenticate, needs_project=True, project=project),
            },
            "project": project or self._project_id,
            "credentialsSource": "bearer token" if token else ("Application Default Credentials" if adc_available else ("gcloud CLI" if gcloud_available else None)),
            "localOnly": True,
        }

    def document_text_detection(
        self,
        payload: bytes,
        width: int,
        height: int,
        source_language: str,
        feature_type: str = "DOCUMENT_TEXT_DETECTION",
    ) -> tuple[CloudOCRDocument, float]:
        if feature_type not in {"DOCUMENT_TEXT_DETECTION", "TEXT_DETECTION"}:
            raise ValueError(f"不支持的 Google Vision 类型：{feature_type}")
        token = self._access_token()
        encoded = base64.b64encode(payload).decode("ascii")
        request: dict[str, object] = {
            "image": {"content": encoded},
            "features": [{"type": feature_type}],
        }
        if source_language and source_language != "auto":
            request["imageContext"] = {"languageHints": [source_language]}
        response = self._post(
            VISION_ENDPOINT,
            {"requests": [request]},
            token,
            project=self._configured_project(),
        )
        item = response.get("responses", [{}])[0]
        if item.get("error"):
            raise ValueError(_google_error(item["error"]))
        full_text = item.get("fullTextAnnotation") or {}
        if full_text.get("pages"):
            document = parse_cloud_document(full_text, width, height, feature_type)
        else:
            document = parse_text_annotations(item.get("textAnnotations") or [], width, height, feature_type)
        return document, 0

    def translate(
        self,
        texts: list[str],
        target_language: str,
        source_language: str,
    ) -> list[TranslationRecord]:
        if not texts:
            return []
        project = self._configured_project()
        if not project:
            raise ValueError("Google Cloud Translation 需要设置 GOOGLE_CLOUD_PROJECT。")
        token = self._access_token()
        body: dict[str, object] = {
            "contents": texts,
            "targetLanguageCode": google_language_code(target_language),
            "mimeType": "text/plain",
        }
        if source_language and source_language != "auto":
            body["sourceLanguageCode"] = google_language_code(source_language)
        endpoint = f"{TRANSLATION_ENDPOINT}/projects/{project}/locations/global:translateText"
        response = self._post(endpoint, body, token, project=project)
        values = response.get("translations") or []
        if len(values) != len(texts):
            raise ValueError("Google Cloud Translation 返回的条数与请求不一致。")
        return [
            TranslationRecord(
                group_id=index + 1,
                source_text=text,
                translated_text=str(value.get("translatedText", "")),
                detected_language=value.get("detectedLanguageCode"),
            )
            for index, (text, value) in enumerate(zip(texts, values))
        ]

    def _access_token(self) -> str:
        explicit = os.environ.get("GOOGLE_OCR_ACCESS_TOKEN") or os.environ.get("GOOGLE_TRANSLATE_ACCESS_TOKEN")
        if explicit:
            return explicit
        if self._credentials is None:
            try:
                import google.auth
                from google.auth.transport.requests import Request

                self._credentials, detected_project = google.auth.default(scopes=SCOPES)
                self._project_id = self._project_id or detected_project
                self._credentials.refresh(Request())
            except Exception:
                self._credentials = False
        if self._credentials is not False:
            if not self._credentials.token:
                raise ValueError("Google Cloud 凭证没有返回访问令牌。")
            return str(self._credentials.token)
        token = self._gcloud_access_token()
        if token:
            return token
        raise ValueError(
            "未找到 Google Cloud 凭证。请设置 GOOGLE_APPLICATION_CREDENTIALS，"
            "或使用已登录的 gcloud CLI。GOOGLE_OCR_ACCESS_TOKEN 也可用于本机短时测试。"
        )

    @staticmethod
    def _post(
        endpoint: str,
        payload: dict[str, object],
        token: str,
        project: str | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"}
        if project:
            headers["x-goog-user-project"] = project
        try:
            response = httpx.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            decoded = response.json()
        except httpx.HTTPStatusError as error:
            raise ValueError(f"Google Cloud 请求失败（HTTP {error.response.status_code}）：{error.response.text[:500]}") from error
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            raise ValueError(f"Google Cloud 请求失败：{error}") from error
        if not isinstance(decoded, dict):
            raise ValueError("Google Cloud 返回了无效 JSON。")
        return decoded

    @staticmethod
    def _google_auth_installed() -> bool:
        try:
            import google.auth  # noqa: F401
        except ImportError:
            return False
        return True

    def _adc_available(self) -> bool:
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            return Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).is_file()
        try:
            import google.auth

            credentials, detected_project = google.auth.default(scopes=SCOPES)
            self._project_id = self._project_id or detected_project
            return credentials is not None
        except Exception:
            return False

    @staticmethod
    def _detail(can_authenticate: bool, needs_project: bool, project: str | None) -> str:
        if not can_authenticate:
            return "未配置 Google Cloud 凭证；浏览器不会保存或发送凭证。"
        if needs_project and not project:
            return "已找到凭证，但还需要 GOOGLE_CLOUD_PROJECT。"
        return "已配置；请求由本地 FastAPI 进程直接发送。"

    def _configured_project(self) -> str | None:
        explicit = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
        if explicit:
            self._project_id = explicit
            return explicit
        if self._project_id:
            return self._project_id
        if not shutil.which("gcloud"):
            return None
        completed = self._run_gcloud(["config", "get-value", "project", "--quiet"])
        project = completed.stdout.strip() if completed.returncode == 0 else ""
        self._project_id = project or None
        return self._project_id

    def _gcloud_authenticated(self) -> bool:
        if not shutil.which("gcloud"):
            return False
        completed = self._run_gcloud(["auth", "list", "--filter=status:ACTIVE", "--format=value(status)"])
        return completed.returncode == 0 and bool(completed.stdout.strip())

    def _gcloud_access_token(self) -> str | None:
        for command in (
            ["auth", "application-default", "print-access-token", "--quiet"],
            ["auth", "print-access-token", "--quiet"],
        ):
            completed = self._run_gcloud(command)
            token = completed.stdout.strip()
            if completed.returncode == 0 and token:
                return token
        return None

    @staticmethod
    def _run_gcloud(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["gcloud", *arguments],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return subprocess.CompletedProcess(["gcloud", *arguments], 1, "", str(error))


def parse_cloud_document(
    raw: dict[str, Any],
    width: int,
    height: int,
    feature_type: str = "DOCUMENT_TEXT_DETECTION",
) -> CloudOCRDocument:
    paragraphs: list[CloudOCRParagraph] = []
    paragraph_number = 0
    for page_number, page in enumerate(raw.get("pages", []), start=1):
        for block_number, block in enumerate(page.get("blocks", []), start=1):
            for paragraph_number_in_block, paragraph in enumerate(block.get("paragraphs", []), start=1):
                paragraph_number += 1
                prefix = f"p{page_number}-b{block_number}-{paragraph_number_in_block}"
                lines = _paragraph_lines(paragraph, width, height, prefix)
                paragraphs.append(
                    CloudOCRParagraph(
                        id=prefix,
                        detected_language=_detected_language(paragraph),
                        confidence=_confidence(lines),
                        lines=lines,
                    )
                )
    return CloudOCRDocument(
        provider="google-cloud-vision",
        width=width,
        height=height,
        feature_type=feature_type,
        paragraphs=paragraphs,
    )


def parse_text_annotations(
    annotations: list[dict[str, Any]],
    width: int,
    height: int,
    feature_type: str = "TEXT_DETECTION",
) -> CloudOCRDocument:
    tokens = [
        CloudOCRToken(
            text=str(annotation.get("description", "")).strip(),
            quad=_quad(annotation.get("boundingPoly") or {}, width, height),
            confidence=1,
        )
        for annotation in annotations[1:]
        if str(annotation.get("description", "")).strip()
    ]
    lines: list[CloudOCRLine] = []
    pending: list[CloudOCRToken] = []
    line_number = 0
    for token in sorted(tokens, key=lambda value: (value.quad[0].y, value.quad[0].x)):
        token_height = max(1, token.quad[3].y - token.quad[0].y)
        baseline = sum(point.y for point in token.quad) / 4
        if pending:
            pending_baseline = sum(point.y for item in pending for point in item.quad) / (4 * len(pending))
            if abs(baseline - pending_baseline) > token_height * 0.8:
                line_number += 1
                lines.append(_make_line("text-detection", line_number, pending, [" ".join(item.text for item in pending)]))
                pending = []
        pending.append(token)
    if pending:
        line_number += 1
        lines.append(_make_line("text-detection", line_number, pending, [" ".join(item.text for item in pending)]))
    paragraph = CloudOCRParagraph(
        id="text-detection-p1",
        confidence=1,
        lines=lines,
    )
    return CloudOCRDocument(
        provider="google-cloud-vision",
        width=width,
        height=height,
        feature_type=feature_type,
        paragraphs=[paragraph] if lines else [],
    )


def cloud_document_to_analysis(
    document: CloudOCRDocument,
    name: str,
    timings: dict[str, float] | None = None,
    grouping: str = "google-paragraph",
) -> AnalysisResult:
    regions: list[DetectedTextRegion] = []
    groups: list[TextFlowGroup] = []
    next_region_id = 1
    for group_id, paragraph in enumerate(document.paragraphs, start=1):
        flow_lines: list[TextFlowLine] = []
        for line_index, line in enumerate(paragraph.lines, start=1):
            region_id = next_region_id
            next_region_id += 1
            writing_mode = _writing_mode(line.quad)
            text = line.text.strip()
            region = DetectedTextRegion(
                id=region_id,
                quad=line.quad,
                confidence=paragraph.confidence,
                recognition_confidence=min((token.confidence for token in line.tokens), default=paragraph.confidence),
                text=text,
                layout=TextLayout(
                    writing_mode=writing_mode,
                    reading_direction=("left-to-right" if writing_mode == "horizontal" else "top-to-bottom"),
                    script_family=_script_family(text),
                    confidence=paragraph.confidence,
                ),
                group_id=group_id,
                line_index=line_index,
            )
            regions.append(region)
            flow_lines.append(TextFlowLine(index=line_index, region_ids=[region_id], text=text))
        if flow_lines:
            groups.append(
                TextFlowGroup(
                    id=group_id,
                    writing_mode=_writing_mode(paragraph.lines[0].quad),
                    lines=flow_lines,
                    text="\n".join(line.text for line in flow_lines),
                )
            )
    if grouping == "verto-geometry":
        groups = group_regions(regions)
    elif grouping != "google-paragraph":
        raise ValueError(f"不支持的 Google OCR 分组方式：{grouping}")
    stage_timings = timings or {}
    return AnalysisResult(
        image_name=name,
        width=document.width,
        height=document.height,
        detector="google-cloud-vision",
        classifier=f"google-{grouping}-layout",
        recognizer="google-cloud-vision",
        model_versions={"detector": document.feature_type, "classifier": f"{grouping}-1", "recognizer": "Google Cloud Vision"},
        regions=regions,
        groups=groups,
        timings_ms=StageTiming(
            preprocessing=stage_timings.get("preprocessing", 0),
            detection=stage_timings.get("detection", 0),
            layout=stage_timings.get("layout", 0),
            recognition=stage_timings.get("recognition", 0),
            total=stage_timings.get("total", 0),
        ),
        warnings=["区域与文字来自 Google Cloud Vision；最终译图仍由本地内存中的原图一次性合成。"],
    )


def _paragraph_lines(paragraph: dict[str, Any], width: int, height: int, prefix: str) -> list[CloudOCRLine]:
    lines: list[CloudOCRLine] = []
    tokens: list[CloudOCRToken] = []
    text_parts: list[str] = []
    line_number = 0
    for word in paragraph.get("words", []):
        symbols = word.get("symbols", [])
        text = "".join(str(symbol.get("text", "")) for symbol in symbols)
        if not text:
            continue
        token = CloudOCRToken(
            text=text,
            quad=_quad(word.get("boundingBox") or {}, width, height),
            confidence=_word_confidence(word),
        )
        tokens.append(token)
        text_parts.append(text)
        break_type = _word_break_type(word)
        if break_type in {"SPACE", "SURE_SPACE", "EOL_SURE_SPACE"}:
            text_parts.append(" ")
        if break_type in {"EOL_SURE_SPACE", "LINE_BREAK"}:
            line_number += 1
            lines.append(_make_line(prefix, line_number, tokens, text_parts))
            tokens, text_parts = [], []
    if tokens:
        line_number += 1
        lines.append(_make_line(prefix, line_number, tokens, text_parts))
    return lines


def _make_line(prefix: str, line_number: int, tokens: list[CloudOCRToken], text_parts: list[str]) -> CloudOCRLine:
    return CloudOCRLine(id=f"{prefix}-l{line_number}", text="".join(text_parts).strip(), quad=_union_quad([token.quad for token in tokens]), tokens=tokens)


def _word_break_type(word: dict[str, Any]) -> str:
    word_break = ((word.get("property") or {}).get("detectedBreak") or {}).get("type")
    if word_break:
        return str(word_break)
    symbols = word.get("symbols") or []
    if not symbols:
        return ""
    properties = symbols[-1].get("property") or {}
    detected_break = properties.get("detectedBreak") or {}
    return str(detected_break.get("type", ""))


def _word_confidence(word: dict[str, Any]) -> float:
    if word.get("confidence") is not None:
        return _bounded(float(word["confidence"]))
    values = [float(symbol.get("confidence", 0)) for symbol in word.get("symbols", []) if symbol.get("confidence") is not None]
    return _bounded(sum(values) / len(values)) if values else 0


def _detected_language(item: dict[str, Any]) -> str | None:
    languages = ((item.get("property") or {}).get("detectedLanguages") or [])
    if not languages:
        return None
    best = max(languages, key=lambda value: float(value.get("confidence", 0)))
    return best.get("languageCode")


def _confidence(lines: list[CloudOCRLine]) -> float:
    values = [token.confidence for line in lines for token in line.tokens]
    return _bounded(sum(values) / len(values)) if values else 0


def _quad(box: dict[str, Any], width: int, height: int) -> list[Point]:
    values = box.get("vertices") or box.get("normalizedVertices") or []
    points: list[tuple[float, float]] = []
    for value in values:
        x = float(value.get("x", 0))
        y = float(value.get("y", 0))
        if box.get("normalizedVertices"):
            x *= width
            y *= height
        points.append((max(0, min(width, x)), max(0, min(height, y))))
    if len(points) < 4:
        if points:
            xs, ys = zip(*points)
            points = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
        else:
            points = [(0, 0), (width, 0), (width, height), (0, height)]
    return [Point(x=x, y=y) for x, y in _ordered_quad(points[:4])]


def _union_quad(quads: list[list[Point]]) -> list[Point]:
    points = [(point.x, point.y) for quad in quads for point in quad]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rectangle = MultiPoint(points).minimum_rotated_rectangle
    if not rectangle.is_empty and hasattr(rectangle, "exterior"):
        coords = list(rectangle.exterior.coords)[:4]
        if len(coords) == 4:
            return [Point(x=x, y=y) for x, y in _ordered_quad(coords)]
    return [Point(x=x, y=y) for x, y in _ordered_quad(points)]


def _ordered_quad(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) <= 4:
        source = points
    else:
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        source = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    center_x = sum(point[0] for point in source) / len(source)
    center_y = sum(point[1] for point in source) / len(source)
    ordered = sorted(source, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    start = min(range(len(ordered)), key=lambda index: ordered[index][0] + ordered[index][1])
    ordered = ordered[start:] + ordered[:start]
    if len(ordered) == 4:
        cross = (ordered[1][0] - ordered[0][0]) * (ordered[2][1] - ordered[1][1]) - (ordered[1][1] - ordered[0][1]) * (ordered[2][0] - ordered[1][0])
        if cross < 0:
            ordered = [ordered[0], ordered[3], ordered[2], ordered[1]]
    return ordered


def _writing_mode(quad: list[Point]) -> str:
    width = math.hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y)
    height = math.hypot(quad[3].x - quad[0].x, quad[3].y - quad[0].y)
    return "vertical" if height > width * 1.25 else "horizontal"


def _script_family(text: str) -> str:
    values = [ord(character) for character in text]
    if any(0x3040 <= value <= 0x30FF or 0x31F0 <= value <= 0x31FF or 0xFF66 <= value <= 0xFF9D for value in values):
        return "kana"
    if any(0x1100 <= value <= 0x11FF or 0x3130 <= value <= 0x318F or 0xAC00 <= value <= 0xD7AF for value in values):
        return "hangul"
    if any(0x3400 <= value <= 0x4DBF or 0x4E00 <= value <= 0x9FFF or 0xF900 <= value <= 0xFAFF for value in values):
        return "han"
    if any(0x0041 <= value <= 0x024F or 0x1E00 <= value <= 0x1EFF for value in values):
        return "latin"
    return "other"


def google_language_code(value: str) -> str:
    return {"zh-Hans": "zh-CN", "zh-Hant": "zh-TW", "ja": "ja", "en": "en"}.get(value, value)


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, value))


def _google_error(error: object) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error)
