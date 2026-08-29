from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import DetectedTextRegion, Point, TextLayout
from .region_crop import rectify_region


@dataclass(frozen=True)
class VisionRegionRead:
    region_id: int
    text: str
    confidence: float


class AppleVisionOCR:
    identifier = "apple-vision-ocr"
    label = "Apple Vision OCR"
    version = "system"

    def __init__(self) -> None:
        self.source = Path(__file__).with_name("apple_vision_helper.swift")
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()[:12]
        self.binary = Path(tempfile.gettempdir()) / f"verto-ocr-lab-vision-{digest}"

    @property
    def available(self) -> bool:
        return platform.system() == "Darwin" and shutil.which("xcrun") is not None

    def recognize(self, image: Image.Image) -> list[DetectedTextRegion]:
        raw = self._run([image])[0]
        return self._detected_regions(raw)

    def recognize_regions(
        self,
        image: Image.Image,
        regions: list[DetectedTextRegion],
    ) -> list[VisionRegionRead]:
        if not regions:
            return []
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
        crops = []
        for region in regions:
            crop, _ = rectify_region(source, region, output_height=96, maximum_width=1600)
            crops.append(Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8), mode="RGB"))
        batches = self._run(crops)
        reads: list[VisionRegionRead] = []
        for region, items in zip(regions, batches):
            text = " ".join(str(item["text"]).strip() for item in items if item["text"].strip())
            confidence = min(
                (float(item["confidence"]) for item in items if item["text"].strip()),
                default=0,
            )
            reads.append(VisionRegionRead(region.id, text, confidence))
        return reads

    def _run(self, images: list[Image.Image]) -> list[list[dict[str, object]]]:
        if not self.available:
            raise ValueError("Apple Vision OCR 只在安装 Xcode Command Line Tools 的 macOS 上可用。")
        self._compile()
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, image in enumerate(images):
                path = Path(temporary) / f"crop-{index}.png"
                image.save(path, format="PNG")
                paths.append(str(path))
            completed = subprocess.run(
                [str(self.binary), *paths],
                check=False,
                capture_output=True,
                timeout=60,
            )
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(message or "Apple Vision helper 执行失败。")
        try:
            raw = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("Apple Vision helper 返回了无效 JSON。") from error
        if not isinstance(raw, list) or any(not isinstance(batch, list) for batch in raw):
            raise ValueError("Apple Vision helper 返回了无效批量结果。")
        return raw

    @staticmethod
    def _detected_regions(raw: list[dict[str, object]]) -> list[DetectedTextRegion]:
        return [
            DetectedTextRegion(
                id=index,
                quad=[Point.model_validate(point) for point in item["quad"]],
                confidence=float(item["confidence"]),
                recognition_confidence=float(item["confidence"]),
                text=str(item["text"]),
                layout=TextLayout(),
            )
            for index, item in enumerate(raw, start=1)
        ]

    def _compile(self) -> None:
        if self.binary.is_file():
            return
        completed = subprocess.run(
            ["xcrun", "swiftc", "-O", str(self.source), "-o", str(self.binary)],
            check=False,
            capture_output=True,
            timeout=120,
        )
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(message or "无法编译 Apple Vision helper。")
