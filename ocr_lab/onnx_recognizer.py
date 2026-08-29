from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image

from .contracts import DetectedTextRegion
from .model_registry import ONNXRecognizerManifest
from .region_crop import rectify_region


@dataclass(frozen=True)
class RecognitionResult:
    region_id: int
    text: str
    confidence: float


class ONNXCTCRecognizer:
    """Perspective-crops detector regions and greedily decodes PP-OCR CTC output."""

    def __init__(self, manifest: ONNXRecognizerManifest) -> None:
        self.manifest = manifest
        self.characters = self._characters(manifest)
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

    def recognize(
        self,
        image: Image.Image,
        regions: list[DetectedTextRegion],
    ) -> list[RecognitionResult]:
        if not regions:
            return []
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
        prepared = [self._prepare_crop(source, region) for region in regions]
        tensors = np.stack([item[0] for item in prepared])
        used_widths = [item[1] for item in prepared]
        try:
            probabilities = self._run(tensors)
        except Exception:
            probabilities = np.concatenate(
                [self._run(tensor[None]) for tensor in tensors],
                axis=0,
            )
        if probabilities.ndim != 3 or probabilities.shape[0] != len(regions):
            raise ValueError(
                f"recognizer 输出必须是 [batch, steps, classes]，实际为 {list(probabilities.shape)}"
            )
        if probabilities.shape[2] != self.manifest.class_count:
            raise ValueError(
                "recognizer 类别数与字符表不匹配："
                f"manifest={self.manifest.class_count}, actual={probabilities.shape[2]}"
            )
        return [
            self._decode(region.id, probabilities[index], used_widths[index])
            for index, region in enumerate(regions)
        ]

    def _run(self, tensor: np.ndarray) -> np.ndarray:
        output_names = [self.manifest.output_name] if self.manifest.output_name else None
        outputs = self.session.run(
            output_names,
            {self.manifest.input_name: np.ascontiguousarray(tensor, dtype=np.float32)},
        )
        if not outputs:
            raise ValueError("ONNX recognizer 没有输出")
        return np.asarray(outputs[0], dtype=np.float32)

    def _prepare_crop(
        self,
        source: np.ndarray,
        region: DetectedTextRegion,
    ) -> tuple[np.ndarray, int]:
        height = self.manifest.input_height
        width = self.manifest.input_width
        crop, used_width = rectify_region(source, region, height, width)
        crop /= 255
        if self.manifest.color_order == "BGR":
            crop = crop[:, :, ::-1]
        mean = np.asarray(self.manifest.mean, dtype=np.float32)
        std = np.asarray(self.manifest.std, dtype=np.float32)
        crop = (crop - mean) / std

        tensor = np.zeros((3, height, width), dtype=np.float32)
        tensor[:, :, :used_width] = crop.transpose(2, 0, 1)
        return tensor, used_width

    def _decode(
        self,
        region_id: int,
        probabilities: np.ndarray,
        used_width: int,
    ) -> RecognitionResult:
        total_steps, class_count = probabilities.shape
        valid_steps = min(
            total_steps,
            max(1, round(total_steps * used_width / self.manifest.input_width)),
        )
        indices = probabilities[:valid_steps].argmax(axis=1)
        values = probabilities[np.arange(valid_steps), indices]
        output: list[str] = []
        accepted: list[float] = []
        previous = -1
        space_index = len(self.characters) + 1
        for index, value in zip(indices.tolist(), values.tolist()):
            if index == previous:
                continue
            previous = index
            if index == 0:
                continue
            if index == space_index:
                output.append(" ")
            elif 1 <= index <= len(self.characters):
                output.append(self.characters[index - 1])
            else:
                continue
            accepted.append(float(value))
        return RecognitionResult(
            region_id=region_id,
            text="".join(output).strip(),
            confidence=float(np.mean(accepted)) if accepted else 0,
        )

    @staticmethod
    def _characters(manifest: ONNXRecognizerManifest) -> list[str]:
        lines = manifest.characters_file.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return lines
