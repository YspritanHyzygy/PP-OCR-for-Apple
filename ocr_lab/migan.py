from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_IDENTIFIER = "migan-512"
MODEL_VERSION = "migan-v1"
MODEL_SIZE = 512


def configured_migan_model() -> Path | None:
    value = os.environ.get("OCR_LAB_MIGAN_MODEL") or os.environ.get("MIGAN_MODEL_PATH")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.exists() else None


class MIGANInpainting:
    """Development-only Core ML adapter for the fixed 512 MI-GAN contract.

    The model is deliberately supplied by an explicit local path.  The
    candidate checkpoint and converted derivative do not have confirmed
    redistribution rights, so the lab never copies the package into Git or
    silently downloads it.
    """

    identifier = MODEL_IDENTIFIER
    label = "MI-GAN 512（整图 mask）"
    version = MODEL_VERSION

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or configured_migan_model()
        self._model = None
        self._load_error: str | None = None
        self._verification_error: str | None = self._verify_manifest()

    @property
    def available(self) -> bool:
        return (
            self.model_path is not None
            and self._verification_error is None
            and platform.system() == "Darwin"
            and self._coremltools() is not None
        )

    def status(self) -> dict[str, object]:
        if self.model_path is None:
            detail = "未配置本地 .mlpackage；设置 OCR_LAB_MIGAN_MODEL 后重启实验台。"
        elif platform.system() != "Darwin":
            detail = "MI-GAN 适配器使用 macOS Core ML；当前主机不是 macOS。"
        elif self._coremltools() is None:
            detail = "当前 Python 环境缺少 coremltools。"
        elif self._verification_error:
            detail = self._verification_error
        elif self._load_error:
            detail = f"模型加载失败：{self._load_error}"
        else:
            detail = "开发机本地模型；不会自动上传或写入仓库。"
        return {
            "id": self.identifier,
            "label": self.label,
            "version": self.version,
            "available": self.available,
            "path": str(self.model_path) if self.model_path else None,
            "detail": detail,
            "redistribution": "unconfirmed",
            "treeSHA256": package_tree_sha256(self.model_path),
        }

    def inpaint(self, image: Image.Image, known_mask: Image.Image) -> Image.Image:
        model = self._load()
        source = image.convert("RGB").resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.LANCZOS)
        mask = known_mask.convert("L").resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.NEAREST)
        image_array = np.asarray(source, dtype=np.float32) / 255.0
        image_tensor = np.ascontiguousarray(image_array.transpose(2, 0, 1)[None].astype(np.float16))
        mask_tensor = np.ascontiguousarray((np.asarray(mask, dtype=np.float32) / 255.0)[None, None].astype(np.float16))
        result = model.predict({"image": image_tensor, "mask": mask_tensor})
        output = np.asarray(result["output"])
        if output.shape != (1, 3, MODEL_SIZE, MODEL_SIZE) or not np.isfinite(output).all():
            raise ValueError("MI-GAN 输出不是预期的 1×3×512×512 有限张量。")
        pixels = np.clip(output[0].transpose(1, 2, 0) * 255.0, 0, 255).round().astype(np.uint8)
        return Image.fromarray(pixels, mode="RGB")

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.available:
            raise ValueError(self.status()["detail"])
        coremltools = self._coremltools()
        assert coremltools is not None
        try:
            self._model = coremltools.models.MLModel(
                str(self.model_path),
                compute_units=coremltools.ComputeUnit.ALL,
            )
        except Exception as error:
            self._load_error = str(error)
            raise ValueError(f"MI-GAN 模型加载失败：{error}") from error
        return self._model

    def _verify_manifest(self) -> str | None:
        if self.model_path is None:
            return None
        manifest = migan_manifest(self.model_path)
        if manifest is None:
            return "MI-GAN 模型旁边缺少 migan-manifest.json，已拒绝加载。"
        package = manifest.get("modelPackage")
        expected = package.get("treeSHA256") if isinstance(package, dict) else None
        actual = package_tree_sha256(self.model_path)
        if not expected or actual != expected:
            return f"MI-GAN manifest SHA-256 不匹配：expected={expected}, actual={actual}。"
        return None

    @staticmethod
    def _coremltools():
        try:
            import coremltools
        except ImportError:
            return None
        return coremltools


def migan_manifest(model_path: Path | None = None) -> dict[str, object] | None:
    path = model_path or configured_migan_model()
    if path is None:
        return None
    candidate = path.parent / "migan-manifest.json"
    if not candidate.is_file():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def package_tree_sha256(root: Path | None) -> str | None:
    if root is None or not root.exists():
        return None
    digest = hashlib.sha256()
    paths = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
    for path in paths:
        relative = path.name.encode("utf-8") if root.is_file() else path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()
