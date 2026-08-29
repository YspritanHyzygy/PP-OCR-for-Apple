from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .contracts import ModelOption


@dataclass(frozen=True)
class ONNXDetectorManifest:
    identifier: str
    label: str
    version: str
    source: str
    license: str
    artifact: Path
    sha256: str
    size_bytes: int
    input_name: str
    output_name: str | None
    input_height: int
    input_width: int
    color_order: Literal["RGB", "BGR"]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    probability_threshold: float
    box_score_threshold: float
    unclip_ratio: float
    maximum_candidates: int


@dataclass(frozen=True)
class ONNXRecognizerManifest:
    identifier: str
    label: str
    version: str
    source: str
    license: str
    artifact: Path
    sha256: str
    size_bytes: int
    input_name: str
    output_name: str | None
    input_height: int
    input_width: int
    color_order: Literal["RGB", "BGR"]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    characters_file: Path
    characters_sha256: str
    characters_count: int
    class_count: int


ModelManifest = ONNXDetectorManifest | ONNXRecognizerManifest


@dataclass(frozen=True)
class RegistryEntry:
    option: ModelOption
    manifest: ModelManifest | None = None


class ModelRegistry:
    """Discovers versioned local manifests and verifies each declared artifact."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._entries: dict[str, list[RegistryEntry]] = {}

    def detector_entries(self) -> list[RegistryEntry]:
        return self._entries_for("detector")

    def recognizer_entries(self) -> list[RegistryEntry]:
        return self._entries_for("recognizer")

    def detector(self, identifier: str) -> ONNXDetectorManifest | None:
        manifest = self._available_manifest(identifier, self.detector_entries())
        return manifest if isinstance(manifest, ONNXDetectorManifest) else None

    def recognizer(self, identifier: str) -> ONNXRecognizerManifest | None:
        manifest = self._available_manifest(identifier, self.recognizer_entries())
        return manifest if isinstance(manifest, ONNXRecognizerManifest) else None

    def _entries_for(self, kind: Literal["detector", "recognizer"]) -> list[RegistryEntry]:
        if kind in self._entries:
            return self._entries[kind]
        entries: list[RegistryEntry] = []
        if not self.root.exists():
            self._entries[kind] = entries
            return entries
        for path in sorted(self.root.glob("*.manifest.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if raw.get("kind") != kind:
                continue
            try:
                manifest = (
                    self._load_detector(path, raw)
                    if kind == "detector"
                    else self._load_recognizer(path, raw)
                )
                self._verify_file(manifest.artifact, manifest.size_bytes, manifest.sha256)
                if isinstance(manifest, ONNXRecognizerManifest):
                    self._verify_file(
                        manifest.characters_file,
                        None,
                        manifest.characters_sha256,
                    )
            except (KeyError, TypeError, ValueError, OSError) as error:
                identifier, label, version = self._identity(path, raw)
                entries.append(
                    RegistryEntry(
                        option=ModelOption(
                            id=identifier,
                            label=label,
                            kind=kind,
                            available=False,
                            version=version,
                            detail=f"manifest 无法使用：{error}",
                        )
                    )
                )
                continue
            entries.append(
                RegistryEntry(
                    option=ModelOption(
                        id=manifest.identifier,
                        label=manifest.label,
                        kind=kind,
                        available=True,
                        version=manifest.version,
                        detail=f"ONNX · {manifest.size_bytes / 1024 / 1024:.1f} MB",
                    ),
                    manifest=manifest,
                )
            )
        self._entries[kind] = entries
        return entries

    def _load_detector(
        self,
        path: Path,
        raw: dict[str, object],
    ) -> ONNXDetectorManifest:
        if raw["output_contract"] != "db-probability-map-v1":
            raise ValueError("detector 只接受 db-probability-map-v1")
        common = self._common(path, raw)
        input_contract = self._input(raw)
        probability_threshold = self._probability(raw, "probability_threshold", 0.3)
        box_score_threshold = self._probability(raw, "box_score_threshold", 0.5)
        unclip_ratio = float(raw.get("unclip_ratio", 1.5))
        maximum_candidates = int(raw.get("maximum_candidates", 3000))
        if unclip_ratio <= 0:
            raise ValueError("unclip_ratio 必须大于 0")
        if maximum_candidates <= 0:
            raise ValueError("maximum_candidates 必须大于 0")
        return ONNXDetectorManifest(
            **common,
            **input_contract,
            probability_threshold=probability_threshold,
            box_score_threshold=box_score_threshold,
            unclip_ratio=unclip_ratio,
            maximum_candidates=maximum_candidates,
        )

    def _load_recognizer(
        self,
        path: Path,
        raw: dict[str, object],
    ) -> ONNXRecognizerManifest:
        if raw["output_contract"] != "ctc-probabilities-v1":
            raise ValueError("recognizer 只接受 ctc-probabilities-v1")
        common = self._common(path, raw)
        input_contract = self._input(raw)
        characters_file = self._child_path(path, str(raw["characters_file"]))
        characters_sha256 = self._sha_string(raw["characters_sha256"])
        characters_count = int(raw["characters_count"])
        class_count = int(raw["class_count"])
        if characters_count <= 0 or class_count != characters_count + 2:
            raise ValueError("class_count 必须等于 characters_count + 2")
        characters = self._characters(characters_file)
        if len(characters) != characters_count:
            raise ValueError(
                f"字符表数量不符：manifest={characters_count}, actual={len(characters)}"
            )
        return ONNXRecognizerManifest(
            **common,
            **input_contract,
            characters_file=characters_file,
            characters_sha256=characters_sha256,
            characters_count=characters_count,
            class_count=class_count,
        )

    def _common(self, path: Path, raw: dict[str, object]) -> dict[str, object]:
        identifier = str(raw["id"]).strip()
        label = str(raw.get("label", identifier)).strip()
        version = str(raw["version"]).strip()
        source = str(raw["source"]).strip()
        license_name = str(raw["license"]).strip()
        size_bytes = int(raw["size_bytes"])
        if not all([identifier, label, version, source, license_name]):
            raise ValueError("id、label、version、source 和 license 不能为空")
        if size_bytes <= 0:
            raise ValueError("size_bytes 必须大于 0")
        return {
            "identifier": identifier,
            "label": label,
            "version": version,
            "source": source,
            "license": license_name,
            "artifact": self._child_path(path, str(raw["artifact"])),
            "sha256": self._sha_string(raw["sha256"]),
            "size_bytes": size_bytes,
            "output_name": raw.get("output_name"),
        }

    def _input(self, raw: dict[str, object]) -> dict[str, object]:
        input_contract = raw["input"]
        if not isinstance(input_contract, dict):
            raise ValueError("input 必须是对象")
        shape = input_contract["shape"]
        if not isinstance(shape, list) or len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise ValueError("输入形状必须是 [1, 3, height, width]")
        height, width = int(shape[2]), int(shape[3])
        if height <= 0 or width <= 0:
            raise ValueError("输入高宽必须大于 0")
        mean = tuple(float(value) for value in input_contract["mean"])
        std = tuple(float(value) for value in input_contract["std"])
        if len(mean) != 3 or len(std) != 3 or any(value == 0 for value in std):
            raise ValueError("mean/std 必须各有三个值，std 不能为 0")
        color_order = input_contract.get("color_order", "RGB")
        if color_order not in {"RGB", "BGR"}:
            raise ValueError("color_order 只接受 RGB 或 BGR")
        return {
            "input_name": str(input_contract["name"]),
            "input_height": height,
            "input_width": width,
            "color_order": color_order,
            "mean": mean,
            "std": std,
        }

    @staticmethod
    def _available_manifest(
        identifier: str,
        entries: list[RegistryEntry],
    ) -> ModelManifest | None:
        return next(
            (
                entry.manifest
                for entry in entries
                if entry.option.id == identifier and entry.option.available
            ),
            None,
        )

    @staticmethod
    def _probability(raw: dict[str, object], key: str, default: float) -> float:
        value = float(raw.get(key, default))
        if not 0 <= value <= 1:
            raise ValueError(f"{key} 必须位于 0...1")
        return value

    @staticmethod
    def _child_path(manifest_path: Path, child: str) -> Path:
        resolved = (manifest_path.parent / child).resolve()
        if manifest_path.parent.resolve() not in resolved.parents:
            raise ValueError("artifact 必须位于 manifest 目录内")
        return resolved

    @staticmethod
    def _sha_string(value: object) -> str:
        sha256 = str(value).lower()
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError("sha256 必须是 64 位十六进制")
        return sha256

    @staticmethod
    def _verify_file(path: Path, size_bytes: int | None, expected_sha256: str) -> None:
        if not path.is_file():
            raise ValueError(f"找不到文件 {path.name}")
        if size_bytes is not None and path.stat().st_size != size_bytes:
            raise ValueError(
                f"文件大小不符：manifest={size_bytes}, actual={path.stat().st_size}"
            )
        checksum = hashlib.sha256()
        with path.open("rb") as artifact:
            while chunk := artifact.read(1024 * 1024):
                checksum.update(chunk)
        if checksum.hexdigest() != expected_sha256:
            raise ValueError(f"{path.name} 的 SHA-256 不匹配")

    @staticmethod
    def _characters(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return lines

    @staticmethod
    def _identity(
        path: Path,
        raw: dict[str, object],
    ) -> tuple[str, str, str]:
        identifier = str(raw.get("id", path.stem))
        return identifier, str(raw.get("label", identifier)), str(raw.get("version", ""))
