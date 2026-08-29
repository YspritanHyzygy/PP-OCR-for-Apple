#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import onnx
import yaml
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
MODEL_ROOT = ROOT / "ocr_lab" / "models"
TIERS = ("tiny", "small", "medium")
SUPPLEMENTAL_RECOGNIZERS = tuple(LOCK.get("supplementalRecognizers", {}))


def transform(config: dict[str, Any], name: str) -> dict[str, Any]:
    for item in config["PreProcess"]["transform_ops"]:
        if name in item:
            return item[name] or {}
    raise ValueError(f"inference.yml does not contain {name}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(source: dict[str, str]) -> tuple[Path, dict[str, Any]]:
    repository = source["repository"]
    revision = source["revision"]
    onnx_source = Path(hf_hub_download(repository, "inference.onnx", revision=revision))
    config_source = Path(hf_hub_download(repository, "inference.yml", revision=revision))
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    return onnx_source, config


def common_manifest(
    identifier: str,
    label: str,
    kind: str,
    version_prefix: str,
    source: dict[str, str],
    artifact: Path,
    model: onnx.ModelProto,
) -> dict[str, object]:
    repository = source["repository"]
    revision = source["revision"]
    return {
        "$schema": "./model-manifest.schema.json",
        "id": identifier,
        "label": label,
        "kind": kind,
        "version": f"{version_prefix}-{revision[:12]}",
        "source": f"https://huggingface.co/{repository}/tree/{revision}",
        "license": LOCK["license"],
        "artifact": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256(artifact),
        "output_name": model.graph.output[0].name,
    }


def write_manifest(identifier: str, manifest: dict[str, object]) -> Path:
    path = MODEL_ROOT / f"{identifier}.manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {manifest['label']}")
    print(f"manifest {path}")
    return path


def prepare_detector(tier: str) -> Path:
    source = LOCK["tiers"][tier]["detector"]
    onnx_source, config = source_files(source)
    artifact = MODEL_ROOT / f"pp-ocrv6-{tier}-det.onnx"
    shutil.copy2(onnx_source, artifact)
    model = onnx.load(str(artifact), load_external_data=False)
    decoded = transform(config, "DecodeImage")
    normalized = transform(config, "NormalizeImage")
    post = config["PostProcess"]
    if decoded.get("img_mode") != "BGR":
        raise ValueError(f"expected upstream BGR input, got {decoded.get('img_mode')!r}")
    manifest = {
        **common_manifest(
            f"pp-ocrv6-{tier}-det",
            f"PP-OCRv6 {tier.title()} Detector",
            "detector",
            tier,
            source,
            artifact,
            model,
        ),
        "output_contract": "db-probability-map-v1",
        "probability_threshold": float(post["thresh"]),
        "box_score_threshold": float(post["box_thresh"]),
        "unclip_ratio": float(post["unclip_ratio"]),
        "maximum_candidates": int(post["max_candidates"]),
        "input": {
            "name": model.graph.input[0].name,
            "shape": [1, 3, 960, 960],
            "color_order": "BGR",
            "mean": [float(value) for value in normalized["mean"]],
            "std": [float(value) for value in normalized["std"]],
        },
    }
    return write_manifest(str(manifest["id"]), manifest)


def prepare_recognizer(
    identifier: str,
    label: str,
    version_prefix: str,
    source: dict[str, str],
) -> Path:
    onnx_source, config = source_files(source)
    artifact = MODEL_ROOT / f"{identifier}.onnx"
    shutil.copy2(onnx_source, artifact)
    model = onnx.load(str(artifact), load_external_data=False)
    characters = [str(value) for value in config["PostProcess"]["character_dict"]]
    if not characters or any("\n" in character for character in characters):
        raise ValueError("recognizer character dictionary is empty or contains a newline")
    characters_file = MODEL_ROOT / f"{identifier}-charset.txt"
    characters_file.write_text("\n".join(characters) + "\n", encoding="utf-8")
    manifest = {
        **common_manifest(
            identifier,
            label,
            "recognizer",
            version_prefix,
            source,
            artifact,
            model,
        ),
        "output_contract": "ctc-probabilities-v1",
        "characters_file": characters_file.name,
        "characters_sha256": sha256(characters_file),
        "characters_count": len(characters),
        "class_count": len(characters) + 2,
        "input": {
            "name": model.graph.input[0].name,
            "shape": [1, 3, 48, 640],
            "color_order": "BGR",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
    }
    return write_manifest(str(manifest["id"]), manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare locked PP-OCR models for OCR Lab")
    parser.add_argument("--tier", choices=(*TIERS, "all"), default="all")
    parser.add_argument(
        "--component",
        choices=("detector", "recognizer", "all", "none"),
        default="all",
    )
    parser.add_argument(
        "--supplemental",
        choices=(*SUPPLEMENTAL_RECOGNIZERS, "all", "none"),
        default="none",
        help="prepare additional script recognizers independently of PP-OCRv6 tiers",
    )
    args = parser.parse_args()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    tiers = TIERS if args.tier == "all" else (args.tier,)
    components = {
        "all": ("detector", "recognizer"),
        "none": (),
    }.get(args.component, (args.component,))
    for tier in tiers:
        for component in components:
            if component == "detector":
                prepare_detector(tier)
            else:
                source = LOCK["tiers"][tier]["recognizer"]
                prepare_recognizer(
                    f"pp-ocrv6-{tier}-rec",
                    f"PP-OCRv6 {tier.title()} Recognizer",
                    tier,
                    source,
                )
    supplemental = (
        SUPPLEMENTAL_RECOGNIZERS
        if args.supplemental == "all"
        else (() if args.supplemental == "none" else (args.supplemental,))
    )
    for key in supplemental:
        source = LOCK["supplementalRecognizers"][key]
        prepare_recognizer(
            source["id"],
            source["label"],
            key,
            source,
        )


if __name__ == "__main__":
    main()
