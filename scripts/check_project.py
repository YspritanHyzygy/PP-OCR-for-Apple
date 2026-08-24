#!/usr/bin/env python3
"""Validate source locks, release manifests, and optional local assets."""

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TIERS = ("tiny", "small", "medium")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sources(source_lock: dict) -> None:
    assert source_lock["schemaVersion"] == 1
    assert source_lock["modelFamily"] == "PP-OCRv6"
    assert source_lock["license"] == "Apache-2.0"
    assert tuple(source_lock["tiers"]) == TIERS
    for tier in TIERS:
        for component, suffix in (("detector", "det_onnx"), ("recognizer", "rec_onnx")):
            source = source_lock["tiers"][tier][component]
            assert source["repository"] == f"PaddlePaddle/PP-OCRv6_{tier}_{suffix}"
            assert REVISION.fullmatch(source["revision"])


def validate_manifest(manifest: dict, source_lock: dict, assets: Path | None) -> None:
    assert manifest["schemaVersion"] == 1
    version = manifest["packVersion"]
    assert version.isdigit()
    assert manifest["releaseTag"] == f"v{version}"
    assert manifest["upstream"] == {
        "project": "PaddleOCR",
        "modelFamily": "PP-OCRv6",
        "license": "Apache-2.0",
    }
    packs = manifest["packs"]
    assert tuple(pack["tier"] for pack in packs) == TIERS
    for pack in packs:
        tier = pack["tier"]
        assert pack["source"] == source_lock["tiers"][tier]
        archive = pack["archive"]
        assert archive["name"] == f"pp-ocr-v6-coreml-{tier}-v{version}.aar"
        assert archive["bytes"] > 0
        assert SHA256.fullmatch(archive["sha256"])
        errors = pack["conversionMaxAbsoluteError"]
        assert 0 <= errors["detector"] <= manifest["detector"]["conversionTolerance"]
        assert 0 <= errors["recognizer"] <= manifest["recognizer"]["conversionTolerance"]
        assert pack["recognizerArgmaxMismatches"] >= 0
        if assets is not None:
            path = assets / archive["name"]
            assert path.stat().st_size == archive["bytes"]
            assert digest(path) == archive["sha256"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()

    source_lock = load_json(ROOT / "sources.lock.json")
    validate_sources(source_lock)

    manifest_path = args.manifest
    if manifest_path is None and (ROOT / "manifests/v1.json").exists():
        manifest_path = ROOT / "manifests/v1.json"
    if manifest_path is not None:
        validate_manifest(load_json(manifest_path), source_lock, args.assets)

    print("Project metadata is valid.")


if __name__ == "__main__":
    main()
