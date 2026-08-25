#!/usr/bin/env python3
"""Validate source locks, build recipes, manifests, and optional assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TIERS = ("tiny", "small", "medium")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
COMPRESSIONS = {"none", "int8", "palette8", "palette6", "w8a8"}
IO_TYPES = {"FLOAT32", "FLOAT16"}
RECIPE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sources(source_lock: dict[str, Any]) -> None:
    assert source_lock["schemaVersion"] == 1
    assert source_lock["modelFamily"] == "PP-OCRv6"
    assert source_lock["license"] == "Apache-2.0"
    assert tuple(source_lock["tiers"]) == TIERS
    for tier in TIERS:
        for component, suffix in (("detector", "det_onnx"), ("recognizer", "rec_onnx")):
            source = source_lock["tiers"][tier][component]
            assert source["repository"] == f"PaddlePaddle/PP-OCRv6_{tier}_{suffix}"
            assert REVISION.fullmatch(source["revision"])


def validate_recipe(recipe: dict[str, Any]) -> None:
    assert recipe["schemaVersion"] == 1
    assert RECIPE_NAME.fullmatch(recipe["name"])
    assert recipe["packVersion"].isdigit()
    assert recipe["minimumDeploymentTarget"] == "iOS17"
    assert recipe["computeUnits"] == "ALL"
    assert recipe["colorOrder"] == "RGB"
    assert tuple(recipe["tiers"]) == TIERS
    for config in recipe["tiers"].values():
        assert config["detectorSourceTier"] in TIERS
        assert config["recognizerSourceTier"] in TIERS
        assert config["detectorInputSize"] in {640, 736, 960}
        assert config["recognizerInputHeight"] == 48
        assert config["recognizerInputWidth"] in {320, 640}
        assert config["detectorCompression"] in COMPRESSIONS
        assert config["recognizerCompression"] in COMPRESSIONS
        assert config["detectorIOType"] in IO_TYPES
        assert config["recognizerIOType"] in IO_TYPES
        for component in ("detector", "recognizer"):
            if config[f"{component}Compression"] == "w8a8":
                assert config[f"{component}IOType"] == "FLOAT32"


def validate_v1_manifest(
    manifest: dict[str, Any], source_lock: dict[str, Any], assets: Path | None
) -> None:
    assert manifest["schemaVersion"] == 1
    packs = manifest["packs"]
    assert tuple(pack["tier"] for pack in packs) == TIERS
    for pack in packs:
        tier = pack["tier"]
        assert pack["source"] == source_lock["tiers"][tier]
        errors = pack["conversionMaxAbsoluteError"]
        assert 0 <= errors["detector"] <= manifest["detector"]["conversionTolerance"]
        assert 0 <= errors["recognizer"] <= manifest["recognizer"]["conversionTolerance"]
        assert pack["recognizerArgmaxMismatches"] >= 0
        validate_archive(pack["archive"], tier, manifest["packVersion"], assets)


def validate_v2_manifest(
    manifest: dict[str, Any], source_lock: dict[str, Any], assets: Path | None,
    require_all_tiers: bool,
) -> None:
    assert manifest["schemaVersion"] == 2
    recipe = manifest["recipe"]
    assert recipe["name"]
    assert recipe["file"].endswith(".json")
    assert SHA256.fullmatch(recipe["sha256"])
    assert manifest["conversion"] == {
        "minimumDeploymentTarget": "iOS17",
        "computeUnits": "ALL",
        "archiveFormat": "AppleArchive",
        "compression": "lzfse",
    }
    manifest_tiers = tuple(pack["tier"] for pack in manifest["packs"])
    assert manifest_tiers
    assert len(set(manifest_tiers)) == len(manifest_tiers)
    assert manifest_tiers == tuple(tier for tier in TIERS if tier in manifest_tiers)
    if require_all_tiers:
        assert manifest_tiers == TIERS
    for pack in manifest["packs"]:
        tier = pack["tier"]
        for component in ("detector", "recognizer"):
            item = pack[component]
            source_tier = item["sourceTier"]
            assert source_tier in TIERS
            assert item["source"] == source_lock["tiers"][source_tier][component]
            assert item["colorOrder"] == "RGB"
            assert item["inputShape"][:2] == [1, 3]
            assert item["inputOutputType"] in IO_TYPES
            assert item["computePrecision"] == "FLOAT16"
            assert item["weightCompression"] in COMPRESSIONS
            assert item.get("activationQuantization", "none") in {"none", "int8"}
            assert item.get("calibrationSamples", 0) >= 0
            if item.get("activationQuantization") == "int8":
                assert item["weightCompression"] == "int8"
                assert item["calibrationSamples"] > 0
            assert item["rgbFoldMaxAbsoluteError"] >= 0
            assert item["conversionMaxAbsoluteError"] >= 0
        post = pack["detector"]["postProcess"]
        expected_box_threshold = 0.4 if pack["detector"]["sourceTier"] == "tiny" else 0.45
        assert post["binarizationThreshold"] == 0.2
        assert post["boxScoreThreshold"] == expected_box_threshold
        assert post["unclipRatio"] == 1.4
        assert post["maximumCandidates"] == 3000
        assert pack["recognizer"]["charactersCount"] > 0
        assert pack["recognizer"]["argmaxMismatches"] >= 0
        validate_archive(pack["archive"], tier, manifest["packVersion"], assets)
    uses_activation_quantization = any(
        pack[component].get("activationQuantization") == "int8"
        for pack in manifest["packs"] for component in ("detector", "recognizer")
    )
    calibration = manifest.get("activationCalibration")
    if uses_activation_quantization:
        assert calibration["corpusFile"].endswith(".json")
        assert SHA256.fullmatch(calibration["sha256"])
        assert calibration["maximumSamplesPerComponent"] > 0
    else:
        assert calibration is None


def validate_archive(archive: dict[str, Any], tier: str, version: str, assets: Path | None) -> None:
    assert archive["name"] == f"pp-ocr-v6-coreml-{tier}-v{version}.aar"
    assert archive["bytes"] > 0
    assert SHA256.fullmatch(archive["sha256"])
    if assets is not None:
        path = assets / archive["name"]
        assert path.stat().st_size == archive["bytes"]
        assert digest(path) == archive["sha256"]


def validate_manifest(
    manifest: dict[str, Any], source_lock: dict[str, Any], assets: Path | None,
    require_all_tiers: bool,
) -> None:
    version = manifest["packVersion"]
    assert version.isdigit()
    assert manifest["releaseTag"] == f"v{version}"
    assert manifest["upstream"] == {
        "project": "PaddleOCR",
        "modelFamily": "PP-OCRv6",
        "license": "Apache-2.0",
    }
    if manifest["schemaVersion"] == 1:
        validate_v1_manifest(manifest, source_lock, assets)
    elif manifest["schemaVersion"] == 2:
        validate_v2_manifest(manifest, source_lock, assets, require_all_tiers)
    else:
        raise AssertionError("unsupported manifest schema")


def validate_corpus_lock(corpus: dict[str, Any], require_ready: bool) -> None:
    assert corpus["schemaVersion"] == 1
    assert corpus["supportedLanguages"] == ["en", "zh-Hans", "ja", "fr", "es", "de"]
    assert len(corpus["publicDatasets"]) == 4
    for dataset in corpus["publicDatasets"]:
        assert dataset["source"].startswith("https://")
        checksum = dataset["archiveSHA256"]
        assert checksum is None or SHA256.fullmatch(checksum)
    holdout = corpus["privateFinalHoldout"]
    assert holdout["images"] == 36
    assert holdout["imagesPerLanguage"] == 6
    assert len(holdout["scenarios"]) == 6
    assert holdout["committed"] is False
    assert holdout["use"] == "finalists-only"
    if require_ready:
        blockers = list(corpus["blockers"])
        missing_hashes = [
            dataset["id"] for dataset in corpus["publicDatasets"]
            if not dataset["archiveSHA256"]
        ]
        if corpus["status"] != "ready" or blockers or missing_hashes:
            details = blockers + [f"Missing archive SHA-256: {item}" for item in missing_hashes]
            raise ValueError("release is blocked: " + "; ".join(details))


def validate_final_report(report: dict[str, Any]) -> None:
    assert report["schemaVersion"] == 1
    assert report["releaseTag"] == "v2"
    assert report["decision"] == "pass"
    holdout = report["privateHoldout"]
    assert holdout["images"] == 36
    assert holdout["finalistsOnly"] is True
    assert holdout["usedForTuning"] is False
    assert holdout["newCrashes"] == 0
    assert holdout["newEmptyResults"] == 0
    assert holdout["newMissingLines"] == 0
    iphone = report["iphone16Pro"]
    assert iphone["computeUnits"] == "ALL"
    assert iphone["warmupRuns"] == 3
    assert iphone["thermalState"]
    compatibility = report["iOS17Compatibility"]
    assert all(compatibility[key] == "pass" for key in ("compile", "load", "inference"))
    tiers = report["tiers"]
    assert tuple(tiers) == TIERS
    assert all(tiers[tier]["qualityGate"] == "pass" for tier in TIERS)
    assert tiers["small"]["substantiveImprovement"] is True


def validate_ane_compatibility_status(report: dict[str, Any], require_ready: bool) -> None:
    assert report["schemaVersion"] == 1
    assert report["kind"] == "apple-neural-engine-compatibility-status"
    assert report["releaseTag"] == "v2"
    assert report["status"] in {"pendingPhysicalEvidence", "measured"}
    assert report["productionComputeUnits"] in {"all", "cpuAndNeuralEngine"}
    matrix = report["deviceMatrix"]
    assert [item["hardwareGroup"] for item in matrix] == [
        "A12-A13", "A14-A16", "A17Pro-A19"
    ]
    assert all(item["evidenceStatus"] in {"notRecorded", "pass", "fail"} for item in matrix)
    assert report["candidates"] == ["B1", "small-rec320", "small-rec320-w8a8"]
    assert set(report["candidateStatus"]) == set(report["candidates"])
    if any(item["evidenceStatus"] == "notRecorded" for item in matrix):
        assert report["productionComputeUnits"] == "all"
        assert report["releaseBlocked"] is True
        assert report["currentDecision"] == {
            "userFacingNpuSwitch": False,
            "chipModelAllowlist": False,
            "hardwareSpecificPackages": False,
            "reason": "No complete same-device A12, A16, and A18 Pro comparison exists.",
        }
    if require_ready:
        assert report["status"] == "measured"
        assert all(item["evidenceStatus"] == "pass" for item in matrix)
        assert report["releaseBlocked"] is False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--recipe", type=Path, default=ROOT / "recipes/v2-corrected-reference.json")
    parser.add_argument("--require-release-ready", action="store_true")
    args = parser.parse_args()

    source_lock = load_json(ROOT / "sources.lock.json")
    validate_sources(source_lock)
    recipe = load_json(args.recipe)
    validate_recipe(recipe)
    for candidate_path in sorted((ROOT / "recipes/candidates").glob("*.json")):
        validate_recipe(load_json(candidate_path))
    if args.require_release_ready and recipe["packVersion"] != "2":
        raise ValueError("release is blocked: the selected recipe packVersion is not 2")
    validate_corpus_lock(
        load_json(ROOT / "evaluation/corpus.lock.json"), args.require_release_ready
    )
    validate_ane_compatibility_status(
        load_json(ROOT / "benchmarks/ane-compatibility-v2.json"), args.require_release_ready
    )
    if args.require_release_ready:
        final_report = ROOT / "benchmarks/v2-final.json"
        if not final_report.exists():
            raise ValueError("release is blocked: benchmarks/v2-final.json is missing")
        validate_final_report(load_json(final_report))

    manifest_path = args.manifest
    if manifest_path is None and (ROOT / "manifests/v1.json").exists():
        manifest_path = ROOT / "manifests/v1.json"
    if manifest_path is not None:
        validate_manifest(
            load_json(manifest_path), source_lock, args.assets,
            require_all_tiers=args.require_release_ready,
        )

    print("Project metadata is valid.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, FileNotFoundError, KeyError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        detail = str(error) or "a validation constraint failed"
        print(f"Project metadata is invalid: {detail}", file=sys.stderr)
        raise SystemExit(1) from None
