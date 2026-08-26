#!/usr/bin/env python3
"""Validate the independent MI-GAN source, license, manifest, and release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_CODE_REVISION = "2381ef9d322caa4f90550f4b7072a6f681efb8c2"
EXPECTED_CHECKPOINT_SHA = "1d6087eee0aac8923ad2606be5d8caeb4824d3e4de331995e420c74e124a466a"
EXPECTED_UPSTREAM_LICENSE_SHA = (
    "674e33456b8d03693b84ab305aa7372d93c91e68b234651b0adad106d518e6eb"
)
EXPECTED_LICENSE_URL = (
    "https://github.com/Picsart-AI-Research/MI-GAN/blob/"
    f"{EXPECTED_CODE_REVISION}/LICENSE"
)
EXPECTED_README_URL = (
    "https://github.com/Picsart-AI-Research/MI-GAN/blob/"
    f"{EXPECTED_CODE_REVISION}/README.md#prepare-environment"
)
# This remains empty until a rights-holder document exists. A release-enabling
# change must pin both its primary URL and exact checked-in bytes in code.
PINNED_RIGHTS_CONFIRMATION: tuple[str, str] | None = None
EXPECTED_INSTRUMENTS_TRACE_FILE = "benchmarks/evidence/migan-coreml.trace.zip"
# Set this only when the raw trace has been reviewed and added at the fixed path.
PINNED_INSTRUMENTS_TRACE_SHA256: str | None = None
EXPECTED_DEVICE_MATRIX_SOURCES = {
    "iphone16pro-migan-benchmark-fresh.json": (
        "b011808cc77045210706f1cc0ed7c2e02048aa24c5609a865ae02116ed802299"
    ),
    "iphone12-xr-migan-benchmark-raw.json": (
        "af5e67906ee7155588b417a950e9bca633ddf28ff0b9406476ec51c537e69c4e"
    ),
}
EXPECTED_DEVICE_MATRIX_ORIGINAL_SOURCES = {
    "iphone12-xr-migan-benchmark-raw.json": (
        "e6cc43f95bc6cbf3ad3684957e6d7fb8f74a10bea642cf31a08b52481c03b631"
    )
}
EXPECTED_INPUTS = [
    {
        "name": "image",
        "shape": [1, 3, 512, 512],
        "dataType": "FLOAT16",
        "range": [0, 1],
    },
    {
        "name": "mask",
        "shape": [1, 1, 512, 512],
        "dataType": "FLOAT16",
        "knownValue": 1,
        "holeValue": 0,
    },
]
EXPECTED_OUTPUT = {
    "name": "output",
    "shape": [1, 3, 512, 512],
    "dataType": "FLOAT32",
    "range": [0, 1],
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_source_lock(lock: dict[str, Any]) -> None:
    if lock.get("schemaVersion") != 1 or lock.get("modelFamily") != "MI-GAN-512-Places2":
        raise ValueError("MI-GAN source lock schema or model family is invalid")
    code = lock["upstreamCode"]
    if code != {
        "repository": "https://github.com/Picsart-AI-Research/MI-GAN.git",
        "revision": EXPECTED_CODE_REVISION,
        "license": "MIT",
        "licenseURL": EXPECTED_LICENSE_URL,
    }:
        raise ValueError("MI-GAN upstream code provenance drifted")

    checkpoint = lock["checkpoint"]
    if checkpoint["fileName"] != "migan_512_places2.pt":
        raise ValueError("MI-GAN checkpoint filename drifted")
    if checkpoint["bytes"] != 29_553_256:
        raise ValueError("MI-GAN checkpoint byte length drifted")
    if checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("MI-GAN checkpoint SHA-256 drifted")
    if not checkpoint["downloadURL"].startswith("https://drive.usercontent.google.com/download?"):
        raise ValueError("MI-GAN checkpoint URL is not the locked official download")
    if checkpoint.get("distributionEvidenceURL") != EXPECTED_README_URL:
        raise ValueError("MI-GAN checkpoint distribution evidence drifted")

    conversion = lock["conversion"]
    expected = {
        "releaseTag": "migan-v1",
        "archiveName": "migan-512-places2-coreml-v1.aar",
        "modelName": "MIGAN512Places2.mlpackage",
        "minimumDeploymentTarget": "iOS17",
        "modelType": "MLProgram",
        "computePrecision": "FLOAT16",
        "computeUnits": "ALL",
        "inputs": EXPECTED_INPUTS,
        "output": EXPECTED_OUTPUT,
        "parityTolerance": {
            "maximumAbsoluteError": 0.09,
            "meanAbsoluteError": 0.01,
        },
    }
    if conversion != expected:
        raise ValueError("MI-GAN conversion contract drifted")


def validate_license_gate(
    lock: dict[str, Any],
    record: dict[str, Any],
    require_release_ready: bool,
    root: Path = ROOT,
) -> None:
    if record.get("schemaVersion") != 1:
        raise ValueError("MI-GAN redistribution record schema is invalid")
    if record.get("checkpointSHA256") != lock["checkpoint"]["sha256"]:
        raise ValueError("redistribution record names a different checkpoint")
    if record.get("status") not in {"unconfirmed", "confirmed"}:
        raise ValueError("redistribution status must be unconfirmed or confirmed")
    scopes = record.get("scopes")
    if not isinstance(scopes, dict) or set(scopes) != {
        "checkpointRedistribution",
        "convertedDerivativeRedistribution",
        "commercialUse",
    }:
        raise ValueError("redistribution scopes are incomplete")

    if record.get("primaryEvidenceURL") != EXPECTED_LICENSE_URL:
        raise ValueError("redistribution record does not pin the upstream MIT license")
    if record.get("supportingEvidenceURLs") != [EXPECTED_README_URL]:
        raise ValueError("redistribution record does not pin the official checkpoint source")
    if record.get("grantor") != "Picsart AI Research (PAIR)":
        raise ValueError("redistribution record grantor drifted")
    evidence_file = record.get("evidenceFile")
    evidence_sha = str(record.get("evidenceSHA256", ""))
    if not evidence_file or not SHA256.fullmatch(evidence_sha):
        raise ValueError("redistribution record lacks checked-in evidence identity")
    evidence = root / str(evidence_file)
    if not evidence.is_file() or digest(evidence) != evidence_sha:
        raise ValueError("checked-in redistribution evidence is missing or changed")

    if record["status"] == "unconfirmed":
        if any(scopes.values()) or any(
            record.get(key) is not None
            for key in (
                "confirmedAt",
                "rightsConfirmationURL",
                "rightsConfirmationFile",
                "rightsConfirmationSHA256",
            )
        ):
            raise ValueError("unconfirmed redistribution record contains confirmed scopes")
    else:
        if not all(scopes.values()) or not record.get("confirmedAt"):
            raise ValueError("confirmed redistribution record is incomplete")
        confirmation_url = str(record.get("rightsConfirmationURL", ""))
        confirmation_file = record.get("rightsConfirmationFile")
        confirmation_sha = str(record.get("rightsConfirmationSHA256", ""))
        if PINNED_RIGHTS_CONFIRMATION is None:
            raise ValueError(
                "release is blocked: validator has no pinned rights-holder confirmation"
            )
        expected_url, expected_sha = PINNED_RIGHTS_CONFIRMATION
        if confirmation_url != expected_url:
            raise ValueError("rights-holder confirmation URL differs from the pinned source")
        if not confirmation_file or not SHA256.fullmatch(confirmation_sha):
            raise ValueError("confirmed redistribution record lacks rights-holder confirmation file")
        if confirmation_sha != expected_sha:
            raise ValueError("rights-holder confirmation SHA differs from the pinned bytes")
        confirmation = root / str(confirmation_file)
        if not confirmation.is_file() or digest(confirmation) != confirmation_sha:
            raise ValueError("rights-holder confirmation is missing or changed")

    if require_release_ready and record["status"] != "confirmed":
        raise ValueError(
            "release is blocked: checkpoint, converted-derivative, and commercial redistribution "
            "rights are unconfirmed"
        )


def validate_benchmark(
    report: dict[str, Any],
    require_release_ready: bool,
    expected_model_sha256: str | None = None,
    root: Path = ROOT,
) -> None:
    if report.get("schemaVersion") != 1 or report.get("releaseTag") != "migan-v1":
        raise ValueError("MI-GAN benchmark schema or release tag is invalid")
    if report.get("computeUnits") != "ALL":
        raise ValueError("MI-GAN benchmark must record computeUnits ALL")
    if report.get("warmupRuns") != 1 or report.get("measuredRuns") != 3:
        raise ValueError("MI-GAN benchmark must record one warm-up and three measured runs")
    if report.get("status") not in {"not-run", "complete"}:
        raise ValueError("MI-GAN benchmark status is invalid")
    if report["status"] == "not-run":
        if require_release_ready:
            raise ValueError("release is blocked: physical-device MI-GAN benchmark is incomplete")
        return
    for key in (
        "modelTreeSHA256",
        "device",
        "os",
        "thermalState",
        "coldLoadMilliseconds",
        "medianPredictionMilliseconds",
        "observedResidentHighWaterBytes",
    ):
        if report.get(key) is None:
            raise ValueError(f"benchmark lacks {key}")
    if not SHA256.fullmatch(str(report["modelTreeSHA256"])):
        raise ValueError("benchmark model SHA-256 is invalid")
    if expected_model_sha256 is not None and report["modelTreeSHA256"] != expected_model_sha256:
        raise ValueError("benchmark model SHA-256 differs from the manifest")
    if require_release_ready and expected_model_sha256 is None:
        raise ValueError("release is blocked: benchmark validation requires the built manifest")
    times = report.get("predictionMilliseconds")
    if not isinstance(times, list) or len(times) != 3 or not all(
        isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        for value in times
    ):
        raise ValueError("benchmark must contain three finite prediction times")
    if report.get("measurementScope") != "model.prediction":
        raise ValueError("benchmark measurement scope drifted")
    for key in (
        "coldLoadMilliseconds",
        "medianPredictionMilliseconds",
        "observedResidentHighWaterBytes",
    ):
        value = report[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"benchmark {key} is invalid")
    if not math.isclose(
        report["medianPredictionMilliseconds"], statistics.median(times), rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("benchmark median does not match the prediction samples")
    instruments = report.get("coreMLInstruments", {})
    if instruments.get("status") not in {"model-named-ane-observed", "confirmed"}:
        raise ValueError("Core ML Instruments evidence status is invalid")
    if not instruments.get("executionDevice") or not SHA256.fullmatch(
        str(instruments.get("traceSHA256", ""))
    ):
        raise ValueError("Instruments evidence is incomplete")
    if require_release_ready and instruments.get("status") != "confirmed":
        raise ValueError("release is blocked: Core ML Instruments execution device is unconfirmed")
    if instruments.get("status") == "confirmed":
        if PINNED_INSTRUMENTS_TRACE_SHA256 is None:
            raise ValueError("release is blocked: validator has no pinned Instruments trace")
        if instruments.get("computeUnits") != "ALL":
            raise ValueError("confirmed Instruments evidence does not bind computeUnits ALL")
        if instruments.get("modelTreeSHA256") != report["modelTreeSHA256"]:
            raise ValueError("confirmed Instruments evidence does not bind the benchmark model")
        if instruments.get("traceFile") != EXPECTED_INSTRUMENTS_TRACE_FILE:
            raise ValueError("confirmed Instruments evidence does not name the pinned trace file")
        trace = root / EXPECTED_INSTRUMENTS_TRACE_FILE
        if instruments["traceSHA256"] != PINNED_INSTRUMENTS_TRACE_SHA256:
            raise ValueError("confirmed Instruments trace SHA differs from the pinned bytes")
        if not trace.is_file() or digest(trace) != PINNED_INSTRUMENTS_TRACE_SHA256:
            raise ValueError("confirmed Instruments trace bytes are missing or changed")


def validate_device_matrix(
    matrix: dict[str, Any],
    expected_model_tree_sha256: str | None = None,
    root: Path = ROOT,
) -> None:
    if matrix.get("schemaVersion") != 1 or matrix.get("releaseTag") != "migan-v1":
        raise ValueError("MI-GAN device matrix schema or release tag is invalid")
    if not SHA256.fullmatch(str(matrix.get("modelTreeSHA256", ""))):
        raise ValueError("MI-GAN device matrix model tree SHA-256 is invalid")
    if (
        expected_model_tree_sha256 is not None
        and matrix["modelTreeSHA256"] != expected_model_tree_sha256
    ):
        raise ValueError("MI-GAN device matrix model tree SHA-256 drifted")
    sources = matrix.get("sourceArtifacts")
    if not isinstance(sources, list):
        raise ValueError("MI-GAN device matrix source artifacts are missing")
    source_map = {item.get("name"): item.get("sha256") for item in sources}
    if source_map != EXPECTED_DEVICE_MATRIX_SOURCES:
        raise ValueError("MI-GAN device matrix source artifacts drifted")
    for item in sources:
        expected_file = f"benchmarks/evidence/{item['name']}"
        if item.get("file") != expected_file:
            raise ValueError("MI-GAN device matrix source path drifted")
        source = root / expected_file
        if not source.is_file() or digest(source) != item["sha256"]:
            raise ValueError("MI-GAN device matrix source bytes are missing or changed")
        original_sha = EXPECTED_DEVICE_MATRIX_ORIGINAL_SOURCES.get(item["name"])
        if original_sha is not None and item.get("sourceSHA256") != original_sha:
            raise ValueError("MI-GAN device matrix original source identity drifted")

    devices = matrix.get("devices")
    if not isinstance(devices, list):
        raise ValueError("MI-GAN device matrix devices are missing")
    by_product = {item.get("productType"): item for item in devices}
    if set(by_product) != {"iPhone17,1", "iPhone13,2", "iPhone11,8"}:
        raise ValueError("MI-GAN device matrix device set drifted")
    for product, expected_decision in {
        "iPhone17,1": "fixed-fixture-pass",
        "iPhone13,2": "fixed-fixture-pass",
        "iPhone11,8": "fixed-fixture-reject",
    }.items():
        item = by_product[product]
        if item.get("anePathDecision") != expected_decision:
            raise ValueError(f"MI-GAN device matrix decision drifted for {product}")
        metrics = item.get("inpaintRegionAllVsGPU", {})
        for key in ("meanAbsoluteError", "maximumAbsoluteError", "fractionAbove0_05"):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"MI-GAN device matrix metric {key} is invalid")
    xr = by_product["iPhone11,8"]
    if xr.get("instrumentsEvidence") != "raw-capture-unfinalized":
        raise ValueError("MI-GAN XR evidence status drifted")
    if "executionDevice" in xr or "traceSHA256" in xr:
        raise ValueError("MI-GAN XR record overclaims unfinalized Instruments evidence")

    gate = matrix.get("provisionalCorrectnessGate")
    if gate != {
        "maximumMeanAbsoluteError": 0.01,
        "maximumFractionAbove0_05": 0.01,
    }:
        raise ValueError("MI-GAN provisional correctness gate drifted")
    for product in ("iPhone17,1", "iPhone13,2"):
        metrics = by_product[product]["inpaintRegionAllVsGPU"]
        if (
            metrics["meanAbsoluteError"] > gate["maximumMeanAbsoluteError"]
            or metrics["fractionAbove0_05"] > gate["maximumFractionAbove0_05"]
        ):
            raise ValueError(f"MI-GAN passing decision contradicts metrics for {product}")
    xr_metrics = xr["inpaintRegionAllVsGPU"]
    if (
        xr_metrics["meanAbsoluteError"] <= gate["maximumMeanAbsoluteError"]
        and xr_metrics["fractionAbove0_05"] <= gate["maximumFractionAbove0_05"]
    ):
        raise ValueError("MI-GAN XR rejection contradicts its metrics")


def validate_manifest(
    manifest: dict[str, Any],
    lock: dict[str, Any],
    license_record: dict[str, Any],
    assets: Path | None,
) -> None:
    if manifest.get("schemaVersion") != 1 or manifest.get("packVersion") != "1":
        raise ValueError("MI-GAN manifest schema or pack version is invalid")
    if manifest.get("releaseTag") != "migan-v1":
        raise ValueError("MI-GAN manifest release tag is invalid")
    if manifest.get("licenseGate") != license_record:
        raise ValueError("MI-GAN manifest license record drifted")
    if manifest.get("source") != {
        "upstreamCode": lock["upstreamCode"],
        "checkpoint": lock["checkpoint"],
    }:
        raise ValueError("MI-GAN manifest source does not match the lock")
    conversion = manifest.get("conversion", {})
    locked = lock["conversion"]
    for key in (
        "modelName",
        "minimumDeploymentTarget",
        "modelType",
        "computePrecision",
        "computeUnits",
        "inputs",
        "output",
    ):
        if conversion.get(key) != locked[key]:
            raise ValueError(f"MI-GAN manifest conversion field {key} drifted")

    parity = manifest.get("parity", {})
    if not isinstance(parity.get("fixtures"), list) or not parity["fixtures"]:
        raise ValueError("MI-GAN manifest has no parity fixtures")
    for fixture in parity["fixtures"]:
        if not SHA256.fullmatch(str(fixture.get("inputSHA256", ""))):
            raise ValueError("MI-GAN parity fixture hash is invalid")
        if fixture.get("maximumAbsoluteError", float("inf")) > locked["parityTolerance"][
            "maximumAbsoluteError"
        ]:
            raise ValueError("MI-GAN maximum parity error exceeds the locked tolerance")
        if fixture.get("meanAbsoluteError", float("inf")) > locked["parityTolerance"][
            "meanAbsoluteError"
        ]:
            raise ValueError("MI-GAN mean parity error exceeds the locked tolerance")

    model_package = manifest.get("modelPackage", {})
    if not isinstance(model_package.get("bytes"), int) or model_package["bytes"] <= 0:
        raise ValueError("MI-GAN model package byte length is invalid")
    if not SHA256.fullmatch(str(model_package.get("treeSHA256", ""))):
        raise ValueError("MI-GAN model package tree SHA-256 is invalid")
    if manifest.get("provenance") != {
        "schemaVersion": 1,
        "upstreamCodeRevision": EXPECTED_CODE_REVISION,
        "checkpointSHA256": EXPECTED_CHECKPOINT_SHA,
        "modelPackage": model_package,
        "evidence": {
            "upstreamLicenseSHA256": EXPECTED_UPSTREAM_LICENSE_SHA,
            "redistributionEvidenceSHA256": license_record["evidenceSHA256"],
        },
    }:
        raise ValueError("MI-GAN manifest provenance drifted")

    archive = manifest.get("archive", {})
    if archive.get("name") != locked["archiveName"]:
        raise ValueError("MI-GAN archive name drifted")
    if not isinstance(archive.get("bytes"), int) or archive["bytes"] <= 0:
        raise ValueError("MI-GAN archive byte length is invalid")
    if not SHA256.fullmatch(str(archive.get("sha256", ""))):
        raise ValueError("MI-GAN archive SHA-256 is invalid")
    if assets is None:
        return
    archive_path = assets / archive["name"]
    if archive_path.stat().st_size != archive["bytes"] or digest(archive_path) != archive["sha256"]:
        raise ValueError("MI-GAN archive does not match its manifest")
    with tempfile.TemporaryDirectory() as directory:
        extracted = Path(directory)
        subprocess.run(
            ["aa", "extract", "-i", str(archive_path), "-d", str(extracted)],
            check=True,
            capture_output=True,
            text=True,
        )
        required = (
            locked["modelName"],
            "MODEL_CARD.md",
            "NOTICE",
            "redistribution-license.json",
            "PAIR-MIT-WEIGHTS-EVIDENCE.md",
            "MI-GAN-MIT.txt",
            "PROVENANCE.json",
        )
        for relative in required:
            if not (extracted / relative).exists():
                raise ValueError(f"MI-GAN archive lacks {relative}")
        if load_json(extracted / "redistribution-license.json") != license_record:
            raise ValueError("MI-GAN archive redistribution record drifted")
        if digest(extracted / "PAIR-MIT-WEIGHTS-EVIDENCE.md") != license_record[
            "evidenceSHA256"
        ]:
            raise ValueError("MI-GAN archive evidence file drifted")
        if digest(extracted / "MI-GAN-MIT.txt") != EXPECTED_UPSTREAM_LICENSE_SHA:
            raise ValueError("MI-GAN archive MIT license drifted")
        if load_json(extracted / "PROVENANCE.json") != manifest["provenance"]:
            raise ValueError("MI-GAN archive provenance drifted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--require-release-ready", action="store_true")
    args = parser.parse_args()

    lock = load_json(ROOT / "migan/sources.lock.json")
    license_record = load_json(ROOT / "migan/redistribution-license.json")
    benchmark = load_json(ROOT / "benchmarks/migan-v1.json")
    device_matrix = load_json(ROOT / "benchmarks/migan-device-matrix-v1.json")
    manifest = load_json(args.manifest) if args.manifest else None
    validate_source_lock(lock)
    validate_license_gate(lock, license_record, args.require_release_ready)
    validate_device_matrix(device_matrix, benchmark.get("modelTreeSHA256"), ROOT)
    validate_benchmark(
        benchmark,
        args.require_release_ready,
        manifest.get("modelPackage", {}).get("treeSHA256") if manifest else None,
        ROOT,
    )
    if manifest:
        validate_manifest(manifest, lock, license_record, args.assets)
    elif args.assets:
        raise ValueError("--assets requires --manifest")
    print("MI-GAN metadata is valid.")


if __name__ == "__main__":
    try:
        main()
    except (
        AssertionError,
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        detail = str(error) or "a validation constraint failed"
        print(f"MI-GAN metadata is invalid: {detail}", file=sys.stderr)
        raise SystemExit(1) from None
