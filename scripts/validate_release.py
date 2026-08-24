#!/usr/bin/env python3
"""Validate migrated archives against pinned ONNX sources and write v1 metadata."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
TIERS = ("tiny", "small", "medium")
OFFICIAL_WAVG = {"tiny": 73.5, "small": 81.3, "medium": 83.2}
PACK_VERSION = "1"
DET_SHAPE = (1, 3, 960, 960)
REC_SHAPE = (1, 3, 48, 640)
DET_TOLERANCE = 0.011
REC_TOLERANCE = 0.30


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def machine_context() -> dict:
    hardware = json.loads(subprocess.run(
        ["system_profiler", "SPHardwareDataType", "-json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout)["SPHardwareDataType"][0]
    return {
        "machine": platform.machine(),
        "device": hardware["machine_name"],
        "modelIdentifier": hardware["machine_model"],
        "chip": hardware["chip_type"],
        "memory": hardware["physical_memory"],
        "macOS": platform.mac_ver()[0],
        "coremltools": ct.__version__,
        "onnxruntime": ort.__version__,
        "computeUnits": "ALL",
    }


def source_directory(source: dict, destination: Path) -> Path:
    snapshot_download(
        source["repository"],
        revision=source["revision"],
        local_dir=str(destination),
    )
    return destination


def compare_and_time(
    onnx_path: Path,
    package_path: Path,
    shape: tuple[int, ...],
    tolerance: float,
    runs: int,
    require_argmax_equal: bool = False,
) -> tuple[float, int, dict]:
    sample = np.random.default_rng(1).random(shape).astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    reference = session.run(None, {session.get_inputs()[0].name: sample})[0]

    load_started = time.perf_counter()
    model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.ALL)
    load_ms = (time.perf_counter() - load_started) * 1000
    output_name = next(iter(model.predict({"x": sample})))

    timings = []
    result = None
    for _ in range(runs):
        started = time.perf_counter()
        result = model.predict({"x": sample})[output_name]
        timings.append((time.perf_counter() - started) * 1000)
    assert result is not None
    delta = float(np.abs(reference - result).max())
    if delta > tolerance:
        raise SystemExit(f"{package_path.name}: conversion error {delta} exceeds {tolerance}")
    argmax_mismatches = 0
    if require_argmax_equal:
        argmax_mismatches = int(
            np.count_nonzero(np.argmax(reference, axis=-1) != np.argmax(result, axis=-1))
        )
    return delta, argmax_mismatches, {
        "loadMilliseconds": load_ms,
        "warmupRuns": 1,
        "timedRuns": runs,
        "predictionMedianMilliseconds": statistics.median(timings),
        "predictionP95Milliseconds": percentile95(timings),
    }


def validate_tier(tier: str, assets: Path, work: Path, source_lock: dict, runs: int) -> tuple[dict, dict]:
    archive_name = f"pp-ocr-v6-coreml-{tier}-v{PACK_VERSION}.aar"
    archive = assets / archive_name
    unpacked = work / tier / "unpacked"
    unpacked.mkdir(parents=True)
    subprocess.run(["aa", "extract", "-i", str(archive), "-d", str(unpacked)], check=True)

    detector = unpacked / "VertoTextDetector.mlpackage"
    recognizer = unpacked / "VertoTextRecognizer.mlpackage"
    charset = unpacked / "charset.txt"
    for required in (detector, recognizer, charset):
        if not required.exists():
            raise SystemExit(f"{archive.name}: missing {required.name}")

    sources = source_lock["tiers"][tier]
    detector_source = source_directory(sources["detector"], work / tier / "detector-source")
    recognizer_source = source_directory(sources["recognizer"], work / tier / "recognizer-source")
    detector_error, _, detector_timing = compare_and_time(
        detector_source / "inference.onnx", detector, DET_SHAPE, DET_TOLERANCE, runs
    )
    recognizer_error, recognizer_argmax_mismatches, recognizer_timing = compare_and_time(
        recognizer_source / "inference.onnx", recognizer, REC_SHAPE, REC_TOLERANCE, runs,
        require_argmax_equal=True,
    )

    pack = {
        "tier": tier,
        "upstreamOfficialRecognitionWAvg": OFFICIAL_WAVG[tier],
        "source": sources,
        "charactersCount": len(charset.read_text(encoding="utf-8").splitlines()),
        "conversionMaxAbsoluteError": {
            "detector": detector_error,
            "recognizer": recognizer_error,
        },
        "recognizerArgmaxMismatches": recognizer_argmax_mismatches,
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": digest(archive),
        },
    }
    timing = {"tier": tier, "detector": detector_timing, "recognizer": recognizer_timing}
    return pack, timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    assets = args.assets.resolve()
    source_lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="pp-ocr-v1-validation-") as temporary:
        validated = [
            validate_tier(tier, assets, Path(temporary), source_lock, args.runs) for tier in TIERS
        ]
    packs = [item[0] for item in validated]
    timings = [item[1] for item in validated]

    manifest = {
        "schemaVersion": 1,
        "packVersion": PACK_VERSION,
        "releaseTag": f"v{PACK_VERSION}",
        "upstream": {
            "project": "PaddleOCR",
            "modelFamily": source_lock["modelFamily"],
            "license": source_lock["license"],
        },
        "conversion": {
            "minimumDeploymentTarget": "iOS 17",
            "computePrecision": "FLOAT16",
            "inputOutputType": "FLOAT32",
            "archiveFormat": "AppleArchive",
            "compression": "lzfse",
        },
        "detector": {
            "inputWidth": 960,
            "inputHeight": 960,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "conversionTolerance": DET_TOLERANCE,
        },
        "recognizer": {
            "inputWidth": 640,
            "inputHeight": 48,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "conversionTolerance": REC_TOLERANCE,
        },
        "packs": packs,
    }
    benchmark = {
        "schemaVersion": 1,
        "releaseTag": f"v{PACK_VERSION}",
        "assetProvenance": "Byte-identical payloads migrated from YspritanHyzygy/Verto ocr-models-v1",
        "environment": machine_context(),
        "method": {
            "input": "Deterministic random Float32 tensor with seed 1",
            "onnxProvider": "CPUExecutionProvider",
            "coreMLComputeUnits": "ALL",
            "warmupRuns": 1,
            "timedRuns": args.runs,
            "loadTimeIncludesCompilation": True,
        },
        "packs": [
            {
                "tier": pack["tier"],
                "assetSHA256": pack["archive"]["sha256"],
                "upstreamOfficialRecognitionWAvg": pack["upstreamOfficialRecognitionWAvg"],
                "conversionMaxAbsoluteError": pack["conversionMaxAbsoluteError"],
                "recognizerArgmaxMismatches": pack["recognizerArgmaxMismatches"],
                "runtime": timing,
            }
            for pack, timing in zip(packs, timings)
        ],
        "limitations": [
            "Runtime measurements use synthetic tensors and are not end-to-end OCR latency.",
            "Upstream recognition W-Avg values are PaddlePaddle results, not measurements by this project.",
            "No iPhone runtime measurement is claimed by this report.",
        ],
    }

    (ROOT / "manifests/v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "benchmarks/v1.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (assets / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (assets / "SHA256SUMS").write_text(
        "".join(f"{pack['archive']['sha256']}  {pack['archive']['name']}\n" for pack in packs),
        encoding="utf-8",
    )
    print("Migrated release assets and Core ML outputs are valid.")


if __name__ == "__main__":
    main()
