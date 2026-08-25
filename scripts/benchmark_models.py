#!/usr/bin/env python3
"""Measure Core ML package load and prediction time on the current Mac."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def sysctl(name: str) -> str:
    result = subprocess.run(
        ["sysctl", "-n", name], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def benchmark_component(
    path: Path,
    shape: list[int],
    io_type: str,
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
    load_milliseconds = (time.perf_counter() - started) * 1000
    dtype = np.float32 if io_type == "FLOAT32" else np.float16
    x = np.random.default_rng(1).random(shape).astype(dtype)
    for _ in range(warmup_runs):
        model.predict({"x": x})
    timings: list[float] = []
    for _ in range(measured_runs):
        started = time.perf_counter()
        model.predict({"x": x})
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "packageBytes": directory_bytes(path),
        "loadMilliseconds": load_milliseconds,
        "warmupRuns": warmup_runs,
        "measuredRuns": measured_runs,
        "predictionP50Milliseconds": percentile(timings, 50),
        "predictionP95Milliseconds": percentile(timings, 95),
        "predictionSamplesMilliseconds": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path, help="directory containing tier subdirectories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measured-runs", type=int, default=30)
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.measured_runs <= 0:
        parser.error("warmup runs must be nonnegative and measured runs must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 2:
        parser.error("benchmarking requires a schemaVersion 2 manifest")
    packs = []
    for pack in manifest["packs"]:
        root = args.models / pack["tier"]
        packs.append(
            {
                "tier": pack["tier"],
                "archive": pack["archive"],
                "detector": benchmark_component(
                    root / "VertoTextDetector.mlpackage",
                    pack["detector"]["inputShape"],
                    pack["detector"]["inputOutputType"],
                    args.warmup_runs,
                    args.measured_runs,
                ),
                "recognizer": benchmark_component(
                    root / "VertoTextRecognizer.mlpackage",
                    pack["recognizer"]["inputShape"],
                    pack["recognizer"]["inputOutputType"],
                    args.warmup_runs,
                    args.measured_runs,
                ),
            }
        )
    report = {
        "schemaVersion": 1,
        "kind": "mac-synthetic-tensor",
        "manifestSHA256": sha256(args.manifest),
        "environment": {
            "machine": platform.machine(),
            "modelIdentifier": sysctl("hw.model"),
            "chip": sysctl("machdep.cpu.brand_string"),
            "memoryBytes": int(sysctl("hw.memsize")),
            "macOS": platform.mac_ver()[0],
            "coremltools": ct.__version__,
            "computeUnits": "ALL",
        },
        "method": {
            "input": "deterministic random tensor with NumPy seed 1",
            "loadTimeIncludesCompilation": True,
            "limitations": [
                "Synthetic tensors do not measure OCR quality.",
                "Mac timings do not establish iPhone latency or energy.",
                "This report excludes Verto preprocessing, post-processing, cropping, and decoding.",
            ],
        },
        "packs": packs,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
