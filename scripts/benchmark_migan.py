#!/usr/bin/env python3
"""Measure a MI-GAN Core ML package on the current Apple host.

The generated report is development evidence. Physical-device and Instruments
fields stay unverified until they are supplied by the iOS benchmark path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import coremltools as ct
import numpy as np


def tree_digest(root: Path) -> str:
    value = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        value.update(len(relative).to_bytes(8, "big"))
        value.update(relative)
        size = path.stat().st_size
        value.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                value.update(chunk)
    return value.hexdigest()


def predict(model: ct.models.MLModel, image: np.ndarray, mask: np.ndarray) -> None:
    output = model.predict({"image": image, "mask": mask})["output"]
    if output.shape != (1, 3, 512, 512) or not np.isfinite(output).all():
        raise SystemExit("MI-GAN prediction returned an invalid output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    image = np.linspace(0, 1, 3 * 512 * 512, dtype=np.float16).reshape(1, 3, 512, 512)
    mask = np.ones((1, 1, 512, 512), dtype=np.float16)
    mask[:, :, 176:336, 144:368] = 0

    load_started = time.perf_counter()
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.ALL)
    load_ms = (time.perf_counter() - load_started) * 1000
    predict(model, image, mask)

    times: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        predict(model, image, mask)
        times.append((time.perf_counter() - started) * 1000)

    report = {
        "schemaVersion": 1,
        "releaseTag": "migan-v1",
        "status": "development-only",
        "modelTreeSHA256": tree_digest(args.model),
        "host": platform.platform(),
        "device": platform.machine(),
        "os": platform.mac_ver()[0],
        "thermalState": None,
        "computeUnits": "ALL",
        "measurementScope": "model.prediction",
        "coldLoadIncludesCompilation": False,
        "warmupRuns": 1,
        "measuredRuns": 3,
        "coldLoadMilliseconds": load_ms,
        "predictionMilliseconds": times,
        "medianPredictionMilliseconds": statistics.median(times),
        "observedResidentHighWaterBytes": None,
        "coreMLInstruments": {
            "status": "unverified",
            "executionDevice": None,
            "traceSHA256": None,
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
