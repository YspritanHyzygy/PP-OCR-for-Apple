#!/usr/bin/env python3
"""Build Verto tests, inject model/corpus paths, and run the real OCR pipeline."""

from __future__ import annotations

import argparse
import plistlib
import subprocess
from pathlib import Path


TIERS = ("tiny", "small", "medium")


def run(arguments: list[str], cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def inject_environment(
    source: Path,
    destination: Path,
    environment: dict[str, str],
) -> None:
    with source.open("rb") as handle:
        document = plistlib.load(handle)
    test_target = document.get("VertoTests")
    if not isinstance(test_target, dict):
        raise SystemExit(f"{source}: VertoTests entry is missing")
    values = test_target.setdefault("EnvironmentVariables", {})
    values.update(environment)
    with destination.open("wb") as handle:
        plistlib.dump(document, handle, fmt=plistlib.FMT_BINARY, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verto", required=True, type=Path, help="Verto checkout")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tier", required=True, choices=TIERS)
    parser.add_argument("--detector-size", type=int, choices=(640, 736, 960), default=960)
    parser.add_argument("--recognizer-width", type=int, choices=(320, 640), default=640)
    parser.add_argument("--box-score-threshold", type=float, choices=(0.4, 0.45))
    parser.add_argument(
        "--b0-color-contract", action="store_true",
        help="reproduce Verto v1's RGB tensor sent to an upstream BGR model",
    )
    parser.add_argument("--destination", required=True, help="xcodebuild destination specifier")
    parser.add_argument("--derived-data", required=True, type=Path)
    parser.add_argument("--result-bundle", required=True, type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--warmup-runs", type=int, default=3)
    args = parser.parse_args()

    verto = args.verto.resolve()
    if args.result_bundle.exists():
        raise SystemExit(f"{args.result_bundle}: remove the old result bundle before rerunning")
    if bool(args.corpus) != bool(args.report):
        raise SystemExit("--corpus and --report must be supplied together")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs cannot be negative")

    run(
        [
            "xcodebuild",
            "build-for-testing",
            "-project",
            "Verto.xcodeproj",
            "-scheme",
            "Verto",
            "-destination",
            args.destination,
            "-derivedDataPath",
            str(args.derived_data.resolve()),
        ],
        verto,
    )
    products = args.derived_data.resolve() / "Build/Products"
    configured = products / f"Verto_{args.tier}_benchmark.xctestrun"
    # Reusing DerivedData is intentional for iterative candidate runs. Exclude the
    # configured copy written by an earlier invocation so it cannot masquerade as
    # a second build product on the next run.
    candidates = sorted(path for path in products.glob("*.xctestrun") if path != configured)
    if len(candidates) != 1:
        raise SystemExit(f"{products}: expected one xctestrun, found {len(candidates)}")
    environment = {
        "VERTO_OCR_MODEL_DIR": str(args.model_dir.resolve()),
        "VERTO_OCR_MODEL_TIER": args.tier,
        "VERTO_OCR_DETECTOR_SIZE": str(args.detector_size),
        "VERTO_OCR_RECOGNIZER_WIDTH": str(args.recognizer_width),
        "VERTO_OCR_WARMUP_RUNS": str(args.warmup_runs),
    }
    if args.b0_color_contract:
        environment["VERTO_OCR_B0_COLOR_CONTRACT"] = "1"
    if args.box_score_threshold is not None:
        environment["VERTO_OCR_BOX_SCORE_THRESHOLD"] = str(args.box_score_threshold)
    test_identifier = "VertoTests/PaddleOCRProbeTests/testProbePaddleRecognitionPipeline"
    if args.corpus and args.report:
        environment["VERTO_OCR_CORPUS_JSON"] = str(args.corpus.resolve())
        environment["VERTO_OCR_REPORT_PATH"] = str(args.report.resolve())
        test_identifier = "VertoTests/PaddleOCRProbeTests/testBenchmarkExternalCorpus"
    inject_environment(candidates[0], configured, environment)

    run(
        [
            "xcodebuild",
            "test-without-building",
            "-xctestrun",
            str(configured),
            "-destination",
            args.destination,
            "-resultBundlePath",
            str(args.result_bundle.resolve()),
            f"-only-testing:{test_identifier}",
        ],
        verto,
    )


if __name__ == "__main__":
    main()
