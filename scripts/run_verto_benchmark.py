#!/usr/bin/env python3
"""Build Verto tests, inject model/corpus paths, and run the real OCR pipeline."""

from __future__ import annotations

import argparse
import json
import plistlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


TIERS = ("tiny", "small", "medium")
COMPUTE_UNITS = ("all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly")
EVIDENCE_CLASSES = ("simulator", "localPhysical", "remotePhysical")
HARDWARE_GROUPS = ("A12-A13", "A14-A16", "A17Pro-A19", "future")
CANDIDATE_KINDS = ("reference", "shape", "w8a8", "boundary", "unspecified")


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


def configured_xctestrun(
    products: Path, tier: str, candidate: str, compute_units: str
) -> Path:
    label = re.sub(r"[^A-Za-z0-9_.-]", "-", candidate)
    return products / f"Configured-Verto_{tier}_{label}_{compute_units}.xctestrun"


def build_xctestruns(products: Path) -> list[Path]:
    return sorted(
        path for path in products.glob("*.xctestrun")
        if not path.name.startswith("Configured-")
    )


def stage_corpus(source: Path, destination: Path) -> Path:
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != 1 or not isinstance(document.get("samples"), list):
        raise SystemExit(f"{source}: corpus must use schemaVersion 1")
    images = destination / "images"
    images.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(document["samples"]):
        image = Path(sample["imagePath"])
        if not image.is_absolute():
            image = source.parent / image
        suffix = image.suffix.lower() or ".img"
        name = f"{index:05d}{suffix}"
        shutil.copy2(image, images / name)
        sample["imagePath"] = f"images/{name}"
    staged = destination / "corpus.json"
    staged.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return staged


def stage_physical_assets(
    products: Path,
    model_directory: Path,
    corpus: Path | None,
    signing_identity: str,
) -> dict[str, str]:
    test_bundles = list(
        products.glob("Debug-iphoneos/Verto.app/PlugIns/VertoTests.xctest")
    )
    if len(test_bundles) != 1:
        raise SystemExit(f"{products}: expected one physical VertoTests.xctest")
    test_bundle = test_bundles[0]
    host_app = test_bundle.parents[1]
    assets = test_bundle / "BenchmarkAssets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir()
    shutil.copytree(model_directory, assets / "model")
    environment = {"VERTO_OCR_BUNDLED_MODEL_PATH": "BenchmarkAssets/model"}
    if corpus is not None:
        staged = stage_corpus(corpus, assets / "corpus")
        environment["VERTO_OCR_BUNDLED_CORPUS_PATH"] = str(
            staged.relative_to(test_bundle)
        )
        environment["VERTO_OCR_REPORT_ATTACHMENT_NAME"] = "verto-ocr-report"

    preserve = "--preserve-metadata=identifier,entitlements,requirements,flags,runtime"
    run(["codesign", "--force", "--sign", signing_identity, preserve, str(test_bundle)], products)
    run(["codesign", "--force", "--sign", signing_identity, preserve, str(host_app)], products)
    return environment


def exported_attachment(manifest: list[dict], name: str) -> str:
    matches = [
        attachment["exportedFileName"]
        for test in manifest
        for attachment in test.get("attachments", [])
        if attachment.get("suggestedHumanReadableName", "").startswith(f"{name}_")
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one {name} xcresult attachment, found {len(matches)}")
    return matches[0]


def extract_report_attachment(result_bundle: Path, name: str, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="verto-ocr-attachments-") as temporary:
        directory = Path(temporary)
        run(
            [
                "xcrun", "xcresulttool", "export", "attachments",
                "--path", str(result_bundle),
                "--output-path", str(directory),
                "--filter", f"{name}*",
            ],
            Path.cwd(),
        )
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        source = directory / exported_attachment(manifest, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verto", required=True, type=Path, help="Verto checkout")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tier", required=True, choices=TIERS)
    parser.add_argument("--candidate", required=True, help="recipe/candidate identifier")
    parser.add_argument("--candidate-kind", choices=CANDIDATE_KINDS, default="unspecified")
    parser.add_argument("--parent-candidate")
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
    parser.add_argument("--measurement-runs", type=int, default=1)
    parser.add_argument("--compute-units", choices=COMPUTE_UNITS, default="all")
    parser.add_argument("--evidence-class", choices=EVIDENCE_CLASSES, required=True)
    parser.add_argument("--hardware-group", choices=HARDWARE_GROUPS)
    parser.add_argument("--device-label")
    parser.add_argument(
        "--development-team",
        help="Xcode development team for physical-device test signing",
    )
    parser.add_argument(
        "--signing-identity",
        help="codesign identity used after embedding physical-device benchmark assets",
    )
    parser.add_argument("--energy-source")
    parser.add_argument("--energy-joules", type=float)
    parser.add_argument("--energy-artifact", type=Path)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse the single xctestrun already present in DerivedData",
    )
    args = parser.parse_args()

    verto = args.verto.resolve()
    if args.result_bundle.exists():
        raise SystemExit(f"{args.result_bundle}: remove the old result bundle before rerunning")
    if bool(args.corpus) != bool(args.report):
        raise SystemExit("--corpus and --report must be supplied together")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs cannot be negative")
    if args.measurement_runs < 1:
        raise SystemExit("--measurement-runs must be positive")
    if args.evidence_class != "simulator" and not args.hardware_group:
        raise SystemExit("physical-device evidence requires --hardware-group")
    if args.evidence_class == "simulator" and args.hardware_group:
        raise SystemExit("simulator evidence cannot claim a hardware group")
    destination_is_simulator = "simulator" in args.destination.lower()
    if (args.evidence_class == "simulator") != destination_is_simulator:
        raise SystemExit("--evidence-class must match the xcodebuild destination")
    if args.evidence_class != "simulator" and not args.development_team:
        raise SystemExit("physical-device builds require --development-team")
    if args.evidence_class != "simulator" and not args.signing_identity:
        raise SystemExit("physical-device benchmark assets require --signing-identity")
    if args.energy_joules is not None and not args.energy_source:
        raise SystemExit("--energy-joules requires --energy-source")

    if not args.skip_build:
        build_arguments = [
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
        ]
        if args.development_team:
            build_arguments.append(f"DEVELOPMENT_TEAM={args.development_team}")
        run(build_arguments, verto)
    products = args.derived_data.resolve() / "Build/Products"
    configured = configured_xctestrun(
        products, args.tier, args.candidate, args.compute_units
    )
    # The xctestrun contains paths relative to its own Build/Products directory,
    # so configured copies must stay beside the original. The reserved prefix
    # prevents previous matrix runs from masquerading as build products.
    candidates = build_xctestruns(products)
    if len(candidates) != 1:
        raise SystemExit(f"{products}: expected one xctestrun, found {len(candidates)}")
    environment = {
        "VERTO_OCR_MODEL_TIER": args.tier,
        "VERTO_OCR_CANDIDATE": args.candidate,
        "VERTO_OCR_CANDIDATE_KIND": args.candidate_kind,
        "VERTO_OCR_DETECTOR_SIZE": str(args.detector_size),
        "VERTO_OCR_RECOGNIZER_WIDTH": str(args.recognizer_width),
        "VERTO_OCR_WARMUP_RUNS": str(args.warmup_runs),
        "VERTO_OCR_MEASUREMENT_RUNS": str(args.measurement_runs),
        "VERTO_OCR_COMPUTE_UNITS": args.compute_units,
        "VERTO_OCR_EVIDENCE_CLASS": args.evidence_class,
    }
    if args.evidence_class == "simulator":
        environment["VERTO_OCR_MODEL_DIR"] = str(args.model_dir.resolve())
    else:
        environment.update(stage_physical_assets(
            products,
            args.model_dir.resolve(),
            args.corpus.resolve() if args.corpus else None,
            args.signing_identity,
        ))
    if args.parent_candidate:
        environment["VERTO_OCR_PARENT_CANDIDATE"] = args.parent_candidate
    if args.hardware_group:
        environment["VERTO_OCR_HARDWARE_GROUP"] = args.hardware_group
    if args.device_label:
        environment["VERTO_OCR_DEVICE_LABEL"] = args.device_label
    if args.energy_source:
        environment["VERTO_OCR_ENERGY_SOURCE"] = args.energy_source
    if args.energy_joules is not None:
        environment["VERTO_OCR_ENERGY_JOULES"] = str(args.energy_joules)
    if args.energy_artifact:
        environment["VERTO_OCR_ENERGY_ARTIFACT"] = str(args.energy_artifact.resolve())
    if args.b0_color_contract:
        environment["VERTO_OCR_B0_COLOR_CONTRACT"] = "1"
    if args.box_score_threshold is not None:
        environment["VERTO_OCR_BOX_SCORE_THRESHOLD"] = str(args.box_score_threshold)
    test_identifier = "VertoTests/PaddleOCRProbeTests/testProbePaddleRecognitionPipeline"
    if args.corpus and args.report:
        if args.evidence_class == "simulator":
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
    if args.evidence_class != "simulator" and args.report:
        extract_report_attachment(
            args.result_bundle.resolve(), "verto-ocr-report", args.report.resolve()
        )


if __name__ == "__main__":
    main()
