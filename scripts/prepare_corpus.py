#!/usr/bin/env python3
"""Verify and convert supported public OCR archives into the Verto corpus schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def locked_dataset(dataset_id: str) -> dict[str, Any]:
    lock = json.loads((ROOT / "evaluation/corpus.lock.json").read_text(encoding="utf-8"))
    for dataset in lock["publicDatasets"]:
        if dataset["id"] == dataset_id:
            return dataset
    raise SystemExit(f"evaluation/corpus.lock.json does not contain {dataset_id}")


def verify_archive(path: Path, dataset_id: str) -> None:
    expected = locked_dataset(dataset_id)["archiveSHA256"]
    if expected is None:
        raise SystemExit(f"{dataset_id}: archive SHA-256 is not locked")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{path}: SHA-256 {actual} does not match locked {expected}")


def prepare_total_text(archive: Path, output: Path) -> Path:
    verify_archive(archive, "total-text-test")
    image_output = output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        annotation_member = source.getmember("total_text/test/test.txt")
        annotation_file = source.extractfile(annotation_member)
        if annotation_file is None:
            raise SystemExit(f"{archive}: total_text/test/test.txt is unreadable")
        rows = annotation_file.read().decode("utf-8").splitlines()
        samples = []
        for row_number, row in enumerate(rows, start=1):
            relative_image, encoded_lines = row.split("\t", maxsplit=1)
            source_name = f"total_text/test/{relative_image}"
            image_member = source.getmember(source_name)
            image_source = source.extractfile(image_member)
            if image_source is None:
                raise SystemExit(f"{archive}: {source_name} is unreadable")
            image_name = Path(relative_image).name
            with (image_output / image_name).open("wb") as destination:
                shutil.copyfileobj(image_source, destination)
            annotations = json.loads(encoded_lines)
            samples.append(
                {
                    "id": f"total-text:{Path(image_name).stem}",
                    "dataset": "total-text-test",
                    "language": "en",
                    "scenario": "rotated-or-curved-text",
                    "imagePath": f"images/{image_name}",
                    "groundTruth": [
                        {
                            "polygon": item["points"],
                            "text": item["transcription"],
                            # The prepared archive uses a placeholder instead
                            # of the real Chinese transcription. Excluding it
                            # is more truthful than scoring the placeholder.
                            "ignore": item["transcription"] in {"###", "chinese_text"},
                        }
                        for item in annotations
                    ],
                }
            )
    corpus = output / "corpus.json"
    corpus.write_text(
        json.dumps({"schemaVersion": 1, "samples": samples}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return corpus


def prepare_ctw1500(archive: Path, output: Path) -> Path:
    """Prepare CTW1500's official detection-only annotations."""
    verify_archive(archive, "ctw1500-test")
    image_output = output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        rows = source.read("ctw1500/imgs/test.txt").decode("utf-8").splitlines()
        samples = []
        for row in rows:
            relative_image, encoded_lines = row.split("\t", maxsplit=1)
            image_name = Path(relative_image).name
            with source.open(f"ctw1500/imgs/{relative_image}") as image_source:
                with (image_output / image_name).open("wb") as destination:
                    shutil.copyfileobj(image_source, destination)
            annotations = json.loads(encoded_lines)
            samples.append(
                {
                    "id": f"ctw1500:{Path(image_name).stem}",
                    "dataset": "ctw1500-test",
                    "language": "zh-Hans",
                    "scenario": "rotated-or-curved-text",
                    # Paddle's archive stores transcription=0 for every polygon.
                    # It can support detection metrics, not honest CER scoring.
                    "evaluationTask": "detection",
                    "imagePath": f"images/{image_name}",
                    "groundTruth": [
                        {"polygon": item["points"], "text": "", "ignore": False}
                        for item in annotations
                    ],
                }
            )
    corpus = output / "corpus.json"
    corpus.write_text(
        json.dumps({"schemaVersion": 1, "samples": samples}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    archives = parser.add_mutually_exclusive_group(required=True)
    archives.add_argument("--total-text-archive", type=Path)
    archives.add_argument("--ctw1500-archive", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.total_text_archive:
        corpus = prepare_total_text(args.total_text_archive.resolve(), args.output_dir.resolve())
    else:
        corpus = prepare_ctw1500(args.ctw1500_archive.resolve(), args.output_dir.resolve())
    print(corpus)


if __name__ == "__main__":
    main()
