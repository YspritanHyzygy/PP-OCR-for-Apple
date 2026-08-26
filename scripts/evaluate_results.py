#!/usr/bin/env python3
"""Score raw OCR predictions according to evaluation/metrics-v1.md."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import regex
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


IOU_THRESHOLD = 0.5
IGNORE_OVERLAP_THRESHOLD = 0.5
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_REPLICATES = 10_000


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return normalized.strip(" \t\n")


def graphemes(value: str) -> list[str]:
    return regex.findall(r"\X", normalize_text(value))


def levenshtein(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def polygon(points: list[list[float]], location: str) -> Polygon:
    if len(points) < 3:
        raise ValueError(f"{location}: polygon needs at least three points")
    value = Polygon(points)
    if not value.is_valid:
        value = value.buffer(0)
    if value.is_empty or value.area <= 0:
        raise ValueError(f"{location}: polygon is empty after repair")
    return value


def intersection_over_union(left: Polygon, right: Polygon) -> float:
    union = left.union(right).area
    return 0.0 if union <= 0 else float(left.intersection(right).area / union)


def ignored_prediction(prediction: Polygon, ignored: list[Polygon]) -> bool:
    if prediction.area <= 0:
        return False
    return any(
        prediction.intersection(region).area / prediction.area >= IGNORE_OVERLAP_THRESHOLD
        for region in ignored
    )


def score_sample(sample: dict[str, Any]) -> dict[str, Any]:
    sample_id = sample["id"]
    ground_truth = sample.get("groundTruth", [])
    predictions = sample.get("predictions", [])
    detection_predictions = sample.get("detectionPredictions", predictions)
    ignored_regions = [
        polygon(item["polygon"], f"{sample_id}.groundTruth[{index}]")
        for index, item in enumerate(ground_truth)
        if item.get("ignore", False)
    ]
    truths = [item for item in ground_truth if not item.get("ignore", False)]
    truth_polygons = [
        polygon(item["polygon"], f"{sample_id}.groundTruth[{index}]")
        for index, item in enumerate(truths)
    ]

    def filter_predictions(
        values: list[dict[str, Any]], field: str
    ) -> tuple[list[dict[str, Any]], list[Polygon]]:
        filtered: list[dict[str, Any]] = []
        shapes: list[Polygon] = []
        for index, item in enumerate(values):
            shape = polygon(item["polygon"], f"{sample_id}.{field}[{index}]")
            if ignored_prediction(shape, ignored_regions):
                continue
            filtered.append(item)
            shapes.append(shape)
        return filtered, shapes

    def matched_indices(prediction_polygons: list[Polygon]) -> list[tuple[int, int]]:
        if not truth_polygons or not prediction_polygons:
            return []
        overlaps = np.array(
            [
                [intersection_over_union(truth, prediction) for prediction in prediction_polygons]
                for truth in truth_polygons
            ],
            dtype=np.float64,
        )
        rows, columns = linear_sum_assignment(-overlaps)
        return [
            (int(row), int(column))
            for row, column in zip(rows, columns)
            if overlaps[row, column] >= IOU_THRESHOLD
        ]

    filtered_detections, detection_polygons = filter_predictions(
        detection_predictions, "detectionPredictions"
    )
    filtered_predictions, prediction_polygons = filter_predictions(predictions, "predictions")
    detection_matches = matched_indices(detection_polygons)
    text_matches = matched_indices(prediction_polygons)

    matched_truths = {truth for truth, _ in text_matches}
    matched_predictions = {prediction for _, prediction in text_matches}
    scores_text = sample.get("evaluationTask", "end-to-end") != "detection"
    edits = 0
    ground_truth_characters = 0
    exact_matches = 0
    if scores_text:
        for truth_index, prediction_index in text_matches:
            expected = graphemes(truths[truth_index]["text"])
            actual = graphemes(filtered_predictions[prediction_index]["text"])
            ground_truth_characters += len(expected)
            edits += levenshtein(expected, actual)
            exact_matches += expected == actual
        for index, item in enumerate(truths):
            if index not in matched_truths:
                expected = graphemes(item["text"])
                ground_truth_characters += len(expected)
                edits += len(expected)
        for index, item in enumerate(filtered_predictions):
            if index not in matched_predictions:
                edits += len(graphemes(item["text"]))

    return {
        "id": sample_id,
        "dataset": sample["dataset"],
        "language": sample["language"],
        "scenario": sample["scenario"],
        "truePositives": len(detection_matches),
        "falseNegatives": len(truths) - len(detection_matches),
        "falsePositives": len(filtered_detections) - len(detection_matches),
        "recognizedTruePositives": len(text_matches),
        "groundTruthLines": len(truths),
        "textScoredGroundTruthLines": len(truths) if scores_text else 0,
        "textScoredMatches": len(text_matches) if scores_text else 0,
        "exactMatches": exact_matches,
        "edits": edits,
        "groundTruthCharacters": ground_truth_characters,
        "timingsMilliseconds": sample.get("timingsMilliseconds", {}),
        "measurements": sample.get("measurements", []),
    }


def aggregate(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(results)
    totals = {
        key: sum(item[key] for item in items)
        for key in (
            "truePositives",
            "falseNegatives",
            "falsePositives",
            "recognizedTruePositives",
            "groundTruthLines",
            "textScoredGroundTruthLines",
            "textScoredMatches",
            "exactMatches",
            "edits",
            "groundTruthCharacters",
        )
    }
    tp = totals["truePositives"]
    precision = tp / max(1, tp + totals["falsePositives"])
    recall = tp / max(1, tp + totals["falseNegatives"])
    hmean = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    characters = totals["groundTruthCharacters"]
    text_matches = totals["textScoredMatches"]
    metrics: dict[str, Any] = {
        **totals,
        "detectionPrecision": precision,
        "detectionRecall": recall,
        "detectionHmean": hmean,
        "characterErrorRate": totals["edits"] / characters if characters else None,
        "endToEndCharacterAccuracy": (
            max(0.0, 1 - totals["edits"] / characters) if characters else None
        ),
        "exactLineAccuracy": totals["exactMatches"] / text_matches if text_matches else None,
        "lineRecall": (
            totals["recognizedTruePositives"] / totals["textScoredGroundTruthLines"]
            if totals["textScoredGroundTruthLines"] else None
        ),
        "sampleCount": len(items),
    }

    timings: dict[str, list[float]] = defaultdict(list)
    for item in items:
        measurements = item["measurements"]
        raw_timings = (
            [measurement.get("timingsMilliseconds", {}) for measurement in measurements]
            if measurements else [item["timingsMilliseconds"]]
        )
        for measurement in raw_timings:
            for key, value in measurement.items():
                if isinstance(value, (int, float)):
                    timings[key].append(float(value))
    metrics["timingsMilliseconds"] = {
        key: {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "count": len(values),
        }
        for key, values in sorted(timings.items())
    }
    return metrics


def slices(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("dataset", "language", "scenario"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in results:
            groups[item[field]].append(item)
        output[field] = {name: aggregate(values) for name, values in sorted(groups.items())}
    return output


def bootstrap_delta(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, float]:
    baseline_by_id = {item["id"]: item for item in baseline}
    candidate_by_id = {item["id"]: item for item in candidate}
    if set(baseline_by_id) != set(candidate_by_id):
        missing = sorted(set(baseline_by_id) ^ set(candidate_by_id))
        raise ValueError(f"baseline and candidate sample IDs differ: {missing[:10]}")
    ids = sorted(baseline_by_id)
    if not ids:
        raise ValueError("cannot bootstrap an empty report")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(ids), size=len(ids))
        baseline_edits = baseline_chars = candidate_edits = candidate_chars = 0
        for index in selected:
            sample_id = ids[int(index)]
            before = baseline_by_id[sample_id]
            after = candidate_by_id[sample_id]
            baseline_edits += before["edits"]
            baseline_chars += before["groundTruthCharacters"]
            candidate_edits += after["edits"]
            candidate_chars += after["groundTruthCharacters"]
        # Do not clamp the paired statistic at zero. Hard scenes can have CER > 1
        # because extra predictions count as insertions; clamping would turn two
        # materially different bad candidates into the same all-zero sample.
        baseline_accuracy = 1 - baseline_edits / max(1, baseline_chars)
        candidate_accuracy = 1 - candidate_edits / max(1, candidate_chars)
        deltas[replicate] = candidate_accuracy - baseline_accuracy
    return {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "lower95": float(np.percentile(deltas, 2.5)),
        "median": float(np.percentile(deltas, 50)),
        "upper95": float(np.percentile(deltas, 97.5)),
    }


def score_report(report: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if report.get("schemaVersion") not in {1, 2}:
        raise ValueError("report schemaVersion must be 1 or 2")
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("report.samples must be a non-empty array")
    scored = [score_sample(sample) for sample in samples]
    return {
        "schemaVersion": 1,
        "run": report.get("run", {}),
        "metrics": aggregate(scored),
        "slices": slices(scored),
    }, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="candidate raw prediction report")
    parser.add_argument("--baseline", type=Path, help="paired baseline raw prediction report")
    parser.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")
    args = parser.parse_args()

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        output, candidate_scored = score_report(report)
        if args.baseline:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            baseline_output, baseline_scored = score_report(baseline)
            output["baselineRun"] = baseline_output["run"]
            output["endToEndAccuracyDelta"] = bootstrap_delta(
                baseline_scored, candidate_scored
            )
        serialized = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(serialized, encoding="utf-8")
        else:
            sys.stdout.write(serialized)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"cannot score {args.report}: {error}")


if __name__ == "__main__":
    main()
