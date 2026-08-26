#!/usr/bin/env python3
"""Summarize raw Verto Core ML placement and paired device measurements."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHYSICAL_EVIDENCE = {"localPhysical", "remotePhysical"}
COMPARABLE_THERMAL_STATES = {"nominal", "fair"}
COMPUTE_UNITS = {"all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly"}
CANDIDATE_KINDS = {"reference", "shape", "w8a8", "boundary", "unspecified"}
REQUIRED_HARDWARE_GROUPS = {"A12-A13", "A14-A16", "A17Pro-A19"}
ANE_SHARE_GATE = 0.90
SIGNIFICANT_OPERATION_SHARE = 0.05
BENEFIT_GATE = 0.15
REGRESSION_GATE = 0.10


def load_explanations(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != 1 or not isinstance(document.get("operations"), dict):
        raise ValueError("explanations must use schemaVersion 1 and an operations object")
    return {
        str(key): str(value).strip()
        for key, value in document["operations"].items()
        if str(value).strip()
    }


def load_quality_decisions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != 1 or not isinstance(document.get("candidates"), dict):
        raise ValueError("quality decisions must use schemaVersion 1 and a candidates object")
    decisions = {str(key): str(value) for key, value in document["candidates"].items()}
    if any(value not in {"pass", "fail"} for value in decisions.values()):
        raise ValueError("quality decisions must be pass or fail")
    return decisions


def validate_report(report: dict[str, Any], location: str) -> None:
    if report.get("schemaVersion") != 2:
        raise ValueError(f"{location}: ANE assessment requires raw schemaVersion 2")
    run = report.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"{location}: run is missing")
    required_run_fields = {
        "tier", "candidate", "candidateKind", "systemVersion", "operatingSystem",
        "computeUnits", "evidenceClass", "availableComputeDevices", "warmupRuns",
        "measurementRunsPerSample", "thermalStateStart", "thermalStateEnd", "energy",
    }
    missing_run_fields = sorted(required_run_fields - set(run))
    if missing_run_fields:
        raise ValueError(f"{location}: run is missing {missing_run_fields}")
    if run.get("computeUnits") not in COMPUTE_UNITS:
        raise ValueError(f"{location}: unknown computeUnits")
    if run.get("candidateKind", "unspecified") not in CANDIDATE_KINDS:
        raise ValueError(f"{location}: unknown candidateKind")
    if run.get("candidateKind") == "w8a8" and not run.get("parentCandidate"):
        raise ValueError(f"{location}: W8A8 evidence needs parentCandidate")
    if run.get("evidenceClass") not in PHYSICAL_EVIDENCE | {"simulator"}:
        raise ValueError(f"{location}: unknown evidenceClass")
    if run["evidenceClass"] in PHYSICAL_EVIDENCE and not run.get("hardwareGroup"):
        raise ValueError(f"{location}: physical evidence needs hardwareGroup")
    if not isinstance(report.get("computePlans"), list):
        raise ValueError(f"{location}: computePlans must be an array")
    if {plan.get("component") for plan in report["computePlans"]} != {"detector", "recognizer"}:
        raise ValueError(f"{location}: computePlans must contain detector and recognizer")
    if not isinstance(report.get("samples"), list) or not report["samples"]:
        raise ValueError(f"{location}: samples must be a non-empty array")
    required_timings = {
        "preprocessing", "detector", "detectionPostProcess",
        "recognizer", "decode", "endToEnd",
    }
    for index, sample in enumerate(report["samples"]):
        measurements = sample.get("measurements")
        if not isinstance(measurements, list) or not measurements:
            raise ValueError(f"{location}: samples[{index}].measurements must be non-empty")
        for measurement_index, measurement in enumerate(measurements):
            timings = measurement.get("timingsMilliseconds")
            if not isinstance(timings, dict) or not required_timings <= set(timings):
                raise ValueError(
                    f"{location}: samples[{index}].measurements[{measurement_index}] timings are incomplete"
                )
            for field in (
                "thermalStateBefore", "thermalStateAfter", "residentMemoryBytesBefore",
                "residentMemoryBytesAfter", "processLifetimePeakResidentBytes",
            ):
                if field not in measurement:
                    raise ValueError(
                        f"{location}: samples[{index}].measurements[{measurement_index}] lacks {field}"
                    )


def percentile(values: list[float], quantile: float) -> float | None:
    return float(np.percentile(values, quantile)) if values else None


def raw_measurements(report: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sample in report["samples"]:
        measurements = sample.get("measurements")
        if isinstance(measurements, list) and measurements:
            output.extend(item for item in measurements if isinstance(item, dict))
        else:
            output.append({"timingsMilliseconds": sample.get("timingsMilliseconds", {})})
    return output


def performance_summary(report: dict[str, Any]) -> dict[str, Any]:
    measurements = raw_measurements(report)
    timings: dict[str, list[float]] = defaultdict(list)
    inference_total = end_to_end_total = 0.0
    memory_values: list[int] = []
    thermal_states: set[str] = set()
    for measurement in measurements:
        raw_timings = measurement.get("timingsMilliseconds", {})
        if isinstance(raw_timings, dict):
            for name, value in raw_timings.items():
                if isinstance(value, (int, float)):
                    timings[name].append(float(value))
            detector = raw_timings.get("detector")
            recognizer = raw_timings.get("recognizer")
            end_to_end = raw_timings.get("endToEnd")
            if all(isinstance(value, (int, float)) for value in (detector, recognizer, end_to_end)):
                inference_total += float(detector) + float(recognizer)
                end_to_end_total += float(end_to_end)
        peak = measurement.get("processLifetimePeakResidentBytes")
        if isinstance(peak, int):
            memory_values.append(peak)
        for field in ("thermalStateBefore", "thermalStateAfter"):
            if isinstance(measurement.get(field), str):
                thermal_states.add(measurement[field])
    core_ml_fraction = (
        inference_total / end_to_end_total if end_to_end_total > 0 else None
    )
    return {
        "measurementCount": len(measurements),
        "timingsMilliseconds": {
            name: {
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "count": len(values),
            }
            for name, values in sorted(timings.items())
        },
        "processLifetimePeakResidentBytes": max(memory_values) if memory_values else None,
        "thermalStates": sorted(thermal_states),
        "energy": report["run"].get("energy", {"source": "notMeasured"}),
        "coreMLInferenceFraction": core_ml_fraction,
        "optimizationFocus": (
            "cpuPreAndPostProcessing"
            if core_ml_fraction is not None and core_ml_fraction < 0.40
            else "aneCandidateEvaluation"
        ),
    }


def placement_summary(
    report: dict[str, Any], explanations: dict[str, str]
) -> list[dict[str, Any]]:
    summaries = []
    is_all = report["run"]["computeUnits"] == "all"
    for component in report["computePlans"]:
        operations = component.get("operations", [])
        weighted = [
            operation for operation in operations
            if isinstance(operation.get("estimatedCostWeight"), (int, float))
            and operation["estimatedCostWeight"] >= 0
        ]
        total = sum(float(operation["estimatedCostWeight"]) for operation in weighted)
        ane = sum(
            float(operation["estimatedCostWeight"])
            for operation in weighted
            if operation.get("preferredComputeDevice") == "neuralEngine"
        )
        significant = []
        for operation in weighted:
            share = float(operation["estimatedCostWeight"]) / total if total > 0 else 0
            if operation.get("preferredComputeDevice") == "neuralEngine" or share < SIGNIFICANT_OPERATION_SHARE:
                continue
            key = f"{component.get('component')}:{operation.get('path')}"
            significant.append({
                "path": operation.get("path"),
                "operatorName": operation.get("operatorName"),
                "costShare": share,
                "preferredComputeDevice": operation.get("preferredComputeDevice"),
                "fallbackReason": operation.get("fallbackReason"),
                "explanation": explanations.get(key),
            })
        unavailable_usage = sum(
            operation.get("fallbackReason") == "deviceUsageUnavailable"
            for operation in operations
        )
        unsupported = sum(
            operation.get("fallbackReason") == "neuralEngineUnsupported"
            for operation in operations
        )
        weighted_unavailable_usage = sum(
            operation.get("fallbackReason") == "deviceUsageUnavailable"
            for operation in weighted
        )
        weighted_unsupported = sum(
            operation.get("fallbackReason") == "neuralEngineUnsupported"
            for operation in weighted
        )
        share = ane / total if total > 0 else None
        all_explained = all(item["explanation"] for item in significant)
        friendly = (
            component.get("status") == "available"
            and bool(operations)
            and weighted_unavailable_usage == 0
            and weighted_unsupported == 0
            and share is not None
            and share >= ANE_SHARE_GATE
            and all_explained
        ) if is_all else None
        summaries.append({
            "component": component.get("component"),
            "status": component.get("status"),
            "operationCount": len(operations),
            "operationsWithoutDeviceUsage": unavailable_usage,
            "neuralEngineUnsupportedOperations": unsupported,
            "weightedOperationsWithoutDeviceUsage": weighted_unavailable_usage,
            "weightedNeuralEngineUnsupportedOperations": weighted_unsupported,
            "knownEstimatedCostWeight": total,
            "preferredNeuralEngineCostShare": share,
            "significantFallbackOperations": significant,
            "allSignificantFallbacksExplained": all_explained,
            "aneFriendly": friendly,
            "error": component.get("error"),
        })
    return summaries


def prediction_signature(report: dict[str, Any]) -> str:
    values = [
        {
            "id": sample.get("id"),
            "detectionPredictions": sample.get("detectionPredictions", []),
            "predictions": sample.get("predictions", []),
        }
        for sample in report["samples"]
    ]
    return json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def relative_change(before: float | int | None, after: float | int | None) -> float | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)) or before <= 0:
        return None
    return (float(after) - float(before)) / float(before)


def paired_comparison(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    before = performance_summary(baseline)
    after = performance_summary(candidate)
    changes: dict[str, float] = {}
    timing_names = set(before["timingsMilliseconds"]) & set(after["timingsMilliseconds"])
    for name in timing_names:
        for quantile in ("p50", "p95"):
            change = relative_change(
                before["timingsMilliseconds"][name][quantile],
                after["timingsMilliseconds"][name][quantile],
            )
            if change is not None:
                changes[f"timingsMilliseconds.{name}.{quantile}"] = change
    memory_change = relative_change(
        before["processLifetimePeakResidentBytes"],
        after["processLifetimePeakResidentBytes"],
    )
    if memory_change is not None:
        changes["processLifetimePeakResidentBytes"] = memory_change
    before_energy = before["energy"].get("joules")
    after_energy = after["energy"].get("joules")
    energy_change = relative_change(before_energy, after_energy)
    if energy_change is not None:
        changes["energy.joules"] = energy_change

    p50_change = changes.get("timingsMilliseconds.endToEnd.p50")
    benefit = (p50_change is not None and p50_change <= -BENEFIT_GATE) or (
        energy_change is not None and energy_change <= -BENEFIT_GATE
    )
    regressions = {
        name: value for name, value in sorted(changes.items())
        if value > REGRESSION_GATE
    }
    physical = baseline["run"]["evidenceClass"] in PHYSICAL_EVIDENCE
    outputs_identical = prediction_signature(baseline) == prediction_signature(candidate)
    baseline_thermal = set(before["thermalStates"])
    candidate_thermal = set(after["thermalStates"])
    thermal_evidence_valid = (
        bool(baseline_thermal)
        and baseline_thermal == candidate_thermal
        and baseline_thermal <= COMPARABLE_THERMAL_STATES
    )
    return {
        "candidate": baseline["run"].get("candidate"),
        "tier": baseline["run"].get("tier"),
        "hardwareGroup": baseline["run"].get("hardwareGroup"),
        "deviceLabel": baseline["run"].get("deviceLabel"),
        "hardwareIdentifier": baseline["run"].get("hardwareIdentifier"),
        "evidenceClass": baseline["run"].get("evidenceClass"),
        "baselineComputeUnits": "all",
        "candidateComputeUnits": "cpuAndNeuralEngine",
        "relativeChanges": changes,
        "hasRequiredBenefit": benefit,
        "regressionsOverTenPercent": regressions,
        "ocrOutputsIdentical": outputs_identical,
        "baselineThermalStates": sorted(baseline_thermal),
        "candidateThermalStates": sorted(candidate_thermal),
        "thermalEvidenceValid": thermal_evidence_valid,
        "eligibleForProduction": (
            physical
            and thermal_evidence_valid
            and benefit
            and not regressions
            and outputs_identical
        ),
    }


def pairing_key(report: dict[str, Any]) -> tuple[Any, ...]:
    run = report["run"]
    return (
        run.get("candidate"), run.get("tier"), run.get("hardwareGroup"),
        run.get("deviceLabel"), run.get("hardwareIdentifier"), run.get("evidenceClass"),
    )


def compare_compute_units(reports: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for report in reports:
        units = report["run"]["computeUnits"]
        if units in {"all", "cpuAndNeuralEngine"}:
            key = pairing_key(report)
            if units in grouped[key]:
                raise ValueError(f"duplicate {units} report for {key}")
            grouped[key][units] = report
    return [
        paired_comparison(pair["all"], pair["cpuAndNeuralEngine"])
        for _, pair in sorted(grouped.items(), key=lambda item: str(item[0]))
        if set(pair) == {"all", "cpuAndNeuralEngine"}
    ]


def candidate_pairing_key(report: dict[str, Any]) -> tuple[Any, ...]:
    run = report["run"]
    return (
        run.get("tier"), run.get("hardwareGroup"), run.get("deviceLabel"),
        run.get("hardwareIdentifier"), run.get("evidenceClass"), run.get("computeUnits"),
    )


def compare_candidates(
    reports: Iterable[dict[str, Any]], quality_decisions: dict[str, str]
) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
    values = list(reports)
    for report in values:
        key = (str(report["run"].get("candidate")), candidate_pairing_key(report))
        if key in lookup:
            raise ValueError(f"duplicate candidate report for {key}")
        lookup[key] = report
    comparisons = []
    for candidate in values:
        run = candidate["run"]
        parent = run.get("parentCandidate")
        if not parent:
            continue
        baseline = lookup.get((str(parent), candidate_pairing_key(candidate)))
        if baseline is None:
            continue
        common = paired_comparison(baseline, candidate)
        comparisons.append({
            "baselineCandidate": parent,
            "candidate": run.get("candidate"),
            "candidateKind": run.get("candidateKind", "unspecified"),
            "tier": run.get("tier"),
            "hardwareGroup": run.get("hardwareGroup"),
            "deviceLabel": run.get("deviceLabel"),
            "hardwareIdentifier": run.get("hardwareIdentifier"),
            "evidenceClass": run.get("evidenceClass"),
            "computeUnits": run.get("computeUnits"),
            "relativeChanges": common["relativeChanges"],
            "hasRequiredBenefit": common["hasRequiredBenefit"],
            "regressionsOverTenPercent": common["regressionsOverTenPercent"],
            "ocrOutputsIdentical": common["ocrOutputsIdentical"],
            "baselineThermalStates": common["baselineThermalStates"],
            "candidateThermalStates": common["candidateThermalStates"],
            "thermalEvidenceValid": common["thermalEvidenceValid"],
            "qualityDecision": quality_decisions.get(str(run.get("candidate")), "notProvided"),
        })
    return comparisons


def w8a8_decision(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    values = [item for item in comparisons if item["candidateKind"] == "w8a8"]
    covered = {
        item["hardwareGroup"] for item in values
        if item["evidenceClass"] in PHYSICAL_EVIDENCE
        and item["thermalEvidenceValid"]
    }
    modern = [item for item in values if item["hardwareGroup"] == "A17Pro-A19"]
    legacy = [
        item for item in values if item["hardwareGroup"] in {"A12-A13", "A14-A16"}
    ]
    quality_pass = bool(values) and all(item["qualityDecision"] == "pass" for item in values)
    modern_pass = bool(modern) and all(
        item["evidenceClass"] in PHYSICAL_EVIDENCE
        and item["thermalEvidenceValid"]
        and item["hasRequiredBenefit"]
        and not item["regressionsOverTenPercent"]
        for item in modern
    )
    legacy_pass = {item["hardwareGroup"] for item in legacy} == {"A12-A13", "A14-A16"} and all(
        item["evidenceClass"] in PHYSICAL_EVIDENCE
        and item["thermalEvidenceValid"]
        and not item["regressionsOverTenPercent"]
        for item in legacy
    )
    eligible = REQUIRED_HARDWARE_GROUPS <= covered and quality_pass and modern_pass and legacy_pass
    return {
        "decision": "eligibleAsUniversalCandidate" if eligible else "notEligible",
        "coveredHardwareGroups": sorted(covered),
        "missingHardwareGroups": sorted(REQUIRED_HARDWARE_GROUPS - covered),
        "qualityGatePass": quality_pass,
        "modernBenefitGatePass": modern_pass,
        "legacyRegressionGatePass": legacy_pass,
    }


def assess(
    reports: list[dict[str, Any]],
    explanations: dict[str, str],
    quality_decisions: dict[str, str] | None = None,
) -> dict[str, Any]:
    quality_decisions = quality_decisions or {}
    summaries = []
    for report in reports:
        summaries.append({
            "run": report["run"],
            "placement": placement_summary(report, explanations),
            "performance": performance_summary(report),
        })
    comparisons = compare_compute_units(reports)
    candidate_comparisons = compare_candidates(reports, quality_decisions)
    covered = {
        item["hardwareGroup"] for item in comparisons
        if item["evidenceClass"] in PHYSICAL_EVIDENCE
        and item["thermalEvidenceValid"]
    }
    eligible = (
        REQUIRED_HARDWARE_GROUPS <= covered
        and bool(comparisons)
        and all(item["eligibleForProduction"] for item in comparisons)
    )
    return {
        "schemaVersion": 1,
        "kind": "apple-neural-engine-assessment",
        "thresholds": {
            "preferredNeuralEngineCostShare": ANE_SHARE_GATE,
            "significantFallbackCostShare": SIGNIFICANT_OPERATION_SHARE,
            "latencyOrEnergyImprovement": BENEFIT_GATE,
            "maximumRegression": REGRESSION_GATE,
            "coreMLInferenceShareStopGate": 0.40,
            "comparableThermalStates": sorted(COMPARABLE_THERMAL_STATES),
        },
        "reports": summaries,
        "pairedComputeUnitComparisons": comparisons,
        "pairedCandidateComparisons": candidate_comparisons,
        "w8a8Decision": w8a8_decision(candidate_comparisons),
        "productionComputeUnitsDecision": {
            "decision": "cpuAndNeuralEngineEligible" if eligible else "keepAll",
            "coveredHardwareGroups": sorted(covered),
            "missingHardwareGroups": sorted(REQUIRED_HARDWARE_GROUPS - covered),
            "allSuppliedComparisonsPass": bool(comparisons) and all(
                item["eligibleForProduction"] for item in comparisons
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--explanations", type=Path)
    parser.add_argument("--quality-decisions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        reports = []
        for path in args.reports:
            report = json.loads(path.read_text(encoding="utf-8"))
            validate_report(report, str(path))
            reports.append(report)
        result = assess(
            reports,
            load_explanations(args.explanations),
            load_quality_decisions(args.quality_decisions),
        )
        serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(serialized, encoding="utf-8")
        else:
            sys.stdout.write(serialized)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
