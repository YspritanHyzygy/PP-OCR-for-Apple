import unittest

from scripts.evaluate_ane import assess, placement_summary, validate_report


def make_report(
    hardware_group="A12-A13",
    compute_units="all",
    end_to_end=100.0,
    evidence_class="localPhysical",
    candidate="b1-small",
    candidate_kind="reference",
    parent_candidate=None,
):
    return {
        "schemaVersion": 2,
        "run": {
            "candidate": candidate,
            "candidateKind": candidate_kind,
            "parentCandidate": parent_candidate,
            "tier": "small",
            "systemVersion": "17.4",
            "operatingSystem": "iOS 17.4",
            "deviceLabel": hardware_group,
            "hardwareIdentifier": "iPhone-test",
            "hardwareGroup": hardware_group,
            "evidenceClass": evidence_class,
            "computeUnits": compute_units,
            "availableComputeDevices": ["cpu", "gpu", "neuralEngine"],
            "warmupRuns": 3,
            "measurementRunsPerSample": 1,
            "thermalStateStart": "nominal",
            "thermalStateEnd": "nominal",
            "energy": {"source": "notMeasured", "joules": None},
        },
        "computePlans": [
            {
                "component": "detector",
                "status": "available",
                "operations": [
                    {
                        "path": "model/0:conv",
                        "operatorName": "conv",
                        "estimatedCostWeight": 0.95,
                        "preferredComputeDevice": "neuralEngine",
                        "supportedComputeDevices": ["cpu", "neuralEngine"],
                        "fallbackReason": None,
                    },
                    {
                        "path": "model/1:shape",
                        "operatorName": "shape",
                        "estimatedCostWeight": 0.05,
                        "preferredComputeDevice": "cpu",
                        "supportedComputeDevices": ["cpu", "neuralEngine"],
                        "fallbackReason": "coreMLPreferredOtherDevice",
                    },
                    {
                        "path": "model/2:const",
                        "operatorName": "const",
                        "estimatedCostWeight": None,
                        "preferredComputeDevice": None,
                        "supportedComputeDevices": [],
                        "fallbackReason": "deviceUsageUnavailable",
                    },
                ],
                "error": None,
            },
            {
                "component": "recognizer",
                "status": "available",
                "operations": [
                    {
                        "path": "model/0:conv",
                        "operatorName": "conv",
                        "estimatedCostWeight": 1.0,
                        "preferredComputeDevice": "neuralEngine",
                        "supportedComputeDevices": ["cpu", "neuralEngine"],
                        "fallbackReason": None,
                    }
                ],
                "error": None,
            },
        ],
        "samples": [
            {
                "id": "sample",
                "detectionPredictions": [],
                "predictions": [{"polygon": [[0, 0], [1, 0], [1, 1]], "text": "ok"}],
                "measurements": [
                    {
                        "timingsMilliseconds": {
                            "preprocessing": 20.0,
                            "detector": end_to_end * 0.1,
                            "detectionPostProcess": 10.0,
                            "recognizer": end_to_end * 0.1,
                            "decode": 5.0,
                            "endToEnd": end_to_end,
                        },
                        "thermalStateBefore": "nominal",
                        "thermalStateAfter": "nominal",
                        "residentMemoryBytesBefore": 900,
                        "residentMemoryBytesAfter": 1000,
                        "processLifetimePeakResidentBytes": 1000,
                    }
                ],
            }
        ],
    }


class ANEAssessmentTests(unittest.TestCase):
    def test_significant_fallback_requires_a_separate_explanation(self):
        report = make_report()
        without = placement_summary(report, {})[0]
        self.assertFalse(without["aneFriendly"])
        self.assertEqual(len(without["significantFallbackOperations"]), 1)

        with_explanation = placement_summary(
            report,
            {"detector:model/1:shape": "Shape bookkeeping remains on the CPU."},
        )[0]
        self.assertTrue(with_explanation["aneFriendly"])

    def test_cpu_and_neural_engine_needs_all_three_physical_groups(self):
        reports = []
        for group in ("A12-A13", "A14-A16", "A17Pro-A19"):
            reports.append(make_report(group, "all", 100.0))
            reports.append(make_report(group, "cpuAndNeuralEngine", 80.0))
        result = assess(reports, {})
        self.assertEqual(
            result["productionComputeUnitsDecision"]["decision"],
            "cpuAndNeuralEngineEligible",
        )

        result = assess(reports[:-2], {})
        self.assertEqual(result["productionComputeUnitsDecision"]["decision"], "keepAll")
        self.assertEqual(
            result["productionComputeUnitsDecision"]["missingHardwareGroups"],
            ["A17Pro-A19"],
        )

    def test_simulator_cannot_make_a_production_compute_unit_claim(self):
        before = make_report(None, "all", 100.0, "simulator")
        after = make_report(None, "cpuAndNeuralEngine", 50.0, "simulator")
        result = assess([before, after], {})
        comparison = result["pairedComputeUnitComparisons"][0]
        self.assertFalse(comparison["eligibleForProduction"])
        self.assertEqual(result["productionComputeUnitsDecision"]["decision"], "keepAll")

    def test_core_ml_below_forty_percent_redirects_optimization(self):
        result = assess([make_report()], {})
        performance = result["reports"][0]["performance"]
        self.assertEqual(performance["coreMLInferenceFraction"], 0.2)
        self.assertEqual(performance["optimizationFocus"], "cpuPreAndPostProcessing")

    def test_raw_schema_rejects_unlabelled_physical_evidence(self):
        report = make_report()
        report["run"]["hardwareGroup"] = None
        with self.assertRaisesRegex(ValueError, "hardwareGroup"):
            validate_report(report, "fixture")

    def test_w8a8_requires_quality_modern_benefit_and_legacy_guards(self):
        reports = []
        for group in ("A12-A13", "A14-A16", "A17Pro-A19"):
            reports.append(make_report(group, "all", 100.0, candidate="small-rec320"))
            reports.append(make_report(
                group,
                "all",
                80.0 if group == "A17Pro-A19" else 100.0,
                candidate="small-rec320-w8a8",
                candidate_kind="w8a8",
                parent_candidate="small-rec320",
            ))
        result = assess(reports, {}, {"small-rec320-w8a8": "pass"})
        self.assertEqual(
            result["w8a8Decision"]["decision"], "eligibleAsUniversalCandidate"
        )

        result = assess(reports, {}, {"small-rec320-w8a8": "fail"})
        self.assertEqual(result["w8a8Decision"]["decision"], "notEligible")


if __name__ == "__main__":
    unittest.main()
