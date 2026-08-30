from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_migan", ROOT / "scripts/check_migan.py"
)
assert SPEC and SPEC.loader
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class MIGANValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = json.loads((ROOT / "migan/sources.lock.json").read_text())
        self.license = json.loads(
            (ROOT / "migan/redistribution-license.json").read_text()
        )
        self.benchmark = json.loads((ROOT / "benchmarks/migan-v1.json").read_text())
        self.device_matrix = json.loads(
            (ROOT / "benchmarks/migan-device-matrix-v1.json").read_text()
        )

    def test_committed_source_lock_is_valid(self) -> None:
        CHECK.validate_source_lock(self.lock)

    def test_unconfirmed_license_is_valid_for_research_and_blocks_release(self) -> None:
        CHECK.validate_license_gate(self.lock, self.license, False)
        with self.assertRaisesRegex(ValueError, "release is blocked"):
            CHECK.validate_license_gate(self.lock, self.license, True)

    def test_confirmed_license_requires_matching_checked_in_evidence(self) -> None:
        record = copy.deepcopy(self.license)
        record.update({"status": "confirmed", "confirmedAt": "2026-08-25"})
        record["scopes"] = {key: True for key in record["scopes"]}
        with self.assertRaisesRegex(ValueError, "no pinned rights-holder confirmation"):
            CHECK.validate_license_gate(self.lock, record, True)

    def test_benchmark_is_structurally_valid_and_preserves_trace_limit(self) -> None:
        CHECK.validate_benchmark(self.benchmark, False)
        with self.assertRaisesRegex(ValueError, "execution device is unconfirmed"):
            CHECK.validate_benchmark(
                self.benchmark,
                True,
                self.benchmark["modelTreeSHA256"],
            )

    def test_complete_benchmark_must_match_the_built_model(self) -> None:
        report = copy.deepcopy(self.benchmark)
        report.update(
            {
                "status": "complete",
                "modelTreeSHA256": "1" * 64,
                "device": "Physical iPhone",
                "os": "iOS",
                "thermalState": "nominal",
                "coldLoadMilliseconds": 1,
                "predictionMilliseconds": [1, 1, 1],
                "medianPredictionMilliseconds": 1,
                "observedResidentHighWaterBytes": 1,
                "coreMLInstruments": {
                    "status": "confirmed",
                    "executionDevice": "Neural Engine",
                    "traceSHA256": "2" * 64,
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "differs from the manifest"):
            CHECK.validate_benchmark(report, True, "3" * 64)
        with self.assertRaisesRegex(ValueError, "no pinned Instruments trace"):
            CHECK.validate_benchmark(report, True, "1" * 64)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / CHECK.EXPECTED_INSTRUMENTS_TRACE_FILE
            trace.parent.mkdir(parents=True)
            trace.write_bytes(b"real trace fixture")
            report["coreMLInstruments"].update(
                {
                    "computeUnits": "ALL",
                    "modelTreeSHA256": "1" * 64,
                    "traceFile": CHECK.EXPECTED_INSTRUMENTS_TRACE_FILE,
                    "traceSHA256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                }
            )
            with mock.patch.object(
                CHECK,
                "PINNED_INSTRUMENTS_TRACE_SHA256",
                hashlib.sha256(trace.read_bytes()).hexdigest(),
            ):
                CHECK.validate_benchmark(report, True, "1" * 64, root)

    def test_complete_benchmark_recomputes_the_reported_median(self) -> None:
        report = copy.deepcopy(self.benchmark)
        report["medianPredictionMilliseconds"] = 999
        with self.assertRaisesRegex(ValueError, "median"):
            CHECK.validate_benchmark(report, False)

    def test_manifest_contract_rejects_output_type_drift(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "packVersion": "1",
            "releaseTag": "migan-v1",
            "licenseGate": copy.deepcopy(self.license),
            "source": {
                "upstreamCode": self.lock["upstreamCode"],
                "checkpoint": self.lock["checkpoint"],
            },
            "conversion": {
                key: copy.deepcopy(self.lock["conversion"][key])
                for key in (
                    "modelName",
                    "minimumDeploymentTarget",
                    "modelType",
                    "computePrecision",
                    "computeUnits",
                    "inputs",
                    "output",
                )
            },
            "modelPackage": {"bytes": 1, "treeSHA256": "2" * 64},
            "provenance": {
                "schemaVersion": 1,
                "upstreamCodeRevision": CHECK.EXPECTED_CODE_REVISION,
                "checkpointSHA256": CHECK.EXPECTED_CHECKPOINT_SHA,
                "modelPackage": {"bytes": 1, "treeSHA256": "2" * 64},
                "evidence": {
                    "upstreamLicenseSHA256": CHECK.EXPECTED_UPSTREAM_LICENSE_SHA,
                    "redistributionEvidenceSHA256": self.license["evidenceSHA256"],
                },
            },
            "parity": {
                "fixtures": [
                    {
                        "inputSHA256": "0" * 64,
                        "maximumAbsoluteError": 0.01,
                        "meanAbsoluteError": 0.001,
                    }
                ]
            },
            "archive": {
                "name": "migan-512-places2-coreml-v1.aar",
                "bytes": 1,
                "sha256": "1" * 64,
            },
        }
        CHECK.validate_manifest(manifest, self.lock, self.license, None)
        manifest["conversion"]["output"]["dataType"] = "FLOAT16"
        with self.assertRaisesRegex(ValueError, "output drifted"):
            CHECK.validate_manifest(manifest, self.lock, self.license, None)

    def test_device_matrix_is_valid_and_xr_cannot_claim_execution_device(self) -> None:
        CHECK.validate_device_matrix(self.device_matrix)
        matrix = copy.deepcopy(self.device_matrix)
        xr = next(item for item in matrix["devices"] if item["productType"] == "iPhone11,8")
        xr["executionDevice"] = "Apple Neural Engine"
        with self.assertRaisesRegex(ValueError, "overclaims"):
            CHECK.validate_device_matrix(matrix)


if __name__ == "__main__":
    unittest.main()
