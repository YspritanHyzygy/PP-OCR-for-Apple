import unittest

from scripts.merge_reports import merge


class MergeReportTests(unittest.TestCase):
    def test_merges_distinct_raw_samples(self):
        merged = merge([
            {"schemaVersion": 1, "run": {"tier": "small"}, "samples": [{"id": "a"}]},
            {"schemaVersion": 1, "run": {"tier": "small"}, "samples": [{"id": "b"}]},
        ])
        self.assertEqual([sample["id"] for sample in merged["samples"]], ["a", "b"])
        self.assertEqual(len(merged["run"]["mergedRuns"]), 2)

    def test_rejects_duplicate_sample_ids(self):
        report = {"schemaVersion": 1, "samples": [{"id": "same"}]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge([report, report])

    def test_preserves_schema_two(self):
        merged = merge([
            {"schemaVersion": 2, "run": {}, "samples": [{"id": "a"}]},
        ])
        self.assertEqual(merged["schemaVersion"], 2)

    def test_rejects_mixed_raw_schema_versions(self):
        with self.assertRaisesRegex(ValueError, "different schema"):
            merge([
                {"schemaVersion": 1, "samples": [{"id": "a"}]},
                {"schemaVersion": 2, "samples": [{"id": "b"}]},
            ])


if __name__ == "__main__":
    unittest.main()
