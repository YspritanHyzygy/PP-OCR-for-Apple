import unittest

from scripts.evaluate_results import (
    aggregate,
    bootstrap_delta,
    graphemes,
    levenshtein,
    normalize_text,
    score_report,
    score_sample,
)


def line(text, x0=0, y0=0, x1=10, y1=10, ignore=False):
    return {
        "text": text,
        "ignore": ignore,
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


def sample(ground_truth, predictions):
    return {
        "id": "sample-1",
        "dataset": "fixture",
        "language": "en",
        "scenario": "screen-or-ui",
        "groundTruth": ground_truth,
        "predictions": predictions,
    }


class TextMetricTests(unittest.TestCase):
    def test_normalization_preserves_case_punctuation_and_internal_space(self):
        self.assertEqual(normalize_text(" \r\nCafé,  OK\t"), "Café,  OK")
        self.assertEqual(graphemes("e\N{COMBINING ACUTE ACCENT}"), ["é"])

    def test_levenshtein_counts_grapheme_edits(self):
        self.assertEqual(levenshtein(graphemes("café"), graphemes("cafe")), 1)

    def test_matching_scores_text_and_detection(self):
        result = score_sample(sample([line("Open")], [line("0pen")]))
        self.assertEqual(result["truePositives"], 1)
        self.assertEqual(result["edits"], 1)
        metrics = aggregate([result])
        self.assertEqual(metrics["detectionHmean"], 1)
        self.assertEqual(metrics["characterErrorRate"], 0.25)

    def test_unmatched_text_contributes_deletions_and_insertions(self):
        result = score_sample(
            sample([line("abc", 0, 0, 10, 10)], [line("xy", 20, 20, 30, 30)])
        )
        self.assertEqual(result["falseNegatives"], 1)
        self.assertEqual(result["falsePositives"], 1)
        self.assertEqual(result["edits"], 5)

    def test_prediction_inside_ignored_region_is_not_false_positive(self):
        result = score_sample(sample([line("###", ignore=True)], [line("noise")]))
        self.assertEqual(result["falsePositives"], 0)

    def test_detection_only_sample_does_not_invent_text_metrics(self):
        value = sample([line("")], [line("predicted text")])
        value["evaluationTask"] = "detection"
        result = score_sample(value)
        self.assertEqual(result["truePositives"], 1)
        self.assertEqual(result["textScoredMatches"], 0)
        self.assertEqual(result["edits"], 0)
        self.assertEqual(result["groundTruthCharacters"], 0)
        metrics = aggregate([result])
        self.assertIsNone(metrics["characterErrorRate"])
        self.assertIsNone(metrics["exactLineAccuracy"])
        self.assertIsNone(metrics["lineRecall"])

    def test_detection_metrics_do_not_depend_on_recognizer_output(self):
        value = sample([line("Open")], [])
        value["detectionPredictions"] = [line("")]
        result = score_sample(value)
        self.assertEqual(result["truePositives"], 1)
        self.assertEqual(result["recognizedTruePositives"], 0)
        self.assertEqual(result["edits"], 4)
        self.assertEqual(aggregate([result])["lineRecall"], 0)

    def test_bootstrap_is_paired_and_deterministic(self):
        baseline = [
            {**score_sample(sample([line("abc")], [line("abx")])), "id": "a"},
            {**score_sample(sample([line("abc")], [line("abc")])), "id": "b"},
        ]
        candidate = [
            {**score_sample(sample([line("abc")], [line("abc")])), "id": "a"},
            {**score_sample(sample([line("abc")], [line("abc")])), "id": "b"},
        ]
        first = bootstrap_delta(baseline, candidate)
        second = bootstrap_delta(baseline, candidate)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["lower95"], 0)

    def test_schema_two_uses_every_raw_timing_measurement(self):
        value = sample([line("Open")], [line("Open")])
        value["measurements"] = [
            {"timingsMilliseconds": {"endToEnd": 10}},
            {"timingsMilliseconds": {"endToEnd": 30}},
        ]
        report, _ = score_report({"schemaVersion": 2, "samples": [value]})
        timing = report["metrics"]["timingsMilliseconds"]["endToEnd"]
        self.assertEqual(timing["count"], 2)
        self.assertEqual(timing["p50"], 20)


if __name__ == "__main__":
    unittest.main()
