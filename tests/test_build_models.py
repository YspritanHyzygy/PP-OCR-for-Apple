import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.build_models import (
    apply_compression,
    calibration_samples,
    load_calibration_corpus,
    load_recipe,
)
from scripts.check_project import validate_ane_compatibility_status


class W8A8BuildTests(unittest.TestCase):
    def test_pending_ane_matrix_blocks_release_readiness(self):
        report = json.loads(
            Path("benchmarks/ane-compatibility-v2.json").read_text(encoding="utf-8")
        )
        validate_ane_compatibility_status(report, require_ready=False)
        with self.assertRaises(AssertionError):
            validate_ane_compatibility_status(report, require_ready=True)

    def test_candidate_recipe_selects_w8a8_only_for_small(self):
        recipe = load_recipe(Path("recipes/candidates/small-rec320-w8a8.json"))
        self.assertEqual(recipe["tiers"]["small"]["detectorCompression"], "w8a8")
        self.assertEqual(recipe["tiers"]["small"]["recognizerCompression"], "w8a8")
        self.assertEqual(recipe["tiers"]["tiny"]["detectorCompression"], "none")
        self.assertEqual(recipe["tiers"]["medium"]["recognizerCompression"], "none")

    def test_w8a8_runs_activation_calibration_before_weight_quantization(self):
        model = object()
        activated = object()
        compressed = object()
        samples = [{"x": np.zeros((1, 3, 4, 4), dtype=np.float32)}]
        with (
            patch(
                "scripts.build_models.cto.coreml.linear_quantize_activations",
                return_value=activated,
            ) as activations,
            patch(
                "scripts.build_models.cto.coreml.linear_quantize_weights",
                return_value=compressed,
            ) as weights,
        ):
            result = apply_compression(model, "w8a8", samples)
        self.assertIs(result, compressed)
        self.assertIs(activations.call_args.args[0], model)
        self.assertIs(weights.call_args.args[0], activated)

    def test_w8a8_rejects_missing_calibration_samples(self):
        with self.assertRaisesRegex(SystemExit, "calibration"):
            apply_compression(object(), "w8a8", None)

    def test_public_corpus_generates_detector_and_recognizer_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 10), (128, 64, 32)).save(root / "sample.png")
            (root / "corpus.json").write_text(json.dumps({
                "schemaVersion": 1,
                "samples": [{
                    "imagePath": "sample.png",
                    "groundTruth": [{
                        "polygon": [[1, 1], [19, 1], [19, 9], [1, 9]],
                        "text": "sample",
                        "ignore": False,
                    }],
                }],
            }), encoding="utf-8")
            corpus = load_calibration_corpus(root / "corpus.json")
            contract = {
                "upstreamMean": [0.406, 0.456, 0.485],
                "upstreamStd": [0.225, 0.224, 0.229],
            }
            detector = calibration_samples(corpus, "detector", (1, 3, 8, 8), contract)
            recognizer = calibration_samples(corpus, "recognizer", (1, 3, 4, 16), {})
            self.assertEqual(detector[0]["x"].shape, (1, 3, 8, 8))
            self.assertEqual(recognizer[0]["x"].shape, (1, 3, 4, 16))
            self.assertTrue(np.isfinite(detector[0]["x"]).all())
            self.assertTrue(np.isfinite(recognizer[0]["x"]).all())


if __name__ == "__main__":
    unittest.main()
