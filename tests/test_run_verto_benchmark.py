import plistlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.run_verto_benchmark import (
    build_xctestruns,
    configured_xctestrun,
    exported_attachment,
    inject_environment,
    stage_corpus,
)


class XCTestrunEnvironmentTests(unittest.TestCase):
    def test_stages_corpus_images_with_device_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (4, 3), "white").save(source / "image.png")
            corpus = source / "corpus.json"
            corpus.write_text(json.dumps({
                "schemaVersion": 1,
                "samples": [{"id": "one", "imagePath": "image.png"}],
            }), encoding="utf-8")

            staged = stage_corpus(corpus, root / "bundle" / "corpus")

            document = json.loads(staged.read_text(encoding="utf-8"))
            self.assertEqual(document["samples"][0]["imagePath"], "images/00000.png")
            self.assertTrue((staged.parent / "images/00000.png").is_file())

    def test_selects_the_named_attachment_from_manifest(self):
        manifest = [{"attachments": [{
            "exportedFileName": "uuid.json",
            "suggestedHumanReadableName": "verto-ocr-report_0_uuid.json",
        }]}]
        self.assertEqual(
            exported_attachment(manifest, "verto-ocr-report"), "uuid.json"
        )

    def test_configured_copy_stays_beside_and_is_not_a_build_product(self):
        with tempfile.TemporaryDirectory() as directory:
            products = Path(directory)
            built = products / "Verto_iphoneos.xctestrun"
            built.touch()
            configured = configured_xctestrun(
                products, "small", "small/rec320", "cpuAndNeuralEngine"
            )
            configured.touch()
            self.assertEqual(configured.parent, products)
            self.assertEqual(
                configured.name,
                "Configured-Verto_small_small-rec320_cpuAndNeuralEngine.xctestrun",
            )
            self.assertEqual(build_xctestruns(products), [built])

    def test_injects_only_the_unit_test_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xctestrun"
            destination = root / "configured.xctestrun"
            with source.open("wb") as handle:
                plistlib.dump(
                    {
                        "VertoTests": {"EnvironmentVariables": {"EXISTING": "1"}},
                        "VertoUITests": {"EnvironmentVariables": {"UNCHANGED": "1"}},
                    },
                    handle,
                )

            inject_environment(source, destination, {"VERTO_OCR_MODEL_TIER": "tiny"})

            with destination.open("rb") as handle:
                configured = plistlib.load(handle)
            self.assertEqual(
                configured["VertoTests"]["EnvironmentVariables"],
                {"EXISTING": "1", "VERTO_OCR_MODEL_TIER": "tiny"},
            )
            self.assertEqual(
                configured["VertoUITests"]["EnvironmentVariables"], {"UNCHANGED": "1"}
            )


if __name__ == "__main__":
    unittest.main()
