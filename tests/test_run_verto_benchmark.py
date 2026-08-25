import plistlib
import tempfile
import unittest
from pathlib import Path

from scripts.run_verto_benchmark import inject_environment


class XCTestrunEnvironmentTests(unittest.TestCase):
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
