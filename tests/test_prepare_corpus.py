import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_corpus import prepare_ctw1500, prepare_total_text


class TotalTextPreparationTests(unittest.TestCase):
    def test_prepares_images_and_excludes_placeholder_transcriptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "total_text.tar"
            annotations = (
                'rgb/img1.jpg\t[{"transcription":"OPEN","points":[[0,0],[2,0],[2,2],[0,2]]},'
                '{"transcription":"chinese_text","points":[[3,0],[5,0],[5,2],[3,2]]}]\n'
            ).encode()
            with tarfile.open(archive, "w") as target:
                for name, payload in (
                    ("total_text/test/test.txt", annotations),
                    ("total_text/test/rgb/img1.jpg", b"fixture-image"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    target.addfile(info, io.BytesIO(payload))
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()

            with patch(
                "scripts.prepare_corpus.locked_dataset",
                return_value={"archiveSHA256": checksum},
            ):
                corpus_path = prepare_total_text(archive, root / "prepared")

            corpus = json.loads(corpus_path.read_text())
            self.assertEqual(corpus["schemaVersion"], 1)
            self.assertEqual(len(corpus["samples"]), 1)
            lines = corpus["samples"][0]["groundTruth"]
            self.assertFalse(lines[0]["ignore"])
            self.assertTrue(lines[1]["ignore"])
            self.assertEqual((root / "prepared/images/img1.jpg").read_bytes(), b"fixture-image")

    def test_prepares_ctw1500_as_detection_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "ctw1500.zip"
            annotations = (
                'test/1001.jpg\t[{"transcription":0,"points":'
                '[[0,0],[2,0],[2,2],[0,2]]}]\n'
            )
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("ctw1500/imgs/test.txt", annotations)
                target.writestr("ctw1500/imgs/test/1001.jpg", b"fixture-image")
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()

            with patch(
                "scripts.prepare_corpus.locked_dataset",
                return_value={"archiveSHA256": checksum},
            ):
                corpus_path = prepare_ctw1500(archive, root / "prepared")

            sample = json.loads(corpus_path.read_text())["samples"][0]
            self.assertEqual(sample["evaluationTask"], "detection")
            self.assertEqual(sample["groundTruth"][0]["text"], "")
            self.assertEqual((root / "prepared/images/1001.jpg").read_bytes(), b"fixture-image")


if __name__ == "__main__":
    unittest.main()
