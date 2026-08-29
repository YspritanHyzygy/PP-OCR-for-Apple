# OCR Lab model manifests

Each runnable ONNX asset has a sibling `*.manifest.json` that records its
source, license, exact input contract, byte size, and SHA-256. The registry only
loads files declared by one of these manifests.

Copy `detector.manifest.example.json` to `<model-id>.manifest.json`, fill every
field from the actual exported artifact, and place the `.onnx` file in this
directory. Recognizer manifests additionally lock the character dictionary,
dictionary hash, character count, and output class count. The lab verifies all
declared local assets before listing a model as available.

`db-probability-map-v1` detectors and `ctc-probabilities-v1` recognizers are
runnable today. Geometry-based horizontal/vertical classification and Apple
Vision OCR are available references. The trained Verto layout classifier is a
pending slot. A missing or invalid model remains visible as unavailable.

The tracked supplemental manifests pin PP-OCRv5 Mobile for Chinese, English,
and Japanese plus the Thai PP-OCRv5 Mobile model. Their ONNX artifacts remain
local and are prepared with the following command:

```bash
uv run python scripts/prepare_ocr_lab_models.py --component none --supplemental all
```
