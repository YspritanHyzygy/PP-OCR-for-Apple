# Verto OCR Lab

Local browser workbench for inspecting text-region geometry, layout decisions,
script routing, and stage timings without rebuilding the iOS app.

## Run

```bash
uv sync
npm --prefix ocr_lab/frontend install
npm --prefix ocr_lab/frontend run build
uv run uvicorn ocr_lab.app:app --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765>.

For frontend development, run `npm --prefix ocr_lab/frontend run dev` in a
second terminal. Vite proxies `/api` to port `8765`.

Uploaded images stay in memory for the request and are discarded afterward.
Training-data collection is outside this tool. The bundled sample is a fixture;
its recognized text and regions are explicitly identified as fixture data.

`reference-geometry` is a deterministic visual baseline. Apple Vision OCR is
available on macOS as an independent detector/recognizer reference. The
production Verto detector and classifier remain pending. An unavailable model
selection returns a visible error, while fixture output stays limited to the
bundled sample.

## Local ONNX models

Set `OCR_LAB_MODEL_ROOT` to a directory containing verified
`*.manifest.json` files, or put them in `ocr_lab/models`. The lab checks each
artifact's byte size and SHA-256 before making it selectable. The implemented
runtime contracts are `db-probability-map-v1` for detectors and
`ctc-probabilities-v1` for recognizers; see the examples in `ocr_lab/models`.

The bundled `verto-text-detector` and `TextLayoutClassifier` entries are
pending slots. Trained Verto detector and classifier assets remain pending.

Prepare the locked official PP-OCRv6 Tiny, Small, and Medium tiers plus the
PP-OCRv5 multilingual and Thai mobile recognizers:

```bash
uv run python scripts/prepare_ocr_lab_models.py --tier all --component all --supplemental all
```

The generated ONNX files stay local. Their manifests record the locked Hugging
Face revision, upstream BGR preprocessing, model contract, byte size, and
SHA-256. Detector manifests also record DB thresholds; recognizer manifests
lock the character dictionary and class count. Restart FastAPI after preparing
models so the registry reloads the verified manifests.

The lab defaults to the PP-OCRv6 Tiny detector and `Verto Auto`. The primary
recognizer is PP-OCRv5 Mobile, which covers Chinese, English, Japanese, and
vertical Japanese. Regions below 0.75 recognition confidence use a local Apple
Vision crop fallback for scripts such as Korean and Thai. The Thai-specific
PP-OCRv5 Mobile recognizer remains directly selectable for comparison.

Every detector region remains an independent line geometry. After recognition,
the lab links same-baseline fragments and adjacent lines into translation
context groups. `G3 / L2`, for example, means translation group 3, logical line
2. Group text preserves newlines while erasing, selection, and rendering can
continue to use the original region quads.

On macOS, the Apple Vision reference helper is compiled on first use with the
installed Xcode command-line tools. It runs `VNRecognizeTextRequest` locally and
does not upload the image. Full-image Vision does not reliably detect genuine
vertical Japanese; the PP-OCRv5 Mobile reference handles that case after the
detector quad is rectified.

## Full translation pipeline

The `运行整套流程` panel calls `/api/pipeline`. It can combine local OCR or
Google Cloud Vision, Google Cloud Translation, and the development-only MI-GAN
512 inpainting candidate. The server runs OCR, translation, repair, and text
layout fully in memory and returns one final JPEG only after every stage has
completed. The browser never receives a progressively repaired background.

MI-GAN uses one 512x512 inference for the complete working image. Every
detected quadrilateral is painted into one combined hole mask; generated pixels
are blended back only through that mask, so pixels outside the detected text
regions remain from the original image. The model is loaded from the explicit
`OCR_LAB_MIGAN_MODEL` (or `MIGAN_MODEL_PATH`) path and is never copied into the
repository. The current candidate's redistribution status is unconfirmed.

For Google Cloud testing, configure credentials in the FastAPI process, never
in the browser:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/service-account.json
export GOOGLE_CLOUD_PROJECT=your-project-id
export OCR_LAB_MIGAN_MODEL=/absolute/path/MIGAN512Places2.mlpackage
uv sync
uv run uvicorn ocr_lab.app:app --host 127.0.0.1 --port 8765
```

For a short-lived local smoke test, `GOOGLE_OCR_ACCESS_TOKEN` can supply a
developer bearer token instead of ADC. The server calls Vision's
`DOCUMENT_TEXT_DETECTION` and Translation v3 directly, keeps Google errors
visible, and does not silently fall back to another OCR provider.

When Google Cloud Vision is selected, the lab exposes two independent OCR
choices: `DOCUMENT_TEXT_DETECTION` for dense document text and `TEXT_DETECTION`
for scene text. The result can either keep Google's original Paragraph
boundaries or pass the returned lines through Verto's deterministic geometry
grouping. The standalone `仅测试 Google OCR` button uses these choices without
calling Translation or MI-GAN.
