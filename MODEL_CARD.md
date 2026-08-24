# Model card

## Model family

PP-OCR for Apple packages the official PP-OCRv6 detector and recognizer tiers for Core ML. It does not train a separate OCR model and does not claim authorship of PP-OCRv6.

The exact detector and recognizer repositories and revisions are locked in [`sources.lock.json`](sources.lock.json). Published archive names, sizes, SHA-256 digests, conversion errors, and preprocessing parameters live in the matching versioned file under [`manifests/`](manifests).

## Intended use

- On-device text detection and recognition in iOS and macOS applications using Core ML.
- Applications that can download a pinned model package separately from the app bundle.
- Printed, handwritten, rotated, curved, and scene text within PP-OCRv6's documented coverage.

The current packages are used by [Verto](https://github.com/YspritanHyzygy/Verto). They are model assets, not a complete OCR SDK: consumers must implement image preprocessing, DB detector post-processing, line cropping, and CTC decoding consistently with the release manifest.

## Conversion

The official ONNX graphs are converted without retraining:

1. Rewrite unsupported `SAME_UPPER` nodes to explicit padding and require bit-identical ONNX output.
2. Trace the fixed detector and recognizer input shapes.
3. Convert weights and compute to Float16 while keeping model inputs and outputs Float32.
4. Compare Core ML output with the official ONNX output using deterministic inputs. Detection and recognition use explicit error bounds; recognition also reports CTC argmax differences as a diagnostic. End-to-end OCR probes, not random tensors, decide whether decoded text remains usable.
5. Package the two `.mlpackage` directories and the unmodified character set as an LZFSE-compressed Apple Archive.

## Language and script limitations

- None of the current PP-OCRv6 recognizers contains Hangul, so Korean text needs another recognizer.
- The tiny recognizer omits Japanese kana; Japanese requires the small or medium tier.
- “50 languages” in the upstream model card covers Chinese, Japanese, Greek, and Latin-script languages. It does not mean universal script coverage.
- Recognition quality depends on detection, perspective correction, cropping, decoding, and image conditions; an upstream recognition score is not an end-to-end application score.

## Evaluation

Upstream accuracy values are attributed to PaddlePaddle and are never presented as measurements made by this project. Core ML fidelity, package size, and Apple-device timing are reported separately under [`benchmarks/`](benchmarks), with the hardware, OS, compute-unit setting, warm-up policy, and sample count attached to every local measurement.

## License and citation

The upstream PP-OCRv6 model cards declare Apache-2.0. See the [official PP-OCRv6 collection](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6) and its linked technical report for the authoritative model description and citation.
