# PP-OCR for Apple

Unofficial PP-OCR model packages for Core ML on Apple platforms, with a reproducible conversion pipeline and measured release metadata.

This project is not affiliated with or endorsed by PaddlePaddle or Apple. The model architecture and upstream ONNX exports come from [PaddlePaddle's PP-OCRv6 collection](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6). Verto uses these packages for its optional high-accuracy, on-device camera OCR path.

## Release packages

The [latest release](https://github.com/YspritanHyzygy/PP-OCR-for-Apple/releases/latest) publishes three tiers:

- `pp-ocr-v6-coreml-tiny-v<version>.aar`
- `pp-ocr-v6-coreml-small-v<version>.aar`
- `pp-ocr-v6-coreml-medium-v<version>.aar`

Each Apple Archive contains one Core ML text detector, one Core ML text recognizer, and the recognizer character set. Release assets also include `manifest.json` and `SHA256SUMS`; consumers should pin a release and verify both byte length and SHA-256 rather than following `latest` at runtime.

## Apple conversion

- Fixed detector input: `1 x 3 x 960 x 960`
- Fixed recognizer input: `1 x 3 x 48 x 640`
- Float16 model compute with Float32 inputs and outputs
- Minimum Core ML deployment target: iOS 17
- Apple Archive with LZFSE compression

Fixed input shapes keep Core ML eligible for Apple Neural Engine execution. The conversion pipeline checks that rewriting ONNX `SAME_UPPER` padding is bit-exact and rejects Core ML conversion error above the recorded limits.

## Rebuild

The build is pinned to Python 3.12, a committed `uv.lock`, and exact upstream Hugging Face revisions in [`sources.lock.json`](sources.lock.json).

```bash
uv sync --locked --python 3.12
uv run python scripts/build_models.py --out dist
python3 scripts/check_project.py --manifest dist/manifest.json
```

The build downloads the pinned official ONNX exports, validates the graph rewrite and Core ML conversion, then writes the three archives, `manifest.json`, and `SHA256SUMS`.

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md) — provenance, intended use, coverage, and limitations
- [`BENCHMARKS.md`](BENCHMARKS.md) — measurement rules and versioned reports
- [`manifests/`](manifests) — immutable metadata for published releases

## License

The conversion code and redistributed model derivatives are available under Apache License 2.0. See [`NOTICE`](NOTICE) for upstream attribution. PP-OCR and PaddlePaddle names belong to their respective owners; Apple and Core ML are trademarks of Apple Inc.
