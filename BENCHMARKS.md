# Benchmarks

This document defines what a performance or quality claim must contain. Version-specific results live under [`benchmarks/`](benchmarks); README text links to those reports instead of copying values that can go stale.

## Required context

Every local performance result must record:

- release tag and asset SHA-256;
- device model, chip, memory, OS, and Core ML version context;
- Core ML compute-unit selection;
- input tensor shape and data type;
- warm-up count, measured iteration count, and reported statistic;
- whether model loading/compilation is included.

## Separate result classes

- **Upstream quality:** PaddlePaddle's published dataset and metric, clearly attributed.
- **Conversion fidelity:** maximum absolute output difference between pinned ONNX and released Core ML packages on deterministic inputs.
- **Apple runtime:** measured load and prediction time on the named Apple device.
- **Application quality:** end-to-end OCR measured by a consuming application, never inferred from recognition-only scores.

An optimization may be called faster, smaller, or more accurate only when its versioned report compares it with the previous published release using the same method. Candidate quantization, shape, or preprocessing changes remain hypotheses until they clear both the accuracy and runtime gates.

Application-quality reports follow [`evaluation/metrics-v1.md`](evaluation/metrics-v1.md). Public and private corpus requirements live in [`evaluation/corpus.lock.json`](evaluation/corpus.lock.json); a release workflow refuses to publish while that lock reports unresolved data or checksum blockers.

Raw inference output is evidence, not a verdict. The consuming application emits source polygons, text, timings, and environment data. [`scripts/evaluate_results.py`](scripts/evaluate_results.py) applies matching, Unicode normalization, CER, slice aggregation, and paired bootstrap rules afterwards.

[`scripts/run_verto_benchmark.py`](scripts/run_verto_benchmark.py) builds the existing Verto test target, injects model and corpus paths into a generated `.xctestrun`, and runs Verto's real preprocessing, detector post-processing, crop, recognizer, and decoder implementation. This avoids a second OCR implementation in the model repository.

Public archives are converted with [`scripts/prepare_corpus.py`](scripts/prepare_corpus.py). The locked Noto Sans font plus [`scripts/generate_diacritic_corpus.py`](scripts/generate_diacritic_corpus.py) deterministically generates the French, Spanish, and German set. Dataset runs remain separate raw evidence files and are combined, without rescoring or rewriting samples, by [`scripts/merge_reports.py`](scripts/merge_reports.py).

## Published reports

- [`v1.json`](benchmarks/v1.json) — migrated PP-OCRv6 Core ML baseline

## Candidate reports

- [`v2-candidates.json`](benchmarks/v2-candidates.json) — conversion and Mac-only single-variable screening; blocked candidates are not release recommendations
- [`v2-public-screening.json`](benchmarks/v2-public-screening.json) — available public-corpus results, rejected candidates, and explicit release blockers
