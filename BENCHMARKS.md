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

## Published reports

- [`v1.json`](benchmarks/v1.json) — migrated PP-OCRv6 Core ML baseline
