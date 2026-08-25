# Apple Neural Engine compatibility

## Current decision

Verto continues to load every production OCR model with `MLComputeUnits.all`. There is no user-facing NPU switch, chip allowlist, or hardware-specific download package. This is a deliberate pending-evidence decision: Core ML can already schedule supported operations across the Neural Engine, GPU, and CPU, while the project does not yet have paired physical-device results for the A12, A16, and A18 Pro hardware groups.

Every iPhone that can run Verto's minimum iOS 17 target has an Apple Neural Engine. The compatibility question is therefore not whether an NPU exists, but whether each Core ML operation is placed there and whether forcing CPU plus Neural Engine improves the complete OCR pipeline. Apple exposes compute-unit policy, not individual Neural Engine core selection.

Primary references:

- [iOS 17 compatible iPhones](https://www.apple.com/newsroom/2023/09/ios-17-is-available-today/)
- [A12 Bionic Neural Engine](https://www.apple.com/newsroom/2018/09/iphone-xs-and-iphone-xs-max-bring-the-best-and-biggest-displays-to-iphone/)
- [Core ML compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits)
- [Core ML compute-plan inspection](https://developer.apple.com/videos/play/wwdc2024/10161/)
- [Core ML optimization compatibility](https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html)
- [Quantization performance guidance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)

## Hardware evidence groups

| Evidence group | Representative iPhones | Purpose |
|---|---|---|
| A12-A13 | iPhone XS/XR, 11, SE (2nd generation) | Oldest supported Neural Engine boundary |
| A14-A16 | iPhone 12-14 families, iPhone 15, SE (3rd generation) | Modern Float16 baseline and pre-A17 INT8 boundary |
| A17 Pro-A19 | iPhone 15 Pro and later generations | Newer INT8-capable path and local iPhone 16 Pro evidence |
| future | Later A-series chips | Re-run the same capability-based protocol; never add a guessed model allowlist |

Simulator and Apple-silicon Mac results can prove compilation and functional correctness only. They are never labelled as iPhone Neural Engine performance evidence.

## What Verto records

The existing `PaddleOCRProbeTests.testBenchmarkExternalCorpus` remains the only end-to-end raw-data probe. It accepts test-only `all`, `cpuAndNeuralEngine`, `cpuAndGPU`, and `cpuOnly` policies while production keeps `all`.

Raw schema version 2 records:

- evidence class, declared hardware group, hardware identifier, OS, thermal state, and available Core ML devices;
- cold compile/load time and every unaggregated timed measurement;
- preprocessing, detector inference, DB post-processing, recognizer inference, CTC decoding, and end-to-end time separately;
- current and process-lifetime peak resident memory; the dedicated runner launches this test by itself so the lifetime peak has a defined process boundary;
- energy source, optional joules, and an external Instruments artifact path without inventing an in-process energy estimate;
- on iOS 17.4 and later, every `MLComputePlan` operation's preferred device, supported devices, and estimated cost weight.

Core ML does not expose a prose explanation for scheduler choices. Verto therefore reports only two factual fallback classifications: the operation does not support the Neural Engine, or Core ML preferred another supported device. Any significant fallback still needs a separately reviewed explanation in the model repository before it can pass.

See [`evaluation/ane-report-schema.md`](../evaluation/ane-report-schema.md) for the machine-readable contract.

## Assessment gates

`scripts/evaluate_ane.py` owns the decisions; Verto never grades itself. For each detector and recognizer under `all`, the assessor requires at least 90% of known estimated cost to prefer the Neural Engine, no cost-bearing operation with missing device usage or no Neural Engine support, and a written explanation for every non-ANE operation representing at least 5% of cost. Unweighted bookkeeping and constants remain visible in the raw counts but do not pretend to consume runtime cost.

A production change to `cpuAndNeuralEngine` remains ineligible until same-device pairs cover all three physical hardware groups. Each pair must preserve identical OCR output, improve warm end-to-end p50 or measured energy by at least 15%, and keep every measured latency, memory, and energy dimension within 10% of `all`.

If detector plus recognizer inference accounts for less than 40% of end-to-end time, the assessor redirects work to CPU preprocessing, DB post-processing, and CTC decoding. A hardware-specific package is outside v2 unless a later experiment shows at least 25% modern-device benefit and a stable public capability check exists.

W8A8 remains an experiment, not a default package. It must first pass the existing v2 quality gates, then improve A18 Pro end-to-end latency or energy by at least 15% without regressing A12/A16 by more than 10%. Weight-only INT8, palette compression, smaller detectors, and cross-tier components already rejected by the quality funnel do not re-enter merely under an NPU label.

The existing recipe builder now supports this as [`recipes/candidates/small-rec320-w8a8.json`](../recipes/candidates/small-rec320-w8a8.json). It refuses to build without a public schema-v1 calibration corpus, calibrates activation ranges before applying per-channel INT8 weights, verifies that activation and weight quantization operators are present, and rejects non-finite or over-tolerance output. A one-sample recognizer smoke proves the coremltools 9.0 compression and inference path; it is not a quality or performance result. A real candidate build uses:

```bash
uv run python scripts/build_models.py \
  --recipe recipes/candidates/small-rec320-w8a8.json \
  --tier small \
  --calibration-corpus /path/to/locked-public-tuning-corpus.json \
  --out /tmp/small-rec320-w8a8
```

## Running a local pair

Use one physical device, one candidate, the same public performance subset, three warm-ups, and 30 raw measurements. Run `all` and `cpuAndNeuralEngine` consecutively, changing only the compute-unit argument.

```bash
uv run python scripts/run_verto_benchmark.py \
  --verto /path/to/Verto \
  --model-dir /path/to/small \
  --tier small \
  --candidate small-rec320 \
  --candidate-kind shape \
  --parent-candidate B1 \
  --destination 'platform=iOS,id=<UDID>' \
  --derived-data /tmp/verto-ane-derived \
  --result-bundle /tmp/small-rec320-all.xcresult \
  --corpus /path/to/public-performance-corpus.json \
  --report /tmp/small-rec320-all.json \
  --warmup-runs 3 \
  --measurement-runs 30 \
  --compute-units all \
  --evidence-class localPhysical \
  --hardware-group A17Pro-A19 \
  --device-label 'iPhone 16 Pro'
```

Then assess paired raw reports:

```bash
uv run python scripts/evaluate_ane.py \
  /tmp/small-rec320-all.json \
  /tmp/small-rec320-cpu-and-ne.json \
  --output /tmp/small-rec320-ane-assessment.json
```

For model candidates, each raw run also names its parent candidate so the assessor compares one funnel step on the same device. W8A8 assessment additionally consumes a model-repository quality decision, for example:

```json
{
  "schemaVersion": 1,
  "candidates": {
    "small-rec320-w8a8": "pass"
  }
}
```

Pass that file with `--quality-decisions`. Without it, W8A8 remains ineligible even when timing looks favorable.

The private 36-photo holdout stays on the local iPhone 16 Pro and is run only for final candidates. Remote physical-device sessions may receive public models and redistributable public or generated images only.

## Remote-device boundary

The intended remote representatives are an iPhone XS on iOS 17.4 or later and an A16 iPhone. A BrowserStack XCUITest session must run B1 and its candidate back-to-back on the same allocated device. The [device selector](https://www.browserstack.com/docs/app-automate/xcuitest/specify-devices) is an inventory request, not proof that a specific device will always be available.

No cloud upload workflow is committed yet because v2 assets and the locked public corpus are still release-blocked, and no BrowserStack account or paid usage has been authorized. Adding credentials, uploading artifacts, or starting a paid session requires repository-owner approval. Until then, the versioned report must say that A12/A16 performance is unverified, and hardware-specific packaging remains forbidden.
