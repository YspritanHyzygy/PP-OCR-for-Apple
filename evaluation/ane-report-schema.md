# Raw ANE benchmark schema v2

Verto emits observations; this repository interprets them. A raw report has `schemaVersion: 2`, one `run`, two `computePlans`, and one or more `samples`.

## Run identity

Required fields identify the tier, candidate, candidate kind, optional parent candidate, requested compute units, evidence class, OS, available Core ML devices, warm-up count, and raw measurement count. Physical evidence also requires a declared `hardwareGroup`; simulator evidence must not declare one. `hardwareIdentifier` records the system machine identifier because `UIDevice.model` alone commonly reports only `iPhone`.

Valid compute-unit values are `all`, `cpuAndNeuralEngine`, `cpuAndGPU`, and `cpuOnly`. Valid evidence classes are `localPhysical`, `remotePhysical`, and `simulator`.

Energy is an evidence object. `source: notMeasured` is valid and makes no energy claim. Joules or an artifact path may be supplied only when an external measurement such as an Instruments trace exists.

## Compute plan

iOS 17.4 or later emits one detector and one recognizer plan with status `available`, `error`, or `unavailableBeforeiOS17.4`. Each operation records:

- stable traversal path and operator name;
- estimated cost weight when Core ML supplies one;
- preferred compute device;
- all supported compute devices;
- a factual fallback classification when the preferred device is not the Neural Engine.

The public API does not expose a prose scheduler reason. Significant fallbacks are explained outside the raw report and passed to `scripts/evaluate_ane.py --explanations` as:

```json
{
  "schemaVersion": 1,
  "operations": {
    "detector:model/function:main/12:reshape": "Shape bookkeeping remains on CPU."
  }
}
```

## Samples and measurements

Each sample preserves ground truth, raw detector polygons, raw recognized lines, and an array of measurements. Every measurement includes raw stage timings, thermal state before and after, resident memory before and after, and process-lifetime peak resident memory. No percentile, ANE score, pass flag, CER, or matching decision is written by Verto.

The quality evaluator accepts schema v1 and v2. For v2 it calculates timing percentiles from every entry in `measurements`; prediction scoring still uses one raw prediction set per sample.

`scripts/evaluate_ane.py` pairs compute-unit policies only when candidate, tier, device, hardware group, and evidence class match. It pairs model candidates only when the candidate names a `parentCandidate` and every device field plus compute units match. W8A8 cannot pass without a separately supplied `pass` from the versioned quality evaluator.
