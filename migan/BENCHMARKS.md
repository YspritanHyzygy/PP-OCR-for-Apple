# MI-GAN benchmark contract

The release benchmark lives in [`../benchmarks/migan-v1.json`](../benchmarks/migan-v1.json). Cross-device correctness and latency evidence lives in [`../benchmarks/migan-device-matrix-v1.json`](../benchmarks/migan-device-matrix-v1.json).

A complete report records:

- release tag and model SHA-256;
- physical device, chip, memory, OS, and thermal state;
- one warm-up plus three measured 512 predictions;
- cold load time, each hot prediction time, median time, and sampled resident memory;
- `computeUnits = ALL`;
- a Core ML Instruments trace hash and the observed execution device.

Simulator and Mac measurements can validate conversion and output shape. Physical-device traces carry their own attribution limitations. The release record distinguishes a model-named Neural Engine observation from a trace that binds the exact model SHA and compute-unit configuration.

The device matrix also records full-output comparisons against the same-device CPU-and-GPU path. Its two raw JSON inputs are checked in under `../benchmarks/evidence/` and verified by SHA-256. Application crop preparation, tensor fill, output conversion, blending, layout, and UI publication remain separate Verto evidence.
