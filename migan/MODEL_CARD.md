# MI-GAN 512 Core ML candidate

## Model

This candidate converts the official MI-GAN 512 Places2 checkpoint into a fixed-shape Core ML ML Program. The repository and checkpoint are pinned in [`sources.lock.json`](sources.lock.json).

The model receives two tensors:

- `image`: Float16 `1 x 3 x 512 x 512`, RGB values from 0 to 1.
- `mask`: Float16 `1 x 1 x 512 x 512`, where 1 preserves a known pixel and 0 marks a hole.

The `output` tensor is Float32 `1 x 3 x 512 x 512` with values clamped to 0 through 1. Consumers must blend the generated pixels through their own feathered mask. Pixels outside that mask remain owned by the original image.

## Intended use

The candidate fills small masked regions in still photographs on Apple devices. Verto uses it only after a deterministic local reconstruction result is already available. Its scope is still-photo inpainting; text detection, OCR, live-video rendering, and general photo editing sit outside that scope.

## Conversion and parity

`scripts/build_migan.py` verifies the exact checkpoint byte length and SHA-256, checks out the pinned upstream source revision, wraps the generator with the two-input contract above, converts with Float16 compute for iOS 17, and reloads the resulting package for deterministic PyTorch/Core ML comparisons.

The generated manifest records every fixture hash plus maximum and mean absolute error. A release cannot proceed when either error exceeds the tolerance in `sources.lock.json`.

## License gate

The upstream source repository is MIT licensed and the pinned README links the official pretrained checkpoint folder. Both sources are preserved in [`licenses/PAIR-MIT-WEIGHTS-EVIDENCE.md`](licenses/PAIR-MIT-WEIGHTS-EVIDENCE.md). Redistribution rights for the separately hosted checkpoint remain unverified, so [`redistribution-license.json`](redistribution-license.json) remains `unconfirmed`.

Local conversion and evaluation may proceed. Uploading the converted package or publishing `migan-v1` remains blocked until primary evidence explicitly covers checkpoint redistribution, converted derivatives, and commercial use. `scripts/check_migan.py --require-release-ready` enforces this boundary.

## Runtime evidence

Core ML is configured with `computeUnits = ALL`. That setting allows the operating system to choose CPU, GPU, or Neural Engine. A Neural Engine claim requires a physical-device Core ML Instruments trace recorded in `benchmarks/migan-v1.json`.
