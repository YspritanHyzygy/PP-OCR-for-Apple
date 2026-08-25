# PP-OCR-for-Apple v1 optimization analysis

This document records the observed v1 model structure, interface mismatches, and evidence-backed optimization opportunities. It is an analysis of the immutable `v1` release, not a roadmap or a release claim.

## Scope and evidence

The inspected artifacts are the three archives published in the [`v1` release](https://github.com/YspritanHyzygy/PP-OCR-for-Apple/releases/tag/v1). File sizes and hashes come from [`manifests/v1.json`](../manifests/v1.json). Model structure comes from the six released Core ML packages. Upstream scores come from PaddlePaddle's [PP-OCRv6 documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md); PaddlePaddle describes those scores as results on internal multi-scenario benchmarks, not measurements made by this project.

## Released tiers

| Tier | Detector Hmean | Recognizer W-Avg | Archive bytes | Detector weights | Recognizer weights | Characters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 80.6 | 73.5 | 2,915,454 | 867,648 | 2,212,212 | 6,904 |
| small | 84.1 | 81.3 | 12,999,796 | 4,917,760 | 10,543,020 | 18,708 |
| medium | 86.2 | 83.2 | 47,521,629 | 30,990,080 | 38,236,652 | 18,708 |

All six models are ML Programs with Float16 weight storage, fixed tensor shapes, Float32 inputs and outputs, and two boundary cast operations. Each detector accepts `1 x 3 x 960 x 960` and returns `1 x 1 x 960 x 960`. The tiny recognizer returns `1 x 80 x 6906`; small and medium return `1 x 80 x 18710`.

The recognizer's final classifier is a large share of its payload: about 1.10 MB for tiny, 4.49 MB for small, and 7.18 MB for medium. Across complete uncompressed packs, weights account for 93.0% of tiny, 97.8% of small, and 99.4% of medium. No duplicate model payload or unused file was found inside a tier.

## Runtime evidence

The versioned v1 report measures deterministic random tensors on an M5 Pro Mac with Core ML compute units set to `ALL`:

| Tier | Detector median | Recognizer median |
| --- | ---: | ---: |
| tiny | 73.31 ms | 6.99 ms |
| small | 141.18 ms | 20.61 ms |
| medium | 395.38 ms | 40.94 ms |

These figures isolate model prediction on one Mac. They do not include camera preprocessing, DB post-processing, per-line cropping, CTC decoding, model download, or iPhone runtime. They therefore cannot establish end-to-end OCR latency or select an application default.

## Verified interface mismatches

### Color order

The six locked PaddlePaddle `inference.yml` files declare `DecodeImage.img_mode: BGR`. Their ONNX graphs consume the input tensor directly in the first convolution and contain no hidden channel reversal. Verto v1 creates RGBA/RGB buffers and writes the red, green, and blue values into planes 0, 1, and 2. Its existing real-model probe renders dark monochrome text, so swapping red and blue does not expose the mismatch.

The Core ML weights are not corrupt; the mismatch is between the upstream preprocessing contract and the consuming application's tensor construction. A corrected Apple-facing model can accept RGB without adding a runtime shuffle by reversing the first convolution's input-channel weights and reversing the detector normalization vectors.

### Detector threshold

The locked tiny detector config declares `box_thresh: 0.4`. The small and medium configs declare `0.45`. Verto v1 applies one global `0.45` value and its test describes that value as shared by all models. The result is a stricter-than-upstream acceptance threshold for faint tiny-tier detections.

### Conversion environment

The v1 lock selects `torch==2.13.0`. Importing coremltools 9.0 reports that PyTorch 2.13.0 has not been tested and that 2.7.0 is the newest tested version. A successful v1 build proves that the combination ran once; it does not make the unsupported pairing a stable conversion contract.

## Optimization evidence

- **Component selection:** detectors and recognizers have compatible external contracts, so detector and recognizer tiers can be measured independently. Character-set and language coverage remain properties of the chosen recognizer.
- **Fixed shapes:** the upstream detector's preferred dynamic shape is 736 square, while v1 fixes 960 square. Recognizer configs use width 320, while v1 fixes 640. Smaller fixed candidates can reduce work, but small, long, rotated, and curved text must decide whether they remain usable.
- **Weight compression:** raw Float16 weights dominate every pack. iOS-17-compatible per-channel INT8 weights and per-tensor palettization can reduce storage; runtime benefit remains hardware- and graph-dependent.
- **Boundary precision:** Float16 inputs and outputs would halve tensor-boundary bytes and remove boundary casts. The consumer currently normalizes and decodes Float32 values, so this is a coordinated model-and-application experiment rather than a file-only change.
- **Classifier output:** small and medium expose about 5.99 MB of recognizer probabilities per line. Verto already decodes the normal Float32 path without copying that tensor, so output reduction matters only if profiling shows decoding or transfer is material.

## Deferred or unsupported conclusions

- W8A8 activation quantization is not a general iOS 17 win. Apple documents its optimized Neural Engine path on A17 Pro and M4-class hardware, while CPU and GPU paths may regress.
- Post-training pruning changes learned weights and lacks a quality case without a representative corpus or fine-tuning data.
- Moving CTC top-1 selection into the model changes output and confidence semantics and has no measured benefit yet.
- Concurrent per-line recognition, model prewarming, connected-component limits, and cancellation are application-pipeline questions, not properties of the three model artifacts.
- LZMA can reduce download bytes, but it changes installation CPU and memory rather than model inference.
- The v1 report contains no iPhone timing, no public end-to-end accuracy corpus, and no private captured-photo holdout. Claims about a faster or more accurate successor remain unverified until those gaps are filled.
