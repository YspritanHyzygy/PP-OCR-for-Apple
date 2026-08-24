# PP-OCRv6 Core ML packages v1

This first release migrates the exact model payloads previously used by Verto into their dedicated model repository. The archive bytes, sizes, SHA-256 digests, Core ML packages, and character sets are unchanged; only the public asset names and repository location changed.

The three tiers are converted from PaddlePaddle's official PP-OCRv6 ONNX exports for Core ML with fixed Apple Neural Engine-compatible input shapes and Float16 compute. See `manifest.json` for immutable source revisions, conversion parameters, package sizes, and checksums.

This is an unofficial redistribution and is not affiliated with or endorsed by PaddlePaddle or Apple.
