#!/usr/bin/env python3
"""Create one single-variable candidate recipe from a committed baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TIERS = ("tiny", "small", "medium")
COMPRESSIONS = ("none", "int8", "palette8", "palette6")
IO_TYPES = ("FLOAT32", "FLOAT16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", required=True, choices=TIERS)
    parser.add_argument("--detector-source-tier", choices=TIERS)
    parser.add_argument("--recognizer-source-tier", choices=TIERS)
    parser.add_argument("--detector-size", type=int, choices=(640, 736, 960))
    parser.add_argument("--recognizer-width", type=int, choices=(320, 640))
    parser.add_argument("--detector-compression", choices=COMPRESSIONS)
    parser.add_argument("--recognizer-compression", choices=COMPRESSIONS)
    parser.add_argument("--detector-io-type", choices=IO_TYPES)
    parser.add_argument("--recognizer-io-type", choices=IO_TYPES)
    args = parser.parse_args()

    recipe = json.loads(args.base.read_text(encoding="utf-8"))
    recipe["name"] = args.name
    config = recipe["tiers"][args.tier]
    updates = {
        "detectorSourceTier": args.detector_source_tier,
        "recognizerSourceTier": args.recognizer_source_tier,
        "detectorInputSize": args.detector_size,
        "recognizerInputWidth": args.recognizer_width,
        "detectorCompression": args.detector_compression,
        "recognizerCompression": args.recognizer_compression,
        "detectorIOType": args.detector_io_type,
        "recognizerIOType": args.recognizer_io_type,
    }
    selected = {key: value for key, value in updates.items() if value is not None}
    if len(selected) != 1:
        parser.error("a candidate recipe must change exactly one setting")
    config.update(selected)
    args.output.write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
