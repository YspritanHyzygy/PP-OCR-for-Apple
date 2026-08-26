#!/usr/bin/env python3
"""Merge raw Verto corpus reports without changing or scoring their samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def merge(documents: list[dict[str, Any]]) -> dict[str, Any]:
    if not documents:
        raise ValueError("at least one report is required")
    samples = []
    seen: set[str] = set()
    runs = []
    schema_versions: set[int] = set()
    for document in documents:
        schema_version = document.get("schemaVersion")
        if schema_version not in {1, 2} or not isinstance(document.get("samples"), list):
            raise ValueError("every report must use raw schemaVersion 1 or 2")
        schema_versions.add(schema_version)
        runs.append(document.get("run", {}))
        for sample in document["samples"]:
            sample_id = sample["id"]
            if sample_id in seen:
                raise ValueError(f"duplicate sample id: {sample_id}")
            seen.add(sample_id)
            samples.append(sample)
    if len(schema_versions) != 1:
        raise ValueError("cannot merge raw reports with different schema versions")
    return {"schemaVersion": schema_versions.pop(), "run": {"mergedRuns": runs}, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
        args.output.write_text(
            json.dumps(merge(documents), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"cannot merge reports: {error}")


if __name__ == "__main__":
    main()
