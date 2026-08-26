#!/usr/bin/env python3
"""Build the pinned MI-GAN 512 Places2 Core ML candidate and Apple Archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "migan/sources.lock.json").read_text(encoding="utf-8"))
FLOAT32 = 65568
FLOAT16 = 65552


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def tree_digest(root: Path) -> tuple[int, str]:
    value = hashlib.sha256()
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        value.update(len(relative).to_bytes(8, "big"))
        value.update(relative)
        size = path.stat().st_size
        total += size
        value.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                value.update(chunk)
    return total, value.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "PP-OCR-for-Apple build_migan.py"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def verify_checkpoint(path: Path) -> None:
    expected = LOCK["checkpoint"]
    if path.stat().st_size != expected["bytes"]:
        raise SystemExit(
            f"{path}: checkpoint size {path.stat().st_size} does not match {expected['bytes']}"
        )
    actual = sha256(path)
    if actual != expected["sha256"]:
        raise SystemExit(f"{path}: checkpoint SHA-256 {actual} does not match the source lock")


def prepare_upstream(path: Path | None, work: Path) -> Path:
    source = LOCK["upstreamCode"]
    if path is None:
        path = work / "MI-GAN"
        if not path.exists():
            subprocess.run(
                ["git", "clone", "--filter=blob:none", source["repository"], str(path)],
                check=True,
            )
    if not (path / ".git").exists():
        raise SystemExit(f"{path}: expected a MI-GAN git checkout")
    subprocess.run(["git", "fetch", "origin", source["revision"]], cwd=path, check=True)
    subprocess.run(["git", "checkout", "--detach", source["revision"]], cwd=path, check=True)
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()
    if actual != source["revision"]:
        raise SystemExit(f"{path}: upstream revision {actual} does not match the source lock")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"{path}: upstream checkout contains local changes")
    return path


def load_generator(upstream: Path, checkpoint: Path) -> nn.Module:
    source = upstream / "lib/model_zoo/migan_inference.py"
    spec = importlib.util.spec_from_file_location("locked_migan_inference", source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{source}: could not load the pinned MI-GAN generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = module.Generator(resolution=512).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    generator.load_state_dict(state, strict=True)
    return generator


class MIGANTwoInputWrapper(nn.Module):
    """Own the published image and mask convention at the model boundary."""

    def __init__(self, generator: nn.Module):
        super().__init__()
        self.generator = generator

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        known = mask.to(torch.float32).clamp(0, 1)
        normalized = image.to(torch.float32).clamp(0, 1) * 2 - 1
        model_input = torch.cat([known - 0.5, normalized * known], dim=1)
        generated = self.generator(model_input)
        return (generated * 0.5 + 0.5).clamp(0, 1).to(torch.float32)


def fixtures() -> list[tuple[str, np.ndarray, np.ndarray]]:
    axis = np.linspace(0, 1, 512, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    gradient = np.stack([x, y, (x + y) * 0.5], axis=0)[None, ...]
    center = np.ones((1, 1, 512, 512), dtype=np.float32)
    center[:, :, 176:336, 144:368] = 0

    checker = ((np.indices((512, 512)).sum(axis=0) // 24) % 2).astype(np.float32)
    texture = np.stack([checker, 1 - checker, checker * 0.5 + 0.25], axis=0)[None, ...]
    stripe = np.ones((1, 1, 512, 512), dtype=np.float32)
    stripe[:, :, 220:292, 64:448] = 0
    return [("center-gradient", gradient, center), ("horizontal-texture", texture, stripe)]


def fixture_hash(image: np.ndarray, mask: np.ndarray) -> str:
    value = hashlib.sha256()
    value.update(image.astype(np.float16).tobytes(order="C"))
    value.update(mask.astype(np.float16).tobytes(order="C"))
    return value.hexdigest()


def validate_spec(model: ct.models.MLModel) -> None:
    spec = model.get_spec()
    if spec.WhichOneof("Type") != "mlProgram":
        raise SystemExit("MI-GAN conversion did not produce an ML Program")
    inputs = {item.name: item for item in spec.description.input}
    output = {item.name: item for item in spec.description.output}
    for name, shape in {"image": [1, 3, 512, 512], "mask": [1, 1, 512, 512]}.items():
        if name not in inputs:
            raise SystemExit(f"MI-GAN Core ML input {name} is missing")
        feature = inputs[name].type.multiArrayType
        if list(feature.shape) != shape or feature.dataType != FLOAT16:
            raise SystemExit(f"MI-GAN Core ML input {name} has an unexpected shape or type")
    if "output" not in output:
        raise SystemExit("MI-GAN Core ML output is missing")
    feature = output["output"].type.multiArrayType
    if list(feature.shape) != [1, 3, 512, 512] or feature.dataType != FLOAT32:
        raise SystemExit("MI-GAN Core ML output has an unexpected shape or type")


def convert(wrapper: nn.Module, output: Path) -> ct.models.MLModel:
    torch.manual_seed(17)
    image = torch.rand(1, 3, 512, 512, dtype=torch.float32)
    mask = torch.ones(1, 1, 512, 512, dtype=torch.float32)
    mask[:, :, 176:336, 144:368] = 0
    traced = torch.jit.trace(wrapper, (image, mask), strict=False)
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="image", shape=image.shape, dtype=np.float16),
            ct.TensorType(name="mask", shape=mask.shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    model.author = "Picsart AI Research; Core ML conversion by Yspritan"
    model.short_description = "MI-GAN 512 Places2 image inpainting candidate"
    model.input_description["image"] = "RGB Float16 values from 0 to 1"
    model.input_description["mask"] = "1 preserves known pixels; 0 marks the inpainting hole"
    model.output_description["output"] = "Generated RGB Float32 values from 0 to 1"
    model.save(str(output))
    reloaded = ct.models.MLModel(str(output), compute_units=ct.ComputeUnit.ALL)
    validate_spec(reloaded)
    return reloaded


def parity(wrapper: nn.Module, model: ct.models.MLModel) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    tolerances = LOCK["conversion"]["parityTolerance"]
    with torch.no_grad():
        for name, image, mask in fixtures():
            reference = wrapper(torch.from_numpy(image), torch.from_numpy(mask)).numpy()
            prediction = model.predict(
                {"image": image.astype(np.float16), "mask": mask.astype(np.float16)}
            )["output"].astype(np.float32)
            delta = np.abs(reference - prediction)
            maximum = float(delta.max())
            mean = float(delta.mean())
            print(f"{name}: max error {maximum:.8f}, mean error {mean:.8f}")
            if maximum > tolerances["maximumAbsoluteError"]:
                raise SystemExit(f"{name}: maximum parity error {maximum} exceeds the tolerance")
            if mean > tolerances["meanAbsoluteError"]:
                raise SystemExit(f"{name}: mean parity error {mean} exceeds the tolerance")
            results.append(
                {
                    "name": name,
                    "inputSHA256": fixture_hash(image, mask),
                    "maximumAbsoluteError": maximum,
                    "meanAbsoluteError": mean,
                }
            )
    return results


def provenance_record(
    upstream: Path, model_bytes: int, model_sha: str
) -> dict[str, Any]:
    evidence = ROOT / "migan/licenses/PAIR-MIT-WEIGHTS-EVIDENCE.md"
    return {
        "schemaVersion": 1,
        "upstreamCodeRevision": LOCK["upstreamCode"]["revision"],
        "checkpointSHA256": LOCK["checkpoint"]["sha256"],
        "modelPackage": {"bytes": model_bytes, "treeSHA256": model_sha},
        "evidence": {
            "upstreamLicenseSHA256": sha256(upstream / "LICENSE"),
            "redistributionEvidenceSHA256": sha256(evidence),
        },
    }


def build_archive(
    model: Path,
    output_root: Path,
    work: Path,
    upstream: Path,
    model_bytes: int,
    model_sha: str,
) -> Path:
    payload = work / "archive-payload"
    shutil.rmtree(payload, ignore_errors=True)
    payload.mkdir(parents=True)
    shutil.copytree(model, payload / model.name)
    shutil.copy2(ROOT / "migan/MODEL_CARD.md", payload / "MODEL_CARD.md")
    shutil.copy2(ROOT / "migan/NOTICE", payload / "NOTICE")
    shutil.copy2(
        ROOT / "migan/redistribution-license.json", payload / "redistribution-license.json"
    )
    shutil.copy2(
        ROOT / "migan/licenses/PAIR-MIT-WEIGHTS-EVIDENCE.md",
        payload / "PAIR-MIT-WEIGHTS-EVIDENCE.md",
    )
    upstream_license = upstream / "LICENSE"
    shutil.copy2(upstream_license, payload / "MI-GAN-MIT.txt")
    provenance = provenance_record(upstream, model_bytes, model_sha)
    (payload / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    archive = output_root / LOCK["conversion"]["archiveName"]
    archive.unlink(missing_ok=True)
    subprocess.run(
        ["aa", "archive", "-d", str(payload), "-o", str(archive), "-a", "lzfse", "-b", "1m"],
        check=True,
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("build/migan-v1"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--upstream", type=Path)
    args = parser.parse_args()

    output_root = args.out.resolve()
    work = args.work.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    checkpoint = args.weights.resolve() if args.weights else work / LOCK["checkpoint"]["fileName"]
    if not checkpoint.exists():
        print(f"download {LOCK['checkpoint']['fileName']}")
        download(LOCK["checkpoint"]["downloadURL"], checkpoint)
    verify_checkpoint(checkpoint)
    upstream = prepare_upstream(args.upstream.resolve() if args.upstream else None, work)
    wrapper = MIGANTwoInputWrapper(load_generator(upstream, checkpoint)).eval()

    package = output_root / LOCK["conversion"]["modelName"]
    shutil.rmtree(package, ignore_errors=True)
    print("convert MI-GAN 512 to Core ML")
    model = convert(wrapper, package)
    parity_results = parity(wrapper, model)
    model_bytes, model_sha = tree_digest(package)
    archive = build_archive(package, output_root, work, upstream, model_bytes, model_sha)

    manifest = {
        "schemaVersion": 1,
        "packVersion": "1",
        "releaseTag": LOCK["conversion"]["releaseTag"],
        "source": {
            "upstreamCode": LOCK["upstreamCode"],
            "checkpoint": LOCK["checkpoint"],
        },
        "conversion": {
            key: LOCK["conversion"][key]
            for key in (
                "modelName",
                "minimumDeploymentTarget",
                "modelType",
                "computePrecision",
                "computeUnits",
                "inputs",
                "output",
            )
        },
        "modelPackage": {"bytes": model_bytes, "treeSHA256": model_sha},
        "parity": {"fixtures": parity_results},
        "provenance": provenance_record(upstream, model_bytes, model_sha),
        "licenseGate": json.loads(
            (ROOT / "migan/redistribution-license.json").read_text(encoding="utf-8")
        ),
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        },
    }
    manifest_path = output_root / "migan-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_root / "MIGAN-SHA256SUMS").write_text(
        f"{manifest['archive']['sha256']}  {archive.name}\n", encoding="utf-8"
    )
    print(
        f"{archive.name}: {archive.stat().st_size / 1048576:.1f} MB, "
        f"sha256 {manifest['archive']['sha256']}"
    )


if __name__ == "__main__":
    main()
