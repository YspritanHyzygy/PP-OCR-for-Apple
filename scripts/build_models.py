#!/usr/bin/env python3
#
#  Copyright 2026 Yspritan
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Build versioned PP-OCRv6 Core ML packages from one explicit recipe.

The builder owns source selection, the Apple-facing RGB contract, input shapes,
weight compression, numerical checks, archive creation, and release metadata.
Candidate and release builds use this same implementation; a recipe changes
inputs, not code paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import coremltools as ct
import coremltools.optimize as cto
import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from PIL import Image
from onnx import helper, numpy_helper
from onnx2torch import convert as onnx2torch_convert


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
DEFAULT_RECIPE = ROOT / "recipes/v2-corrected-reference.json"
TIERS = ("tiny", "small", "medium")
COMPONENTS = ("detector", "recognizer")

# PaddlePaddle's internal multi-scenario benchmark. These values provide
# provenance and tier context; project reports never relabel them as local or
# end-to-end accuracy.
UPSTREAM_METRICS = {
    "tiny": {"detectorHmean": 80.6, "recognizerWAvg": 73.5},
    "small": {"detectorHmean": 84.1, "recognizerWAvg": 81.3},
    "medium": {"detectorHmean": 86.2, "recognizerWAvg": 83.2},
}

DET_TOLERANCE = 0.011
REC_TOLERANCE = 0.30
REC_MEAN = [0.5, 0.5, 0.5]
REC_STD = [0.5, 0.5, 0.5]
FLOAT32 = 65568
FLOAT16 = 65552
VALID_COMPRESSIONS = {"none", "int8", "palette8", "palette6", "w8a8"}
VALID_IO_TYPES = {"FLOAT32", "FLOAT16"}
RECIPE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if recipe.get("schemaVersion") != 1:
        raise SystemExit(f"{path}: unsupported recipe schema")
    if not RECIPE_NAME.fullmatch(str(recipe.get("name", ""))):
        raise SystemExit(f"{path}: name must use lowercase letters, digits, and hyphens")
    if not str(recipe.get("packVersion", "")).isdigit():
        raise SystemExit(f"{path}: packVersion must contain digits only")
    if recipe.get("minimumDeploymentTarget") != "iOS17":
        raise SystemExit(f"{path}: v2 recipes must preserve the iOS17 floor")
    if recipe.get("computeUnits") != "ALL":
        raise SystemExit(f"{path}: computeUnits must be ALL")
    if recipe.get("colorOrder") != "RGB":
        raise SystemExit(f"{path}: Apple-facing v2 recipes must use RGB")
    if tuple(recipe.get("tiers", {})) != TIERS:
        raise SystemExit(f"{path}: tiers must be ordered as {TIERS}")

    for output_tier, config in recipe["tiers"].items():
        for key in ("detectorSourceTier", "recognizerSourceTier"):
            if config.get(key) not in TIERS:
                raise SystemExit(f"{path}: {output_tier}.{key} is invalid")
        if config.get("detectorInputSize") not in {640, 736, 960}:
            raise SystemExit(f"{path}: {output_tier}.detectorInputSize is not a measured candidate")
        if config.get("recognizerInputHeight") != 48:
            raise SystemExit(f"{path}: recognizer height must remain 48")
        if config.get("recognizerInputWidth") not in {320, 640}:
            raise SystemExit(f"{path}: {output_tier}.recognizerInputWidth is not a measured candidate")
        for component in COMPONENTS:
            compression = config.get(f"{component}Compression")
            io_type = config.get(f"{component}IOType")
            if compression not in VALID_COMPRESSIONS:
                raise SystemExit(f"{path}: {output_tier}.{component}Compression is invalid")
            if io_type not in VALID_IO_TYPES:
                raise SystemExit(f"{path}: {output_tier}.{component}IOType is invalid")
            if compression == "w8a8" and io_type != "FLOAT32":
                raise SystemExit(f"{path}: W8A8 calibration currently requires Float32 I/O")
    return recipe


def fetch(repo: str, revision: str, destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    print(f"  download {repo}@{revision}")
    snapshot_download(repo, revision=revision, local_dir=str(destination))
    return destination


def source_for(tier: str, component: str) -> dict[str, str]:
    source = SOURCE_LOCK["tiers"][tier][component]
    return {"repository": source["repository"], "revision": source["revision"]}


def transform(config: dict[str, Any], name: str) -> dict[str, Any]:
    for item in config["PreProcess"]["transform_ops"]:
        if name in item:
            return item[name] or {}
    raise SystemExit(f"inference.yml does not contain {name}")


def read_source_contract(directory: Path, component: str) -> dict[str, Any]:
    config = yaml.safe_load((directory / "inference.yml").read_text(encoding="utf-8"))
    decoded = transform(config, "DecodeImage")
    if decoded.get("img_mode") != "BGR":
        raise SystemExit(f"{directory}: expected the locked upstream model to declare BGR input")

    if component == "detector":
        normalized = transform(config, "NormalizeImage")
        post = config["PostProcess"]
        return {
            "upstreamColorOrder": "BGR",
            "upstreamMean": [float(value) for value in normalized["mean"]],
            "upstreamStd": [float(value) for value in normalized["std"]],
            "postProcess": {
                "binarizationThreshold": float(post["thresh"]),
                "boxScoreThreshold": float(post["box_thresh"]),
                "unclipRatio": float(post["unclip_ratio"]),
                "maximumCandidates": int(post["max_candidates"]),
            },
        }

    resized = transform(config, "RecResizeImg")
    return {
        "upstreamColorOrder": "BGR",
        "upstreamImageShape": [int(value) for value in resized["image_shape"]],
        "characterDict": [str(value) for value in config["PostProcess"]["character_dict"]],
    }


def initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    for tensor in model.graph.initializer:
        if tensor.name == name:
            return tensor
    raise SystemExit(f"ONNX initializer {name!r} was not found")


def fold_bgr_input_to_rgb(source: Path, destination: Path) -> None:
    """Reverse the first convolution's input-channel weights.

    Upstream receives normalized B, G, R planes. The resulting graph receives
    the mathematically equivalent R, G, B tensor and contains no runtime
    channel-shuffle operation.
    """

    model = onnx.load(str(source))
    input_name = model.graph.input[0].name
    consumers = [node for node in model.graph.node if input_name in node.input]
    if len(consumers) != 1 or consumers[0].op_type != "Conv":
        details = [(node.name, node.op_type) for node in consumers]
        raise SystemExit(f"{source}: expected one first Conv consuming {input_name}, found {details}")
    node = consumers[0]
    attributes = {attribute.name: attribute for attribute in node.attribute}
    groups = int(attributes["group"].i) if "group" in attributes else 1
    if groups != 1:
        raise SystemExit(f"{source}: first convolution uses group={groups}; channel folding is undefined")

    weight = initializer(model, node.input[1])
    values = numpy_helper.to_array(weight)
    if values.ndim < 2 or values.shape[1] != 3:
        raise SystemExit(f"{source}: first convolution weight shape {values.shape} has no RGB axis")
    folded = np.ascontiguousarray(values[:, [2, 1, 0], ...])
    weight.CopyFrom(numpy_helper.from_array(folded, name=weight.name))
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))


def run_onnx(path: Path, x: np.ndarray) -> np.ndarray:
    return ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    ).run(None, {"x": x})[0]


def assert_rgb_fold_is_equivalent(original: Path, folded: Path, shape: tuple[int, ...]) -> float:
    bgr = np.random.default_rng(17).normal(0, 1, shape).astype(np.float32)
    rgb = np.ascontiguousarray(bgr[:, [2, 1, 0], ...])
    reference = run_onnx(original, bgr)
    candidate = run_onnx(folded, rgb)
    delta = float(np.abs(reference - candidate).max())
    if not np.allclose(reference, candidate, rtol=1e-5, atol=1e-5):
        raise SystemExit(f"RGB first-convolution fold changed ONNX output (max error {delta})")
    print(f"     RGB fold equivalence max error {delta:.8f}")
    return delta


def rewrite_auto_pad(source: Path, destination: Path) -> int:
    """Replace the supported SAME_UPPER nodes with explicit padding."""

    model = onnx.load(str(source))
    changed = 0
    for node in model.graph.node:
        attributes = {attribute.name: attribute for attribute in node.attribute}
        auto_pad = attributes.get("auto_pad")
        if auto_pad is None or auto_pad.s.decode() != "SAME_UPPER":
            continue
        kernel = list(attributes["kernel_shape"].ints)
        strides = list(attributes["strides"].ints) if "strides" in attributes else [1] * len(kernel)
        dilations = list(attributes["dilations"].ints) if "dilations" in attributes else [1] * len(kernel)
        if any(value != 1 for value in strides) or any(value != 1 for value in dilations):
            raise SystemExit(
                f"{node.op_type}: stride={strides} dilation={dilations} cannot use a fixed SAME_UPPER rewrite"
            )
        begin = [(value - 1) // 2 for value in kernel]
        end = [(value - 1) - start for value, start in zip(kernel, begin)]
        node.attribute.remove(auto_pad)
        if "pads" in attributes:
            node.attribute.remove(attributes["pads"])
        node.attribute.append(helper.make_attribute("pads", begin + end))
        changed += 1
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    return changed


def assert_rewrite_is_exact(original: Path, rewritten: Path, shape: tuple[int, ...]) -> None:
    x = np.random.default_rng(0).random(shape).astype(np.float32)
    delta = float(np.abs(run_onnx(original, x) - run_onnx(rewritten, x)).max())
    if delta != 0:
        raise SystemExit(f"auto_pad rewrite changed output (max error {delta})")
    print("     auto_pad rewrite is bit-exact")


def apply_compression(
    model: ct.models.MLModel,
    mode: str,
    calibration_samples: list[dict[str, np.ndarray]] | None,
) -> ct.models.MLModel:
    if mode == "none":
        return model
    if mode == "int8":
        op_config = cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", granularity="per_channel", weight_threshold=2048
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        return cto.coreml.linear_quantize_weights(model, config=config)
    if mode == "w8a8":
        if not calibration_samples:
            raise SystemExit("W8A8 requires real public calibration samples")
        op_config = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric")
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        activated = cto.coreml.linear_quantize_activations(
            model,
            config=config,
            sample_data=calibration_samples,
            # PP-OCR programs contain hundreds of eligible operations. Smaller
            # calibration groups keep the temporary output-expanded model bounded.
            calibration_op_group_size=50,
        )
        return cto.coreml.linear_quantize_weights(activated, config=config)
    if mode in {"palette8", "palette6"}:
        bits = int(mode.removeprefix("palette"))
        op_config = cto.coreml.OpPalettizerConfig(
            mode="kmeans", nbits=bits, granularity="per_tensor", weight_threshold=2048
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        return cto.coreml.palettize_weights(model, config=config)
    raise SystemExit(f"unsupported compression mode {mode}")


def numpy_dtype(io_type: str) -> type[np.floating[Any]]:
    return np.float32 if io_type == "FLOAT32" else np.float16


def protobuf_dtype(io_type: str) -> int:
    return FLOAT32 if io_type == "FLOAT32" else FLOAT16


def operation_types(model: ct.models.MLModel) -> set[str]:
    types: set[str] = set()

    def visit(block: Any) -> None:
        for operation in block.operations:
            types.add(str(operation.op_type))
            for child in operation.blocks:
                visit(child)

    for function in model._mil_program.functions.values():
        visit(function)
    return types


def assert_compression_ops(model: ct.models.MLModel, mode: str, label: str) -> None:
    if mode == "none":
        return
    types = operation_types(model)
    weight_quantized = any(
        name in types for name in ("constexpr_affine_dequantize", "constexpr_blockwise_shift_scale")
    )
    if mode in {"int8", "w8a8"} and not weight_quantized:
        raise SystemExit(f"{label}: {mode} model has no quantized-weight operation")
    if mode in {"palette8", "palette6"} and "constexpr_lut_to_dense" not in types:
        raise SystemExit(f"{label}: palette model has no lookup-table operation")
    if mode == "w8a8" and not {"quantize", "dequantize"} <= types:
        raise SystemExit(f"{label}: W8A8 model has no activation quantize/dequantize pair")


def to_coreml(
    onnx_path: Path,
    output: Path,
    shape: tuple[int, ...],
    tolerance: float,
    label: str,
    compression: str,
    io_type: str,
    calibration_samples: list[dict[str, np.ndarray]] | None = None,
    require_argmax_equal: bool = False,
) -> tuple[float, int]:
    torch_model = onnx2torch_convert(onnx.load(str(onnx_path)))
    torch_model.eval()
    torch.manual_seed(0)
    traced = torch.jit.trace(torch_model, torch.rand(*shape), strict=False)
    boundary_dtype = numpy_dtype(io_type)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=shape, dtype=boundary_dtype)],
        outputs=[ct.TensorType(dtype=boundary_dtype)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    model = apply_compression(model, compression, calibration_samples)
    assert_compression_ops(model, compression, label)
    model.save(str(output))

    expected_type = protobuf_dtype(io_type)
    for feature in list(model.get_spec().description.input) + list(model.get_spec().description.output):
        actual = feature.type.multiArrayType.dataType
        if actual != expected_type:
            raise SystemExit(f"{label}: feature {feature.name} has data type {actual}, expected {expected_type}")

    boundary_input = np.random.default_rng(1).random(shape).astype(boundary_dtype)
    reference = run_onnx(onnx_path, boundary_input.astype(np.float32))
    candidate = list(ct.models.MLModel(str(output)).predict({"x": boundary_input}).values())[0]
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise SystemExit(f"{label}: ONNX/Core ML output contains a non-finite value")
    if reference.shape != candidate.shape:
        raise SystemExit(f"{label}: ONNX {reference.shape} and Core ML {candidate.shape} differ")
    delta = float(np.abs(reference - candidate.astype(np.float32)).max())
    print(f"     {label} Core ML max error {delta:.5f} (limit {tolerance})")
    if delta > tolerance:
        raise SystemExit(f"{label}: Core ML error {delta} exceeds {tolerance}")

    mismatches = 0
    if require_argmax_equal:
        mismatches = int(
            np.count_nonzero(np.argmax(reference, axis=-1) != np.argmax(candidate, axis=-1))
        )
        print(f"     {label} CTC argmax mismatches {mismatches}")
    return delta, mismatches


def load_calibration_corpus(path: Path | None) -> tuple[Path, dict[str, Any]] | None:
    if path is None:
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != 1 or not isinstance(document.get("samples"), list):
        raise SystemExit(f"{path}: calibration corpus must use schemaVersion 1")
    if not document["samples"]:
        raise SystemExit(f"{path}: calibration corpus is empty")
    return path.resolve(), document


def calibration_image(path: str, corpus_path: Path) -> Image.Image:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = corpus_path.parent / resolved
    with Image.open(resolved) as image:
        return image.convert("RGB")


def calibration_samples(
    corpus: tuple[Path, dict[str, Any]] | None,
    component: str,
    shape: tuple[int, ...],
    contract: dict[str, Any],
    maximum: int = 32,
) -> list[dict[str, np.ndarray]] | None:
    if corpus is None:
        return None
    corpus_path, document = corpus
    output: list[dict[str, np.ndarray]] = []
    height, width = shape[-2:]
    for sample in document["samples"]:
        image = calibration_image(sample["imagePath"], corpus_path)
        if component == "detector":
            ratio = min(width / image.width, height / image.height)
            resized = image.resize(
                (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
                Image.Resampling.LANCZOS,
            )
            canvas = Image.new("RGB", (width, height))
            canvas.paste(resized, (0, 0))
            values = np.asarray(canvas, dtype=np.float32) / 255
            mean = np.asarray(list(reversed(contract["upstreamMean"])), dtype=np.float32)
            std = np.asarray(list(reversed(contract["upstreamStd"])), dtype=np.float32)
            tensor = ((values - mean) / std).transpose(2, 0, 1)[None]
            output.append({"x": np.ascontiguousarray(tensor)})
        else:
            for line in sample.get("groundTruth", []):
                if line.get("ignore", False):
                    continue
                points = line.get("polygon", [])
                if len(points) < 3:
                    continue
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                left, top = max(0, int(min(xs))), max(0, int(min(ys)))
                right = min(image.width, max(left + 1, int(np.ceil(max(xs)))))
                bottom = min(image.height, max(top + 1, int(np.ceil(max(ys)))))
                crop = image.crop((left, top, right, bottom))
                used = max(16, min(width, round(height * crop.width / crop.height)))
                resized = crop.resize((used, height), Image.Resampling.BILINEAR)
                tensor = np.zeros(shape, dtype=np.float32)
                values = (np.asarray(resized, dtype=np.float32) / 255 - 0.5) / 0.5
                tensor[0, :, :, :used] = values.transpose(2, 0, 1)
                output.append({"x": tensor})
                if len(output) >= maximum:
                    return output
        if len(output) >= maximum:
            return output
    return output


def build_component(
    output_tier: str,
    component: str,
    source_tier: str,
    source_dir: Path,
    work_dir: Path,
    output_dir: Path,
    shape: tuple[int, ...],
    compression: str,
    io_type: str,
    calibration_corpus: tuple[Path, dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_source_contract(source_dir, component)
    component_work = work_dir / output_tier / component
    component_work.mkdir(parents=True, exist_ok=True)
    rgb_onnx = component_work / "inference_rgb.onnx"
    fixed_onnx = component_work / "inference_rgb_fixed.onnx"

    source_onnx = source_dir / "inference.onnx"
    fold_bgr_input_to_rgb(source_onnx, rgb_onnx)
    rgb_error = assert_rgb_fold_is_equivalent(source_onnx, rgb_onnx, shape)
    changed = rewrite_auto_pad(rgb_onnx, fixed_onnx)
    print(f"     {component}: rewrote {changed} SAME_UPPER nodes")
    assert_rewrite_is_exact(rgb_onnx, fixed_onnx, shape)

    filename = "VertoTextDetector.mlpackage" if component == "detector" else "VertoTextRecognizer.mlpackage"
    tolerance = DET_TOLERANCE if component == "detector" else REC_TOLERANCE
    samples = calibration_samples(
        calibration_corpus, component, shape, contract
    ) if compression == "w8a8" else None
    conversion_error, argmax_mismatches = to_coreml(
        fixed_onnx,
        output_dir / filename,
        shape,
        tolerance,
        f"{output_tier}.{component}",
        compression,
        io_type,
        calibration_samples=samples,
        require_argmax_equal=component == "recognizer",
    )
    source = source_for(source_tier, component)
    metadata = {
        "sourceTier": source_tier,
        "source": source,
        "inputShape": list(shape),
        "inputOutputType": io_type,
        "computePrecision": "FLOAT16",
        "weightCompression": "int8" if compression == "w8a8" else compression,
        "activationQuantization": "int8" if compression == "w8a8" else "none",
        "calibrationSamples": len(samples or []),
        "rgbFoldMaxAbsoluteError": rgb_error,
        "conversionMaxAbsoluteError": conversion_error,
    }
    if component == "recognizer":
        metadata["argmaxMismatches"] = argmax_mismatches
    return contract, metadata


def build_tier(
    output_tier: str,
    config: dict[str, Any],
    version: str,
    sources_root: Path,
    work_root: Path,
    output_root: Path,
    calibration_corpus: tuple[Path, dict[str, Any]] | None,
) -> dict[str, Any]:
    output = output_root / output_tier
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    detector_source_tier = config["detectorSourceTier"]
    recognizer_source_tier = config["recognizerSourceTier"]
    source_dirs: dict[str, Path] = {}
    for component, source_tier in (
        ("detector", detector_source_tier),
        ("recognizer", recognizer_source_tier),
    ):
        source = source_for(source_tier, component)
        source_dirs[component] = fetch(
            source["repository"],
            source["revision"],
            sources_root / source_tier / component,
        )

    print(f"\n=== {output_tier} ===")
    detector_shape = (1, 3, config["detectorInputSize"], config["detectorInputSize"])
    recognizer_shape = (
        1,
        3,
        config["recognizerInputHeight"],
        config["recognizerInputWidth"],
    )
    detector_contract, detector_metadata = build_component(
        output_tier,
        "detector",
        detector_source_tier,
        source_dirs["detector"],
        work_root,
        output,
        detector_shape,
        config["detectorCompression"],
        config["detectorIOType"],
        calibration_corpus,
    )
    recognizer_contract, recognizer_metadata = build_component(
        output_tier,
        "recognizer",
        recognizer_source_tier,
        source_dirs["recognizer"],
        work_root,
        output,
        recognizer_shape,
        config["recognizerCompression"],
        config["recognizerIOType"],
        calibration_corpus,
    )

    characters = recognizer_contract["characterDict"]
    if any("\n" in character for character in characters):
        raise SystemExit(f"{output_tier}: character dictionary contains a newline")
    (output / "charset.txt").write_text("\n".join(characters) + "\n", encoding="utf-8")

    archive = output_root / f"pp-ocr-v6-coreml-{output_tier}-v{version}.aar"
    archive.unlink(missing_ok=True)
    subprocess.run(
        ["aa", "archive", "-d", str(output), "-o", str(archive), "-a", "lzfse", "-b", "1m"],
        check=True,
    )

    detector_mean = list(reversed(detector_contract["upstreamMean"]))
    detector_std = list(reversed(detector_contract["upstreamStd"]))
    return {
        "tier": output_tier,
        "upstream": {
            "detectorHmean": UPSTREAM_METRICS[detector_source_tier]["detectorHmean"],
            "recognizerWAvg": UPSTREAM_METRICS[recognizer_source_tier]["recognizerWAvg"],
        },
        "detector": {
            **detector_metadata,
            "colorOrder": "RGB",
            "mean": detector_mean,
            "std": detector_std,
            "postProcess": detector_contract["postProcess"],
        },
        "recognizer": {
            **recognizer_metadata,
            "colorOrder": "RGB",
            "mean": REC_MEAN,
            "std": REC_STD,
            "charactersCount": len(characters),
        },
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--work", type=Path, default=Path("build"), help="intermediate directory")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE, help="versioned build recipe")
    parser.add_argument("--tier", action="append", choices=TIERS, help="build one or more tiers")
    parser.add_argument(
        "--calibration-corpus",
        type=Path,
        help="public schema-v1 corpus; required by a selected W8A8 component",
    )
    args = parser.parse_args()

    recipe_path = args.recipe.resolve()
    recipe = load_recipe(recipe_path)
    work_root = args.work.resolve()
    recipe_work = work_root / recipe["name"]
    output_root = args.out.resolve()
    sources_root = recipe_work / "sources"
    recipe_work.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = args.tier or list(TIERS)
    selected_uses_w8a8 = any(
        recipe["tiers"][tier][f"{component}Compression"] == "w8a8"
        for tier in selected for component in COMPONENTS
    )
    if selected_uses_w8a8 and args.calibration_corpus is None:
        raise SystemExit("the selected W8A8 recipe requires --calibration-corpus")
    corpus = load_calibration_corpus(args.calibration_corpus)
    packs = [
        build_tier(
            tier,
            recipe["tiers"][tier],
            recipe["packVersion"],
            sources_root,
            recipe_work / "converted",
            output_root,
            corpus,
        )
        for tier in selected
    ]
    manifest = {
        "schemaVersion": 2,
        "packVersion": recipe["packVersion"],
        "releaseTag": f"v{recipe['packVersion']}",
        "recipe": {
            "name": recipe["name"],
            "file": recipe_path.name,
            "sha256": sha256(recipe_path),
        },
        "upstream": {
            "project": "PaddleOCR",
            "modelFamily": SOURCE_LOCK["modelFamily"],
            "license": SOURCE_LOCK["license"],
        },
        "conversion": {
            "minimumDeploymentTarget": recipe["minimumDeploymentTarget"],
            "computeUnits": recipe["computeUnits"],
            "archiveFormat": "AppleArchive",
            "compression": "lzfse",
        },
        "activationCalibration": (
            {
                "corpusFile": corpus[0].name,
                "sha256": sha256(corpus[0]),
                "maximumSamplesPerComponent": 32,
            }
            if selected_uses_w8a8 and corpus is not None else None
        ),
        "packs": packs,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "SHA256SUMS").write_text(
        "".join(f"{pack['archive']['sha256']}  {pack['archive']['name']}\n" for pack in packs),
        encoding="utf-8",
    )

    for pack in packs:
        print(
            f"{pack['tier']:<7} {pack['archive']['bytes'] / 1048576:6.1f} MB  "
            f"sha256 {pack['archive']['sha256'][:16]}..."
        )


if __name__ == "__main__":
    main()
