# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests: NVIDIA Cosmos3-Edge Reasoner against real weights.

Verifies the exported Reasoner graphs stage by stage using the real
``nvidia/Cosmos3-Edge`` checkpoint and the PyTorch transcription of the
published ``cosmos3_edge`` modeling code in
:mod:`tests._cosmos3_edge_reference`::

    pytest tests/cosmos3_edge_integration_test.py -m integration -sv

To test an existing export without rebuilding, set ``COSMOS3_EDGE_EXPORT``
to its directory and ``COSMOS3_EDGE_REFERENCE`` to the original checkpoint
snapshot directory, then select ``-k saved_vision``. This loads only the saved
vision encoder. ``COSMOS3_EDGE_PROVIDER`` defaults to ``CUDAExecutionProvider``;
the requested provider must be installed. The other tests still rebuild FP32
Reasoner graphs and do not inspect the saved export.

For a separate FP32 CPU baseline using the same local checkpoint, set
``COSMOS3_EDGE_REFERENCE`` and select ``-k fp32_vision_baseline``. Only vision
weights are loaded; the temporary graph does not replace the saved export.

Select ``-k bf16_vision_unfused`` with both paths set to compare an
``onnx-standard`` BF16 vision baseline against the saved CUDA export. Optional
ORT graph optimizations are disabled for the unfused control.

Select ``-k bf16_vision_pytorch`` with both paths set to compare the saved
BF16 CUDA graph with the installed Transformers vision tower and projector.
The FP32 reference runs on CPU. BF16 references (eager and SDPA attention) use
CUDA when available; set ``COSMOS3_EDGE_TORCH_DEVICE=cpu`` or ``cuda`` to select
explicitly. Matching devices and dtypes does not guarantee identical kernels.

Select ``-k bf16_vision_capture`` with both paths set to capture matching CUDA
BF16 checkpoints into pytest's temporary directory. This validates capture
coverage, shapes, and finite values only; it does not assert numerical parity
between runtimes. It requires the instrumented ONNX final output to match the
original export exactly for this input and records per-checkpoint error metrics.
The report also replays block 0 projections with identical normalized inputs
to distinguish fused-linear rounding from separate BF16 matmul-plus-bias.

Stages checked: SigLIP2 vision tower + merger projector (image and video),
image/video token fusion in the embedding graph, decoder logits under
interleaved 3D M-RoPE, and a cached greedy-decode generation smoke test.

.. note::
   Only the *understanding* (Reasoner) tower is verified. The Cosmos3-Edge
   Generator, Action head and Sound tower share the same checkpoint but are
   proprietary rectified-flow components with no published reference
   implementation, so their numerics remain unverifiable here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tests._cosmos3_edge_reference import (
    EdgeRefConfig,
    patchify_images,
    patchify_videos,
    ref_text_decoder_logits,
    ref_vision_features,
    smart_resize,
)

MODEL_ID = "nvidia/Cosmos3-Edge"
PATCH_SIZE = 16
MERGE_SIZE = 2
IMAGE_TOKEN_ID = 19
VIDEO_TOKEN_ID = 18

pytestmark = [pytest.mark.integration, pytest.mark.integration_slow]


def _checkpoint_dir() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, allow_patterns=["*.json", "*/*.safetensors"])


def _normalized(image: torch.Tensor) -> torch.Tensor:
    """Apply the checkpoint's ``image_mean``/``image_std`` of 0.5."""
    return (image - 0.5) / 0.5


def _split_image(height: int, width: int) -> torch.Tensor:
    """Left half red, right half green — a deterministic, describable image."""
    image = torch.zeros(1, 3, height, width, dtype=torch.float32)
    image[:, 0, :, : width // 2] = 1.0
    image[:, 1, :, width // 2 :] = 1.0
    return _normalized(image)


def test_saved_vision_features_match_reference():
    """Run the unmodified saved vision graph against original checkpoint weights."""
    import ml_dtypes
    import onnx_ir as ir
    import onnxruntime as ort
    from safetensors import safe_open

    export = os.environ.get("COSMOS3_EDGE_EXPORT")
    if not export:
        pytest.skip("Set COSMOS3_EDGE_EXPORT to test an existing ONNX export")
    directory = Path(export)
    model_path = directory / "reasoner_vision_encoder" / "model.onnx"
    assert model_path.is_file(), f"Saved vision graph not found: {model_path}"
    provider = os.environ.get("COSMOS3_EDGE_PROVIDER", "CUDAExecutionProvider")
    assert provider in ort.get_available_providers(), (
        f"Requested {provider}; installed providers: {ort.get_available_providers()}. "
        "Install a compatible ORT build; this test does not silently substitute CPU."
    )
    if provider == "CUDAExecutionProvider":
        ort.preload_dlls()
    session = ort.InferenceSession(str(model_path), providers=[provider])
    assert session.get_providers()[0] == provider

    reference_path = os.environ.get("COSMOS3_EDGE_REFERENCE")
    assert reference_path, "Set COSMOS3_EDGE_REFERENCE to the export's checkpoint snapshot"
    snapshot = Path(reference_path)
    ref_config = EdgeRefConfig.from_hf_config(
        json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    )
    export_config = EdgeRefConfig.from_hf_config(
        json.loads((directory / "config.json").read_text(encoding="utf-8"))
    )
    assert export_config == ref_config, "Export and reference architecture configs differ"
    reference = {}
    for shard in sorted(snapshot.rglob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for name in sorted(weights.keys()):
                if name.startswith(("model.visual.", "model.projector.")):
                    reference[name.removeprefix("model.")] = weights.get_tensor(name).float()
    assert reference, f"No vision/projector reference weights found in {snapshot}"

    inputs = {value.name: value for value in session.get_inputs()}
    assert set(inputs) == {"pixel_values", "grid_thw"}
    assert inputs["pixel_values"].shape[1] == PATCH_SIZE * PATCH_SIZE * 3
    assert inputs["grid_thw"].shape == [3]
    output = session.get_outputs()[0]
    assert output.name == "image_features"
    assert output.type == inputs["pixel_values"].type
    dtype, element_type = {
        "tensor(float)": (np.float32, ir.DataType.FLOAT),
        "tensor(float16)": (np.float16, ir.DataType.FLOAT16),
        "tensor(bfloat16)": (ml_dtypes.bfloat16, ir.DataType.BFLOAT16),
    }[output.type]
    packed, grid_h, grid_w = patchify_images(
        _split_image(256, 256), patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    pixels = np.ascontiguousarray(packed[0].numpy(), dtype=dtype)
    grid = np.array([1, grid_h, grid_w], dtype=np.int64)
    got = np.empty((grid_h * grid_w // MERGE_SIZE**2, ref_config.hidden_size), dtype=dtype)
    binding = session.io_binding()
    pixel_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(pixels, int(element_type))
    binding.bind_ortvalue_input("pixel_values", pixel_value)
    binding.bind_cpu_input("grid_thw", grid)
    binding.bind_output(
        output.name,
        device_type="cpu",
        device_id=0,
        element_type=int(element_type),
        shape=got.shape,
        buffer_ptr=got.ctypes.data,
    )
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    got = got.astype(np.float32)
    with torch.inference_mode():
        expected = ref_vision_features(
            torch.from_numpy(pixels.astype(np.float32)),
            torch.from_numpy(grid.reshape(1, 3)),
            reference,
            ref_config,
        ).numpy()
    assert np.isfinite(got).all()
    difference = np.abs(got - expected)
    correlation = np.corrcoef(got.reshape(-1), expected.reshape(-1))[0, 1]
    print(
        f"\nSaved graph: {model_path}\nProvider: {provider}; dtype: {output.type}; "
        f"shape: {got.shape}\nMax error: {difference.max():.6g}; "
        f"mean error: {difference.mean():.6g}; correlation: {correlation:.8f}"
    )
    tolerance = 1e-3 if dtype == np.float32 else 1e-2
    np.testing.assert_allclose(got, expected, atol=tolerance, rtol=tolerance)


def _run_bf16_vision(session, pixels, grid, output_shape):
    return _run_bf16_vision_outputs(session, pixels, grid, {"image_features": output_shape})[
        "image_features"
    ]


def _run_bf16_vision_outputs(session, pixels, grid, output_shapes):
    import ml_dtypes
    import onnx_ir as ir
    import onnxruntime as ort

    assert session.get_providers()[0] == "CUDAExecutionProvider"
    assert session.get_inputs()[0].type == "tensor(bfloat16)"
    assert session.get_outputs()[0].type == "tensor(bfloat16)"
    pixels = np.ascontiguousarray(pixels, dtype=ml_dtypes.bfloat16)
    outputs = {
        name: np.empty(shape, dtype=ml_dtypes.bfloat16)
        for name, shape in output_shapes.items()
    }
    pixel_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
        pixels, int(ir.DataType.BFLOAT16)
    )
    binding = session.io_binding()
    binding.bind_ortvalue_input("pixel_values", pixel_value)
    binding.bind_cpu_input("grid_thw", grid)
    for name, output in outputs.items():
        binding.bind_output(
            name,
            device_type="cpu",
            device_id=0,
            element_type=int(ir.DataType.BFLOAT16),
            shape=output.shape,
            buffer_ptr=output.ctypes.data,
        )
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    return {name: output.astype(np.float32) for name, output in outputs.items()}


def _capture_onnx_vision(model_path, capture_dir, pixels, grid, config):
    import onnx_ir as ir
    import onnxruntime as ort

    model = ir.load(model_path)
    values = [value for node in model.graph for value in node.outputs]

    def find(prefix):
        matches = [value for value in values if (value.name or "").startswith(prefix)]
        assert len(matches) == 1, f"Expected one checkpoint for {prefix}: {matches}"
        return matches[0]

    def residual(normalized):
        node = normalized.producer()
        assert node.domain == "com.microsoft" and node.op_type == "SkipLayerNormalization"
        value = node.outputs[3]
        value.type = normalized.type
        value.shape = normalized.shape
        return value

    prefix = "v_vision_encoder.visual."
    checkpoints = {
        "patch_projection": find(prefix + "embeddings.patch_embedding.Add_"),
        "embeddings": residual(find(prefix + "encoder.layers.0.layer_norm1.")),
        "block_00_norm1": find(prefix + "encoder.layers.0.layer_norm1."),
    }
    for label, suffix in {
        "block_00_query": "self_attn.q_proj.Add_",
        "block_00_key": "self_attn.k_proj.Add_",
        "block_00_value": "self_attn.v_proj.Add_",
        "block_00_attention_context": "self_attn.Attention_",
        "block_00_norm2": "layer_norm2.",
    }.items():
        checkpoints[label] = find(prefix + "encoder.layers.0." + suffix)
    checkpoints["block_00_attention_residual"] = residual(checkpoints["block_00_norm2"])
    for index in range(config.vision_config.num_hidden_layers):
        next_norm = (
            f"encoder.layers.{index + 1}.layer_norm1."
            if index + 1 < config.vision_config.num_hidden_layers
            else "post_layernorm."
        )
        checkpoints[f"block_{index:02d}"] = residual(find(prefix + next_norm))
    checkpoints["post_layernorm"] = find(prefix + "post_layernorm.")
    checkpoints["projector"] = model.graph.outputs[0]
    for value in checkpoints.values():
        if value not in model.graph.outputs:
            model.graph.outputs.append(value)
    diagnostic_path = capture_dir / "vision_checkpoints.onnx"
    ir.save(model, diagnostic_path, external_data="vision_checkpoints.onnx.data")
    patch_count = pixels.shape[0]
    shapes = {}
    for label, value in checkpoints.items():
        if label == "projector":
            shape = (patch_count // MERGE_SIZE**2, config.text_config.hidden_size)
        elif label == "patch_projection":
            shape = (patch_count, config.vision_config.hidden_size)
        else:
            shape = (int(grid[0]), int(grid[1] * grid[2]), config.vision_config.hidden_size)
        shapes[value.name] = shape
    session = ort.InferenceSession(str(diagnostic_path), providers=["CUDAExecutionProvider"])
    outputs = _run_bf16_vision_outputs(session, pixels, grid, shapes)
    del session
    captured = {
        label: outputs[value.name].reshape(-1, shapes[value.name][-1])
        for label, value in checkpoints.items()
    }
    session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    original = _run_bf16_vision(session, pixels, grid, captured["projector"].shape)
    np.testing.assert_array_equal(
        captured["projector"],
        original,
        err_msg="Exposing checkpoints changed the final output; do not attribute these captures to the original execution",
    )
    print("Instrumented ONNX final output exactly matches the original export")
    manifest = {
        "source": str(model_path.resolve()),
        "torch_version": torch.__version__,
        "ort_version": ort.__version__,
        "attention": "sdpa",
        "dtype": "bfloat16",
        "storage_dtype": "float32",
        "grid_thw": grid.tolist(),
        "checkpoints": {
            label: {"onnx_value": value.name, "onnx_shape": shapes[value.name]}
            for label, value in checkpoints.items()
        },
    }
    (capture_dir / "checkpoints.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return captured


def test_fp32_vision_baseline_matches_reference(tmp_path):
    """Isolate full-size FP32 vision numerics from the saved BF16 CUDA graph."""
    _check_vision_baseline(tmp_path, dtype="f32")


def _vision_error_metrics(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    difference = actual - expected
    return {
        "mean_abs": float(np.abs(difference).mean()),
        "max_abs": float(np.abs(difference).max()),
        "relative_l2": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(expected), np.finfo(np.float32).tiny)
        ),
        "outside_tolerance_fraction": float(
            np.mean(~np.isclose(actual, expected, atol=1e-2, rtol=1e-2))
        ),
        "exact_fraction": float(np.mean(actual == expected)),
    }


def _replay_vision_projections(captures, reference, config, device):
    import torch.nn.functional as functional

    hidden = torch.from_numpy(captures["block_00_norm1"]).to(device, torch.bfloat16)
    metrics = {}
    with torch.inference_mode():
        for projection, label in (("q", "query"), ("k", "key"), ("v", "value")):
            prefix = f"visual.encoder.layers.0.self_attn.{projection}_proj."
            weight = reference[prefix + "weight"].to(device, torch.bfloat16)
            bias = reference[prefix + "bias"].to(device, torch.bfloat16)
            expected = captures[f"block_00_{label}"]
            for mode, output in (
                ("fused_linear", functional.linear(hidden, weight, bias)),
                ("separate_matmul_add", functional.linear(hidden, weight) + bias),
            ):
                metrics[f"{label}/{mode}"] = _vision_error_metrics(
                    output.float().cpu().numpy(), expected
                )
        num_heads = config.vision_config.num_attention_heads
        head_dim = config.vision_config.hidden_size // num_heads
        query, key, value = (
            torch.from_numpy(captures[f"block_00_{label}"])
            .to(device, torch.bfloat16)
            .reshape(1, hidden.shape[0], num_heads, head_dim)
            .transpose(1, 2)
            for label in ("query", "key", "value")
        )
        context = (
            functional.scaled_dot_product_attention(
                query, key, value, scale=float(np.float32(head_dim**-0.5))
            )
            .transpose(1, 2)
            .reshape(hidden.shape)
        )
        metrics["attention/same_qkv"] = _vision_error_metrics(
            context.float().cpu().numpy(), captures["block_00_attention_context"]
        )
    for label, values in metrics.items():
        print(f"Same-input replay {label}: {values}")
    return metrics


def test_bf16_vision_pytorch_matches_saved_export():
    """Compare saved CUDA output with upstream PyTorch BF16, without rebuilding."""
    _check_bf16_vision_pytorch()


def test_bf16_vision_capture(tmp_path):
    """Capture checkpoints only; numerical parity is a separate diagnostic."""
    _check_bf16_vision_pytorch(capture_dir=tmp_path)


def _check_bf16_vision_pytorch(*, capture_dir=None):
    from functools import partial

    import onnxruntime as ort
    from safetensors import safe_open
    from transformers.models.cosmos3_edge.configuration_cosmos3_edge import Cosmos3EdgeConfig
    from transformers.models.cosmos3_edge.modeling_cosmos3_edge import (
        Cosmos3EdgePatchMerger,
        Cosmos3EdgeVisionModel,
    )

    export = os.environ.get("COSMOS3_EDGE_EXPORT")
    reference_path = os.environ.get("COSMOS3_EDGE_REFERENCE")
    if not export or not reference_path:
        pytest.skip("Set COSMOS3_EDGE_EXPORT and COSMOS3_EDGE_REFERENCE")
    bf16_device = torch.device(
        os.environ.get(
            "COSMOS3_EDGE_TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    if capture_dir is not None:
        assert bf16_device.type == "cuda", "Checkpoint capture requires CUDA PyTorch"
    if bf16_device.type == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA PyTorch is required for the requested reference"
        )
        assert torch.cuda.is_bf16_supported(), "The GPU must support BF16"
        print(f"\nPyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(bf16_device)}")
    snapshot = Path(reference_path)
    config_data = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config = Cosmos3EdgeConfig.from_dict(config_data)
    reference = {}
    for shard in sorted(snapshot.rglob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for name in sorted(weights.keys()):
                if name.startswith(("model.visual.", "model.projector.")):
                    reference[name.removeprefix("model.")] = weights.get_tensor(name)
    assert reference, "No vision/projector checkpoint weights found"
    packed, grid_h, grid_w = patchify_images(
        _split_image(256, 256), patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    grid = np.array([1, grid_h, grid_w], dtype=np.int64)
    output_shape = (grid_h * grid_w // MERGE_SIZE**2, config.text_config.hidden_size)
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    ort.preload_dlls()
    model_path = Path(export) / "reasoner_vision_encoder" / "model.onnx"
    if capture_dir is not None:
        onnx_captures = _capture_onnx_vision(
            model_path, capture_dir, packed[0].numpy(), grid, config
        )
        np.savez(capture_dir / "inputs.npz", pixel_values=packed[0].numpy(), grid_thw=grid)
        np.savez(capture_dir / "onnx_checkpoints.npz", **onnx_captures)
    else:
        session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
        got = _run_bf16_vision(session, packed[0].numpy(), grid, output_shape)
        del session
    results = {}
    runs = (
        (
            (torch.float32, "sdpa"),
            (torch.bfloat16, "eager"),
            (torch.bfloat16, "sdpa"),
        )
        if capture_dir is None
        else ((torch.bfloat16, "sdpa"),)
    )
    for precision, attention in runs:
        device = torch.device("cpu") if precision == torch.float32 else bf16_device
        config.vision_config._attn_implementation = attention
        with torch.device("meta"):
            visual = Cosmos3EdgeVisionModel(config.vision_config)
            projector = Cosmos3EdgePatchMerger(config)
        for prefix, module in (("visual.", visual), ("projector.", projector)):
            module.load_state_dict(
                {
                    name.removeprefix(prefix): value.to(device=device, dtype=precision)
                    for name, value in reference.items()
                    if name.startswith(prefix)
                },
                strict=True,
                assign=True,
            )
            module.eval()
        torch_captures = {}
        handles = []
        if capture_dir is not None:
            modules = {
                "patch_projection": visual.embeddings.patch_embedding,
                "embeddings": visual.embeddings,
                "block_00_norm1": visual.encoder.layers[0].layer_norm1,
                "block_00_query": visual.encoder.layers[0].self_attn.q_proj,
                "block_00_key": visual.encoder.layers[0].self_attn.k_proj,
                "block_00_value": visual.encoder.layers[0].self_attn.v_proj,
                "block_00_norm2": visual.encoder.layers[0].layer_norm2,
                **{
                    f"block_{index:02d}": layer
                    for index, layer in enumerate(visual.encoder.layers)
                },
                "post_layernorm": visual.post_layernorm,
                "projector": projector,
            }

            def capture(captures, label, module, inputs, output):
                assert output.dtype == torch.bfloat16 and output.is_cuda
                assert label not in captures, f"Repeated checkpoint: {label}"
                captures[label] = output.detach().float().cpu().numpy()

            handles = [
                module.register_forward_hook(partial(capture, torch_captures, label))
                for label, module in modules.items()
            ]

            def capture_input(callback, captures, label, module, inputs):
                callback(captures, label, module, inputs, inputs[0])

            handles.append(
                visual.encoder.layers[0].self_attn.out_proj.register_forward_pre_hook(
                    partial(
                        capture_input, capture, torch_captures, "block_00_attention_context"
                    )
                )
            )
            handles.append(
                visual.encoder.layers[0].layer_norm2.register_forward_pre_hook(
                    partial(
                        capture_input, capture, torch_captures, "block_00_attention_residual"
                    )
                )
            )
        try:
            with torch.inference_mode():
                features = visual(
                    packed[0].to(device=device, dtype=precision),
                    torch.from_numpy(grid.reshape(1, 3)).to(device),
                ).last_hidden_state
                output = projector(features)
        finally:
            for handle in handles:
                handle.remove()
        assert output.dtype == precision
        assert output.device.type == device.type
        results[f"{device}/{precision}/{attention}"] = output.float().cpu().numpy()
        del visual, projector, module, features, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if capture_dir is not None:
        assert onnx_captures.keys() == torch_captures.keys()
        assert len(torch_captures) == config.vision_config.num_hidden_layers + 11
        metrics = {}
        for label, expected in torch_captures.items():
            actual = onnx_captures[label]
            assert actual.shape == expected.shape, label
            assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
            metrics[label] = _vision_error_metrics(actual, expected)
            print(f"{label}: {metrics[label]}")
        np.savez(capture_dir / "torch_checkpoints.npz", **torch_captures)
        (capture_dir / "comparison.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        replay = _replay_vision_projections(onnx_captures, reference, config, bf16_device)
        (capture_dir / "same_input_replay.json").write_text(
            json.dumps(replay, indent=2), encoding="utf-8"
        )
        print(f"\nCheckpoint artifacts: {capture_dir}")
        return
    fp32 = results["cpu/torch.float32/sdpa"]
    with torch.inference_mode():
        transcribed = ref_vision_features(
            packed[0],
            torch.from_numpy(grid.reshape(1, 3)),
            {name: value.float() for name, value in reference.items()},
            EdgeRefConfig.from_hf_config(config_data),
        ).numpy()
    np.testing.assert_allclose(fp32, transcribed, atol=1e-3, rtol=1e-3)
    failures = []
    for label, expected in results.items():
        assert np.isfinite(expected).all()
        for source_name, source in (("saved BF16 ONNX", got), ("upstream FP32", fp32)):
            error = np.abs(source - expected)
            correlation = np.corrcoef(source.reshape(-1), expected.reshape(-1))[0, 1]
            print(
                f"\n{source_name} vs upstream {label}: "
                f"max error: {error.max():.6g}; mean error: {error.mean():.6g}; "
                f"correlation: {correlation:.8f}"
            )
        if "bfloat16" in label:
            try:
                np.testing.assert_allclose(got, expected, atol=1e-2, rtol=1e-2)
            except AssertionError as error:
                failures.append(f"{label}: {error}")
    assert not failures, "\n".join(failures)


def test_bf16_vision_unfused_matches_reference(tmp_path):
    """Compare unfused BF16 CUDA, the saved fused export, and the FP32 reference."""
    _check_vision_baseline(tmp_path, dtype="bf16")


def _check_vision_baseline(tmp_path, *, dtype):
    import onnx_ir as ir
    import onnxruntime as ort
    from safetensors import safe_open

    import mobius
    from mobius._model_package import ModelPackage
    from mobius.models.cosmos import Cosmos3EdgeVLModel

    reference_path = os.environ.get("COSMOS3_EDGE_REFERENCE")
    if not reference_path:
        pytest.skip("Set COSMOS3_EDGE_REFERENCE to the original checkpoint snapshot")
    snapshot = Path(reference_path)
    if dtype == "bf16":
        export = os.environ.get("COSMOS3_EDGE_EXPORT")
        if not export:
            pytest.skip("Set COSMOS3_EDGE_EXPORT to compare the saved fused export")
        assert "CUDAExecutionProvider" in ort.get_available_providers()
        ort.preload_dlls()
    ref_config = EdgeRefConfig.from_hf_config(
        json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    )
    package = mobius.build(
        str(snapshot),
        task="cosmos3-edge-vl",
        dtype=dtype,
        load_weights=False,
        execution_provider="onnx-standard" if dtype == "bf16" else "default",
    )
    module = Cosmos3EdgeVLModel(package.config)
    package = ModelPackage(
        {"vision_encoder": package["vision_encoder"]}, config=package.config
    )
    reference = {}
    for shard in sorted(snapshot.rglob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            vision_weights = {
                name: weights.get_tensor(name).float()
                for name in sorted(weights.keys())
                if name.startswith(("model.visual.", "model.projector."))
            }
        if vision_weights:
            package.apply_weights_partial(module.preprocess_weights(vision_weights))
            reference.update(
                {name.removeprefix("model."): value for name, value in vision_weights.items()}
            )
    assert reference, f"No vision/projector reference weights found in {snapshot}"
    package.finalize_weights()
    package.validate_weights()
    path = tmp_path / f"vision_{dtype}.onnx"
    if dtype == "bf16":
        custom_ops = {
            (node.domain, node.op_type)
            for node in package["vision_encoder"].graph
            if node.domain not in ("", "ai.onnx")
        }
        assert not custom_ops, f"Unfused control contains custom operators: {custom_ops}"
    ir.save(package["vision_encoder"], path, external_data=f"vision_{dtype}.onnx.data")
    packed, grid_h, grid_w = patchify_images(
        _split_image(256, 256), patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    grid = np.array([1, grid_h, grid_w], dtype=np.int64)
    with torch.inference_mode():
        expected = ref_vision_features(
            packed[0], torch.from_numpy(grid.reshape(1, 3)), reference, ref_config
        ).numpy()
    if dtype == "bf16":
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CUDAExecutionProvider"]
        )
        got = _run_bf16_vision(session, packed[0].numpy(), grid, expected.shape)
        del session
        fused_path = Path(export) / "reasoner_vision_encoder" / "model.onnx"
        for label, session_options in (("disabled", options), ("default", None)):
            session = ort.InferenceSession(
                str(fused_path),
                sess_options=session_options,
                providers=["CUDAExecutionProvider"],
            )
            fused = _run_bf16_vision(session, packed[0].numpy(), grid, expected.shape)
            del session
            assert np.isfinite(fused).all()
            for target_name, target in (("FP32 reference", expected), ("unfused BF16", got)):
                error = np.abs(fused - target)
                correlation = np.corrcoef(fused.reshape(-1), target.reshape(-1))[0, 1]
                print(
                    f"\nSaved fused (ORT optimizations {label}) vs {target_name}: "
                    f"max error: {error.max():.6g}; mean error: {error.mean():.6g}; "
                    f"correlation: {correlation:.8f}"
                )
    else:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = session.run(None, {"pixel_values": packed[0].numpy(), "grid_thw": grid})[0]
    assert got.dtype == expected.dtype == np.float32
    assert np.isfinite(got).all()
    difference = np.abs(got - expected)
    correlation = np.corrcoef(got.reshape(-1), expected.reshape(-1))[0, 1]
    print(
        f"\n{dtype} baseline vs FP32 reference: {path}\nShape: {got.shape}; "
        f"max error: {difference.max():.6g}; mean error: {difference.mean():.6g}; "
        f"correlation: {correlation:.8f}"
    )
    tolerance = 1e-3 if dtype == "f32" else 1e-2
    np.testing.assert_allclose(got, expected, atol=tolerance, rtol=tolerance)


@pytest.fixture(scope="module")
def edge_package(tmp_path_factory):
    """Build the Reasoner graphs with real fp32 weights and open ORT sessions."""
    import onnx_ir as ir
    import onnxruntime as ort

    import mobius
    from mobius.integrations._weight_loading import iter_weight_shards
    from mobius.models.cosmos import Cosmos3EdgeVLModel

    snapshot = _checkpoint_dir()
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as handle:
        ref_config = EdgeRefConfig.from_hf_config(json.load(handle))

    package = mobius.build(MODEL_ID, task="cosmos3-edge-vl", dtype="f32", load_weights=False)
    module = Cosmos3EdgeVLModel(package.config)
    reference: dict[str, torch.Tensor] = {}
    for shard in iter_weight_shards(MODEL_ID):
        package.apply_weights_partial(module.preprocess_weights(shard))
        for key, value in shard.items():
            if "k_norm_und_for_gen" in key or "moe_gen" in key:
                continue
            if key.startswith(("model.visual.", "model.projector.")):
                reference[key.removeprefix("model.")] = value.float()
            elif key.startswith("layers.") or key in (
                "embed_tokens.weight",
                "norm.weight",
                "lm_head.weight",
            ):
                reference[key] = value.float()
    package.finalize_weights()
    package.validate_weights()

    directory = tmp_path_factory.mktemp("cosmos3_edge_real")
    sessions = {}
    for name in ("vision_encoder", "embedding", "decoder"):
        path = directory / f"{name}.onnx"
        ir.save(package[name], str(path), external_data=f"{name}.onnx.data")
        sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return package.config, ref_config, sessions, reference


def test_vision_encoder_input_contract(edge_package):
    """The exported vision graph takes packed patches, not a single image."""
    _, _, sessions, _ = edge_package
    inputs = {value.name: value for value in sessions["vision_encoder"].get_inputs()}
    assert set(inputs) == {"pixel_values", "grid_thw"}
    assert inputs["pixel_values"].shape[1] == PATCH_SIZE * PATCH_SIZE * 3
    assert inputs["grid_thw"].shape == [3]


@pytest.mark.parametrize(("height", "width"), [(256, 256), (128, 512), (320, 192)])
def test_vision_features_match_reference(edge_package, height, width):
    _, ref_config, sessions, reference = edge_package
    image = _split_image(height, width)
    packed, grid_h, grid_w = patchify_images(
        image, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    packed = packed[0]

    got = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": packed.numpy(),
            "grid_thw": np.array([1, grid_h, grid_w], dtype=np.int64),
        },
    )[0]
    expected = ref_vision_features(
        packed, torch.tensor([[1, grid_h, grid_w]]), reference, ref_config
    ).numpy()

    assert got.shape == (grid_h * grid_w // MERGE_SIZE**2, ref_config.hidden_size)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)
    correlation = np.corrcoef(got.reshape(-1), expected.reshape(-1))[0, 1]
    assert correlation > 0.9999


def test_smart_resized_photo_resolution_matches_reference(edge_package):
    """Drive the processor's own ``smart_resize`` policy, not a hand-picked size.

    A natural 1000x750 photo is resized to a multiple of ``patch*merge`` (32)
    inside the checkpoint's pixel-area bounds, then patchified and encoded.
    """
    _, ref_config, sessions, reference = edge_package
    height, width = smart_resize(
        750,
        1000,
        factor=PATCH_SIZE * MERGE_SIZE,
        min_pixels=256 * 256,
        max_pixels=4096 * 4096,
    )
    assert height % (PATCH_SIZE * MERGE_SIZE) == 0
    assert width % (PATCH_SIZE * MERGE_SIZE) == 0
    assert 256 * 256 <= height * width <= 4096 * 4096

    image = _split_image(height, width)
    packed, grid_h, grid_w = patchify_images(
        image, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    got = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": packed[0].numpy(),
            "grid_thw": np.array([1, grid_h, grid_w], dtype=np.int64),
        },
    )[0]
    expected = ref_vision_features(
        packed[0], torch.tensor([[1, grid_h, grid_w]]), reference, ref_config
    ).numpy()

    assert got.shape == (grid_h * grid_w // MERGE_SIZE**2, ref_config.hidden_size)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)


def test_video_features_match_reference_and_token_count(edge_package):
    _, ref_config, sessions, reference = edge_package
    frames = 4
    video = torch.stack([_split_image(128, 160)[0] for _ in range(frames)]).unsqueeze(0)
    packed, grid_t, grid_h, grid_w = patchify_videos(
        video, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    packed = packed[0]

    got = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": packed.numpy(),
            "grid_thw": np.array([grid_t, grid_h, grid_w], dtype=np.int64),
        },
    )[0]
    expected = ref_vision_features(
        packed, torch.tensor([[grid_t, grid_h, grid_w]]), reference, ref_config
    ).numpy()

    tokens_per_frame = grid_h * grid_w // MERGE_SIZE**2
    assert got.shape == (frames * tokens_per_frame, ref_config.hidden_size)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)


def test_image_and_video_fusion_and_decoder_logits(edge_package):
    config, ref_config, sessions, reference = edge_package

    image = _split_image(256, 256)
    image_packed, img_h, img_w = patchify_images(
        image, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    image_features = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": image_packed[0].numpy(),
            "grid_thw": np.array([1, img_h, img_w], dtype=np.int64),
        },
    )[0]

    frames = 2
    video = torch.stack([_split_image(64, 96)[0] for _ in range(frames)]).unsqueeze(0)
    video_packed, vid_t, vid_h, vid_w = patchify_videos(
        video, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    video_features = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": video_packed[0].numpy(),
            "grid_thw": np.array([vid_t, vid_h, vid_w], dtype=np.int64),
        },
    )[0]

    ids = [101, 102]
    image_start = len(ids)
    ids += [IMAGE_TOKEN_ID] * image_features.shape[0]
    video_start = len(ids)
    ids += [VIDEO_TOKEN_ID] * video_features.shape[0]
    ids += [201]
    input_ids = np.array([ids], dtype=np.int64)

    inputs_embeds = sessions["embedding"].run(
        None,
        {
            "input_ids": input_ids,
            "image_features": image_features,
            "video_features": video_features,
        },
    )[0]

    expected_embeds = torch.nn.functional.embedding(
        torch.from_numpy(input_ids), reference["embed_tokens.weight"]
    )
    expected_embeds = expected_embeds.masked_scatter(
        torch.from_numpy(input_ids == IMAGE_TOKEN_ID).unsqueeze(-1),
        torch.from_numpy(image_features),
    )
    expected_embeds = expected_embeds.masked_scatter(
        torch.from_numpy(input_ids == VIDEO_TOKEN_ID).unsqueeze(-1),
        torch.from_numpy(video_features),
    )
    np.testing.assert_allclose(inputs_embeds, expected_embeds.numpy(), atol=1e-3)

    length = len(ids)
    positions = np.zeros((3, 1, length), dtype=np.int64)
    for index in range(image_start):
        positions[:, 0, index] = index
    base = image_start
    merged_w = img_w // MERGE_SIZE
    for token in range(image_features.shape[0]):
        positions[0, 0, image_start + token] = base
        positions[1, 0, image_start + token] = base + token // merged_w
        positions[2, 0, image_start + token] = base + token % merged_w
    base += max(img_h, img_w) // MERGE_SIZE
    tokens_per_frame = video_features.shape[0] // frames
    merged_vw = vid_w // MERGE_SIZE
    for token in range(video_features.shape[0]):
        frame, spatial = divmod(token, tokens_per_frame)
        positions[0, 0, video_start + token] = base + frame
        positions[1, 0, video_start + token] = base + spatial // merged_vw
        positions[2, 0, video_start + token] = base + spatial % merged_vw
    base += max(frames, vid_h // MERGE_SIZE, merged_vw)
    positions[:, 0, length - 1] = base

    feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((1, length), dtype=np.int64),
        "position_ids": positions,
    }
    empty = np.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32)
    for layer in range(config.num_hidden_layers):
        feeds[f"past_key_values.{layer}.key"] = empty
        feeds[f"past_key_values.{layer}.value"] = empty
    logits = sessions["decoder"].run(["logits"], feeds)[0]

    expected_logits = ref_text_decoder_logits(
        expected_embeds, torch.from_numpy(positions), reference, ref_config
    ).numpy()
    np.testing.assert_allclose(logits, expected_logits, atol=5e-3, rtol=5e-3)
    assert (logits.argmax(-1) == expected_logits.argmax(-1)).all()


def test_text_only_decoder_matches_reference(edge_package):
    """Regression guard: the text path must stay correct after the vision fix."""
    config, ref_config, sessions, reference = edge_package
    input_ids = np.array([[5, 77, 900, 12, 34, 56, 78, 90]], dtype=np.int64)
    length = input_ids.shape[1]
    empty_features = np.zeros((0, config.hidden_size), dtype=np.float32)

    inputs_embeds = sessions["embedding"].run(
        None,
        {
            "input_ids": input_ids,
            "image_features": empty_features,
            "video_features": empty_features,
        },
    )[0]
    expected_embeds = torch.nn.functional.embedding(
        torch.from_numpy(input_ids), reference["embed_tokens.weight"]
    )
    np.testing.assert_allclose(inputs_embeds, expected_embeds.numpy(), atol=0)

    positions = np.tile(np.arange(length, dtype=np.int64), (3, 1, 1))
    feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((1, length), dtype=np.int64),
        "position_ids": positions,
    }
    empty = np.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32)
    for layer in range(config.num_hidden_layers):
        feeds[f"past_key_values.{layer}.key"] = empty
        feeds[f"past_key_values.{layer}.value"] = empty
    logits = sessions["decoder"].run(["logits"], feeds)[0]

    expected = ref_text_decoder_logits(
        expected_embeds, torch.from_numpy(positions), reference, ref_config
    ).numpy()
    np.testing.assert_allclose(logits, expected, atol=5e-3, rtol=5e-3)
    assert (logits.argmax(-1) == expected.argmax(-1)).all()


def test_greedy_generation_smoke_with_image(edge_package):
    """Prefill with an image, then decode a few tokens through the KV cache."""
    config, _, sessions, _ = edge_package
    image = _split_image(256, 256)
    packed, grid_h, grid_w = patchify_images(
        image, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    image_features = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": packed[0].numpy(),
            "grid_thw": np.array([1, grid_h, grid_w], dtype=np.int64),
        },
    )[0]
    empty_features = np.zeros((0, config.hidden_size), dtype=np.float32)

    ids = [101, 20] + [IMAGE_TOKEN_ID] * image_features.shape[0] + [21, 102]
    input_ids = np.array([ids], dtype=np.int64)
    length = input_ids.shape[1]

    inputs_embeds = sessions["embedding"].run(
        None,
        {
            "input_ids": input_ids,
            "image_features": image_features,
            "video_features": empty_features,
        },
    )[0]
    feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((1, length), dtype=np.int64),
        "position_ids": np.tile(np.arange(length, dtype=np.int64), (3, 1, 1)),
    }
    empty = np.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32)
    for layer in range(config.num_hidden_layers):
        feeds[f"past_key_values.{layer}.key"] = empty
        feeds[f"past_key_values.{layer}.value"] = empty

    names = [value.name for value in sessions["decoder"].get_outputs()]
    generated: list[int] = []
    position = length
    for _ in range(4):
        outputs = dict(zip(names, sessions["decoder"].run(None, feeds), strict=True))
        logits = outputs["logits"]
        assert np.isfinite(logits).all()
        token = int(logits[0, -1].argmax())
        assert 0 <= token < config.vocab_size
        generated.append(token)
        step_embeds = sessions["embedding"].run(
            None,
            {
                "input_ids": np.array([[token]], dtype=np.int64),
                "image_features": empty_features,
                "video_features": empty_features,
            },
        )[0]
        feeds = {
            "inputs_embeds": step_embeds,
            "attention_mask": np.ones((1, position + 1), dtype=np.int64),
            "position_ids": np.full((3, 1, 1), position, dtype=np.int64),
        }
        for layer in range(config.num_hidden_layers):
            feeds[f"past_key_values.{layer}.key"] = outputs[f"present.{layer}.key"]
            feeds[f"past_key_values.{layer}.value"] = outputs[f"present.{layer}.value"]
        position += 1

    assert len(generated) == 4
