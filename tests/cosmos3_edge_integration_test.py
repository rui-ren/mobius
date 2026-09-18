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
coverage, shapes, and finite values; it does not assert BF16 numerical parity
between runtimes. Separate FP32 controls do assert parity. It requires the instrumented ONNX final output to match the
original export exactly for this input and records per-checkpoint error metrics.
The report also replays block 0 projections with identical normalized inputs
to distinguish fused-linear rounding from separate BF16 matmul-plus-bias.
Additional diagnostics replay Q/K/V-only rounding and combined encoder kernel
arithmetic (weight layout, GELU and residual ordering), then verify that the
unmodified reference is restored exactly. A separate unchanged-export session
disables only ORT's bias-to-SkipLayerNorm fusion. These are attribution
experiments, not replacement references or parity gates.

Per-block replays reset inputs to ONNX captures to separate local error from
accumulated drift. Blocks 0, 13 and 26 additionally compare identical-QKV SDPA
backends and exact-attention-context substitution; unavailable Flash builds
are reported explicitly. The substituted runs are attribution controls, not
end-to-end inference results.

ORT dispatch controls verify Flash, memory-efficient and unfused kernel records
on the unchanged export. Isolated Attention replays compare identical captured
BF16 Q/K/V against FP64 math, not against a full FP64 vision model.

With efficient attention aligned, first-block normalization substitutions test
the source of the remaining local drift. Same-residual FP64 normalization probes
measure precision separately; substituted outputs are not a production fix.

All-layer normalization controls additionally align the projector's fused Gemm
and BF16 normal-CDF rounding. Exact recovery is an attribution check on the
captured input, not evidence that the unmodified export matches HuggingFace.

Independent controls promote only GELU intermediates, or promote the entire
saved graph to FP32. The latter compares against upstream FP32 math attention
with TF32 disabled and identical BF16-rounded input pixels and weights, for
both the split image and an RGB gradient. These runs contain no substituted
checkpoints. They distinguish structural export errors from BF16 sensitivity;
activation-only promotion is not assumed to improve whole-model parity.

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
        "block_00_attention_projection": "self_attn.out_proj.MatMul_",
        "block_00_mlp_up": "mlp.up_proj.MatMul_",
        "block_00_mlp_activation": "mlp.Gelu_",
        "block_00_mlp_down": "mlp.down_proj.MatMul_",
        "block_00_norm2": "layer_norm2.",
    }.items():
        checkpoints[label] = find(prefix + "encoder.layers.0." + suffix)
    checkpoints["block_00_attention_residual"] = residual(checkpoints["block_00_norm2"])
    for index in (
        config.vision_config.num_hidden_layers // 2,
        config.vision_config.num_hidden_layers - 1,
    ):
        for label, suffix in {
            "norm1": "layer_norm1.",
            "query": "self_attn.q_proj.Add_",
            "key": "self_attn.k_proj.Add_",
            "value": "self_attn.v_proj.Add_",
            "attention_context": "self_attn.Attention_",
        }.items():
            checkpoints[f"block_{index:02d}_{label}"] = find(
                prefix + f"encoder.layers.{index}." + suffix
            )
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
        elif label in ("block_00_mlp_up", "block_00_mlp_activation"):
            shape = (
                int(grid[0]),
                int(grid[1] * grid[2]),
                config.vision_config.intermediate_size,
            )
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
        "ort_build": ort.get_build_info(),
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
    del session
    session = ort.InferenceSession(
        str(diagnostic_path),
        providers=[("CUDAExecutionProvider", {"sdpa_kernel": "2"})],
    )
    norm_labels = ("block_00_norm1", "block_00_norm2")
    normalized = _run_bf16_vision_outputs(
        session,
        pixels,
        grid,
        {checkpoints[label].name: shapes[checkpoints[label].name] for label in norm_labels},
    )
    np.savez(
        capture_dir / "aligned_norms.npz",
        **{
            label: normalized[checkpoints[label].name].reshape(captured[label].shape)
            for label in norm_labels
        },
    )
    del session
    all_checkpoints = dict(checkpoints)
    for index in range(config.vision_config.num_hidden_layers):
        for norm in (1, 2):
            label = f"block_{index:02d}_norm{norm}"
            value = find(prefix + f"encoder.layers.{index}.layer_norm{norm}.")
            all_checkpoints[label] = value
            shapes[value.name] = shapes[checkpoints["block_00_norm1"].name]
            if value not in model.graph.outputs:
                model.graph.outputs.append(value)
    for label, suffix, shape in (
        (
            "projector_norm",
            "norm.LayerNormalization_",
            (patch_count // MERGE_SIZE**2, MERGE_SIZE**2, config.vision_config.hidden_size),
        ),
        (
            "projector_up",
            "linear_fc1.Add_",
            (patch_count // MERGE_SIZE**2, config.projector_hidden_size),
        ),
        (
            "projector_activation",
            "Gelu_",
            (patch_count // MERGE_SIZE**2, config.projector_hidden_size),
        ),
    ):
        value = find("v_vision_encoder.projector." + suffix)
        all_checkpoints[label] = value
        shapes[value.name] = shape
        if value not in model.graph.outputs:
            model.graph.outputs.append(value)
    aligned_path = capture_dir / "aligned_vision_checkpoints.onnx"
    ir.save(model, aligned_path, external_data="aligned_vision_checkpoints.onnx.data")
    providers = [("CUDAExecutionProvider", {"sdpa_kernel": "2"})]
    session = ort.InferenceSession(str(aligned_path), providers=providers)
    outputs = _run_bf16_vision_outputs(session, pixels, grid, shapes)
    del session
    aligned = {
        label: outputs[value.name].reshape(-1, shapes[value.name][-1])
        for label, value in all_checkpoints.items()
    }
    session = ort.InferenceSession(str(model_path), providers=providers)
    original = _run_bf16_vision(session, pixels, grid, aligned["projector"].shape)
    np.testing.assert_array_equal(aligned["projector"], original)
    np.savez(capture_dir / "aligned_encoder_checkpoints.npz", **aligned)
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


def _bf16_fast_gelu(hidden):
    """Replay BF16 arithmetic from ORT f2c39fe2f's gelu_approximate_impl.cu.

    Source: https://github.com/microsoft/onnxruntime/blob/f2c39fe2f/
    onnxruntime/core/providers/cuda/tensor/gelu_approximate_impl.cu
    """
    half = hidden.new_tensor(0.5)
    linear = hidden.new_tensor(0.7978845608028654)
    cubic = hidden.new_tensor(0.035677408136300125)
    return hidden * (half + half * torch.tanh(hidden * (cubic * hidden * hidden + linear)))


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
        context = torch.from_numpy(captures["block_00_attention_context"]).to(
            device, torch.bfloat16
        )
        residual = torch.from_numpy(captures["embeddings"]).to(device, torch.bfloat16)
        prefix = "visual.encoder.layers.0.self_attn.out_proj."
        weight = reference[prefix + "weight"].to(device, torch.bfloat16)
        bias = reference[prefix + "bias"].to(device, torch.bfloat16)
        projected = functional.linear(context, weight)
        metrics["attention_projection/same_context"] = _vision_error_metrics(
            projected.float().cpu().numpy(), captures["block_00_attention_projection"]
        )
        projected = torch.from_numpy(captures["block_00_attention_projection"]).to(
            device, torch.bfloat16
        )
        variants = {
            "fused_linear_then_residual": functional.linear(context, weight, bias) + residual,
            "separate_bias_then_residual": (projected + bias) + residual,
            "separate_residual_then_bias": (projected + residual) + bias,
            "separate_residual_bias_then_projection": (residual + bias) + projected,
            "fp32_three_term_sum": (projected.float() + residual.float() + bias.float()).to(
                torch.bfloat16
            ),
            "fp32_bias_then_residual": (
                projected.float() + bias.float() + residual.float()
            ).to(torch.bfloat16),
        }
        for mode, output in variants.items():
            metrics[f"attention_residual/{mode}"] = _vision_error_metrics(
                output.float().cpu().numpy(), captures["block_00_attention_residual"]
            )
        residual = torch.from_numpy(captures["block_00_attention_residual"]).to(
            device, torch.bfloat16
        )
        prefix = "visual.encoder.layers.0."
        norm_weight = reference[prefix + "layer_norm2.weight"].to(device, torch.bfloat16)
        norm_bias = reference[prefix + "layer_norm2.bias"].to(device, torch.bfloat16)
        normalized = functional.layer_norm(
            residual,
            (hidden.shape[-1],),
            norm_weight,
            norm_bias,
            eps=config.vision_config.layer_norm_eps,
        )
        metrics["norm2/same_residual"] = _vision_error_metrics(
            normalized.float().cpu().numpy(), captures["block_00_norm2"]
        )
        normalized = torch.from_numpy(captures["block_00_norm2"]).to(device, torch.bfloat16)
        up_weight = reference[prefix + "mlp.fc1.weight"].to(device, torch.bfloat16)
        up_bias = reference[prefix + "mlp.fc1.bias"].to(device, torch.bfloat16)
        down_weight = reference[prefix + "mlp.fc2.weight"].to(device, torch.bfloat16)
        down_bias = reference[prefix + "mlp.fc2.bias"].to(device, torch.bfloat16)
        up = functional.linear(normalized, up_weight)
        metrics["mlp_up/same_normalized"] = _vision_error_metrics(
            up.float().cpu().numpy(), captures["block_00_mlp_up"]
        )
        contiguous_weight = up_weight.T.contiguous()
        metrics["mlp_up/contiguous_transposed_weight"] = _vision_error_metrics(
            (normalized @ contiguous_weight).float().cpu().numpy(),
            captures["block_00_mlp_up"],
        )
        previous_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_tf32 = False
            for mode, output in (
                ("no_reduced_precision", functional.linear(normalized, up_weight)),
                (
                    "fp32_accumulation",
                    functional.linear(normalized.float(), up_weight.float()).to(
                        torch.bfloat16
                    ),
                ),
            ):
                metrics[f"mlp_up/{mode}"] = _vision_error_metrics(
                    output.float().cpu().numpy(), captures["block_00_mlp_up"]
                )
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                previous_reduction
            )
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        up = torch.from_numpy(captures["block_00_mlp_up"]).to(device, torch.bfloat16)
        for mode, activated in (
            ("bf16_bias", functional.gelu(up + up_bias, approximate="tanh")),
            (
                "fp32_bias",
                functional.gelu(up.float() + up_bias.float(), approximate="tanh").to(
                    torch.bfloat16
                ),
            ),
            ("bf16_kernel_arithmetic", _bf16_fast_gelu(up + up_bias)),
        ):
            metrics[f"mlp_activation/{mode}"] = _vision_error_metrics(
                activated.float().cpu().numpy(), captures["block_00_mlp_activation"]
            )
        activated = torch.from_numpy(captures["block_00_mlp_activation"]).to(
            device, torch.bfloat16
        )
        down = functional.linear(activated, down_weight)
        metrics["mlp_down/same_activation"] = _vision_error_metrics(
            down.float().cpu().numpy(), captures["block_00_mlp_down"]
        )
        output = (residual + down_bias) + down
        metrics["mlp_residual/same_activation_residual_bias"] = _vision_error_metrics(
            output.float().cpu().numpy(), captures["block_00"]
        )
        for mode in ("upstream", "separate_linear_residual_bias"):
            up = (
                functional.linear(normalized, up_weight, up_bias)
                if mode == "upstream"
                else functional.linear(normalized, up_weight) + up_bias
            )
            activated = functional.gelu(up, approximate="tanh")
            output = (
                residual + functional.linear(activated, down_weight, down_bias)
                if mode == "upstream"
                else (residual + down_bias) + functional.linear(activated, down_weight)
            )
            metrics[f"mlp_residual/{mode}"] = _vision_error_metrics(
                output.float().cpu().numpy(), captures["block_00"]
            )
    for label, values in metrics.items():
        print(f"Same-input replay {label}: {values}")
    return metrics


def _replay_vision_attention(captures, reference, config, device):
    """Compare sampled attention kernels with identical source tensors."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    metrics = {}
    heads = config.vision_config.num_attention_heads
    head_dim = config.vision_config.hidden_size // heads
    for index in (
        0,
        config.vision_config.num_hidden_layers // 2,
        config.vision_config.num_hidden_layers - 1,
    ):
        prefix = f"block_{index:02d}_"
        expected = captures[prefix + "attention_context"]
        residual_label = "embeddings" if index == 0 else f"block_{index - 1:02d}"
        residual = torch.from_numpy(captures[residual_label]).to(device, torch.bfloat16)
        weight_prefix = f"visual.encoder.layers.{index}."
        with torch.inference_mode():
            normalized = torch.nn.functional.layer_norm(
                residual,
                (config.vision_config.hidden_size,),
                reference[weight_prefix + "layer_norm1.weight"].to(device, torch.bfloat16),
                reference[weight_prefix + "layer_norm1.bias"].to(device, torch.bfloat16),
                eps=config.vision_config.layer_norm_eps,
            )
            metrics[f"block_{index:02d}/norm1_same_residual"] = _vision_error_metrics(
                normalized.float().cpu().numpy(), captures[prefix + "norm1"]
            )
            normalized = torch.from_numpy(captures[prefix + "norm1"]).to(
                device, torch.bfloat16
            )
            for projection, label in (("q", "query"), ("k", "key"), ("v", "value")):
                stem = weight_prefix + f"self_attn.{projection}_proj."
                weight = reference[stem + "weight"].to(device, torch.bfloat16).T.contiguous()
                bias = reference[stem + "bias"].to(device, torch.bfloat16)
                output = (normalized @ weight) + bias
                metrics[f"block_{index:02d}/{label}_same_normalized"] = _vision_error_metrics(
                    output.float().cpu().numpy(), captures[prefix + label]
                )
        query, key, value = (
            torch.from_numpy(captures[prefix + label])
            .to(device, torch.bfloat16)
            .reshape(1, -1, heads, head_dim)
            .transpose(1, 2)
            for label in ("query", "key", "value")
        )
        for label, backend, dtype in (
            ("flash_bf16", SDPBackend.FLASH_ATTENTION, torch.bfloat16),
            ("efficient_bf16", SDPBackend.EFFICIENT_ATTENTION, torch.bfloat16),
            ("math_bf16", SDPBackend.MATH, torch.bfloat16),
            ("math_fp32", SDPBackend.MATH, torch.float32),
        ):
            name = f"block_{index:02d}/{label}"
            if (
                backend == SDPBackend.FLASH_ATTENTION
                and not torch.backends.cuda.is_flash_attention_available()
            ):
                metrics[name] = {
                    "unavailable": "PyTorch was not compiled with Flash Attention"
                }
                continue
            with torch.inference_mode(), sdpa_kernel(backend):
                output = (
                    torch.nn.functional.scaled_dot_product_attention(
                        query.to(dtype),
                        key.to(dtype),
                        value.to(dtype),
                        scale=float(np.float32(head_dim**-0.5)),
                    )
                    .to(torch.bfloat16)
                    .transpose(1, 2)
                    .reshape(expected.shape)
                )
            metrics[name] = _vision_error_metrics(output.float().cpu().numpy(), expected)
            print(f"Same-QKV attention {name}: {metrics[name]}")
        with torch.inference_mode():
            scale = float(np.float32(head_dim**-0.5))
            for label, scores in (
                ("bf16_scores_then_scale", (query @ key.transpose(-1, -2)) * scale),
                (
                    "scaled_scores_to_bf16",
                    ((query.float() @ key.float().transpose(-1, -2)) * scale).to(
                        torch.bfloat16
                    ),
                ),
            ):
                probabilities = scores.float().softmax(dim=-1).to(torch.bfloat16)
                output = (probabilities @ value).transpose(1, 2).reshape(expected.shape)
                name = f"block_{index:02d}/{label}"
                metrics[name] = _vision_error_metrics(output.float().cpu().numpy(), expected)
                print(f"Same-QKV attention {name}: {metrics[name]}")
    return metrics


def _replay_vision_rounding(
    visual,
    projector,
    packed,
    grid,
    captures,
    device,
    *,
    kernel_arithmetic=False,
    reset_inputs=False,
    reset_attention=False,
    output_captures=None,
    normalized_captures=None,
    projector_arithmetic=False,
):
    """Replay kernel arithmetic, optionally removing upstream error at boundaries."""
    from contextlib import ExitStack
    from functools import partial
    from unittest.mock import patch

    import torch.nn.functional as functional

    metrics = {}

    def separate_linear(module, hidden):
        return functional.linear(hidden, module.weight) + module.bias

    def layout_linear(weight, bias, hidden):
        output = hidden @ weight
        return output if bias is None else output + bias

    def layer_forward(layer, hidden_states, cu_seqlens, **kwargs):
        attention = layer.self_attn(
            hidden_states=layer.layer_norm1(hidden_states), cu_seqlens=cu_seqlens, **kwargs
        )
        hidden_states = (hidden_states + layer.self_attn.out_proj.bias) + attention
        mlp = layer.mlp(layer.layer_norm2(hidden_states))
        return (hidden_states + layer.mlp.fc2.bias) + mlp

    def capture(label, module, inputs, output):
        assert output.dtype == torch.bfloat16 and output.is_cuda
        actual = output.detach().float().cpu().numpy().reshape(captures[label].shape)
        metrics[label] = _vision_error_metrics(actual, captures[label])
        if output_captures is not None:
            output_captures[label] = actual

    def reset_input(label, module, inputs):
        hidden = torch.from_numpy(captures[label]).to(device, torch.bfloat16)
        assert hidden.shape == inputs[0].shape
        return (hidden, *inputs[1:])

    def record(label, module, inputs, output):
        output_captures[label] = (
            output.detach().float().cpu().numpy().reshape(captures[label].shape)
        )

    def record_input(label, module, inputs):
        record(label, module, inputs, inputs[0])

    def replace_normalized(label, module, inputs, output):
        replacement = torch.from_numpy(normalized_captures[label]).to(
            output.device, output.dtype
        )
        return replacement.reshape(output.shape)

    with ExitStack() as stack, torch.inference_mode():
        if kernel_arithmetic:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            stack.enter_context(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))
        if normalized_captures is not None:
            for index, layer in enumerate(visual.encoder.layers):
                for norm, module in ((1, layer.layer_norm1), (2, layer.layer_norm2)):
                    label = f"block_{index:02d}_norm{norm}"
                    if label in normalized_captures:
                        handle = module.register_forward_hook(
                            partial(replace_normalized, label)
                        )
                        stack.callback(handle.remove)
            for label, module in (
                ("post_layernorm", visual.post_layernorm),
                ("projector_norm", projector.norm),
            ):
                if label in normalized_captures:
                    handle = module.register_forward_hook(partial(replace_normalized, label))
                    stack.callback(handle.remove)
        if projector_arithmetic:

            def projector_gelu(hidden):
                """Model _Gelu<BFloat16> in ORT f2c39fe2f cuda/cu_inc/common.cuh."""
                probability = 0.5 * torch.erfc(-hidden.float() * (2.0**-0.5))
                return hidden * probability.to(hidden.dtype)

            stack.enter_context(patch.object(projector.act_fn, "forward", projector_gelu))
            for projection in (projector.linear_fc1, projector.linear_fc2):
                stack.enter_context(
                    patch.object(
                        projection,
                        "forward",
                        partial(
                            torch.addmm, projection.bias, mat2=projection.weight.T.contiguous()
                        ),
                    )
                )
        if output_captures is not None:
            layer = visual.encoder.layers[0]
            for label, module in (
                ("embeddings", visual.embeddings),
                ("block_00_norm1", layer.layer_norm1),
                ("block_00_query", layer.self_attn.q_proj),
                ("block_00_key", layer.self_attn.k_proj),
                ("block_00_value", layer.self_attn.v_proj),
                ("block_00_attention_projection", layer.self_attn.out_proj),
                ("block_00_norm2", layer.layer_norm2),
                ("block_00_mlp_activation", layer.mlp.activation_fn),
                ("block_00_mlp_down", layer.mlp.fc2),
            ):
                handle = module.register_forward_hook(partial(record, label))
                stack.callback(handle.remove)
            for label, module in (
                ("block_00_attention_context", layer.self_attn.out_proj),
                ("block_00_attention_residual", layer.layer_norm2),
            ):
                handle = module.register_forward_pre_hook(partial(record_input, label))
                stack.callback(handle.remove)
            for label, module in (
                ("projector_norm", projector.norm),
                ("projector_up", projector.linear_fc1),
                ("projector_activation", projector.act_fn),
            ):
                if label in captures:
                    handle = module.register_forward_hook(partial(record, label))
                    stack.callback(handle.remove)
        for index, layer in enumerate(visual.encoder.layers):
            attention_label = f"block_{index:02d}_attention_context"
            if reset_attention and attention_label in captures:
                handle = layer.self_attn.out_proj.register_forward_pre_hook(
                    partial(reset_input, attention_label)
                )
                stack.callback(handle.remove)
            if reset_inputs:
                label = "embeddings" if index == 0 else f"block_{index - 1:02d}"
                handle = layer.register_forward_pre_hook(partial(reset_input, label))
                stack.callback(handle.remove)
            for name in ("q_proj", "k_proj", "v_proj"):
                projection = getattr(layer.self_attn, name)
                stack.enter_context(
                    patch.object(projection, "forward", partial(separate_linear, projection))
                )
            if kernel_arithmetic:
                projections = (
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                    layer.self_attn.out_proj,
                    layer.mlp.fc1,
                    layer.mlp.fc2,
                )
                for projection in projections:
                    bias = (
                        None
                        if projection in (layer.self_attn.out_proj, layer.mlp.fc2)
                        else projection.bias
                    )
                    stack.enter_context(
                        patch.object(
                            projection,
                            "forward",
                            partial(layout_linear, projection.weight.T.contiguous(), bias),
                        )
                    )
                stack.enter_context(
                    patch.object(layer.mlp.activation_fn, "forward", _bf16_fast_gelu)
                )
                stack.enter_context(
                    patch.object(layer, "forward", partial(layer_forward, layer))
                )
            handle = layer.register_forward_hook(partial(capture, f"block_{index:02d}"))
            stack.callback(handle.remove)
        for label, module in (
            ("post_layernorm", visual.post_layernorm),
            ("projector", projector),
        ):
            if reset_inputs:
                source = (
                    f"block_{len(visual.encoder.layers) - 1:02d}"
                    if label == "post_layernorm"
                    else "post_layernorm"
                )
                handle = module.register_forward_pre_hook(partial(reset_input, source))
                stack.callback(handle.remove)
            handle = module.register_forward_hook(partial(capture, label))
            stack.callback(handle.remove)
        features = visual(
            packed[0].to(device=device, dtype=torch.bfloat16),
            torch.from_numpy(grid.reshape(1, 3)).to(device),
        ).last_hidden_state
        projector(features)
    assert len(metrics) == len(visual.encoder.layers) + 2
    for label, values in metrics.items():
        mode = "Encoder kernel arithmetic" if kernel_arithmetic else "Separate-QKV rounding"
        if reset_inputs:
            mode += " same-input"
        if reset_attention:
            mode += " exact-attention"
        print(f"{mode} {label}: {values}")
    return metrics


def test_bf16_vision_pytorch_matches_saved_export():
    """Compare saved CUDA output with upstream PyTorch BF16, without rebuilding."""
    _check_bf16_vision_pytorch()


def _compare_ort_same_qkv(capture_dir, capfd):
    """Compare isolated ORT kernels to FP64 math on the exact captured BF16 Q/K/V."""
    import ml_dtypes
    import onnx_ir as ir
    import onnxruntime as ort
    from torch.nn.attention import SDPBackend, sdpa_kernel

    manifest = json.loads((capture_dir / "checkpoints.json").read_text(encoding="utf-8"))
    model = ir.load(manifest["source"])
    attention = next(node for node in model.graph if node.op_type == "Attention")
    assert attention.domain == "" and all(value is None for value in attention.inputs[3:])
    assert (
        "is_causal" not in attention.attributes
        or attention.attributes["is_causal"].as_int() == 0
    )
    assert (
        "softcap" not in attention.attributes
        or attention.attributes["softcap"].as_float() == 0
    )
    assert (
        attention.attributes["q_num_heads"].as_int()
        == attention.attributes["kv_num_heads"].as_int()
    )
    shape = tuple(manifest["checkpoints"]["block_00_query"]["onnx_shape"])
    inputs = [
        ir.Value(name=name, type=ir.TensorType(ir.DataType.BFLOAT16), shape=ir.Shape(shape))
        for name in ("query", "key", "value")
    ]
    output = ir.Value(
        name="context", type=ir.TensorType(ir.DataType.BFLOAT16), shape=ir.Shape(shape)
    )
    node = ir.Node(
        "", "Attention", inputs, attributes=attention.attributes.values(), outputs=[output]
    )
    graph = ir.Graph(inputs, [output], nodes=[node], opset_imports=model.graph.opset_imports)
    path = capture_dir / "isolated_attention.onnx"
    ir.save(ir.Model(graph, ir_version=model.ir_version), path)
    heads = attention.attributes["q_num_heads"].as_int()
    scale = attention.attributes["scale"].as_float()
    del attention, model
    with np.load(capture_dir / "onnx_checkpoints.npz") as arrays:
        captures = dict(arrays)
    prefixes = sorted(
        label.removesuffix("attention_context")
        for label in captures
        if label.endswith("attention_context")
    )
    report = {}
    for prefix in prefixes:
        arrays = {
            name: np.ascontiguousarray(
                captures[prefix + name].reshape(shape), dtype=ml_dtypes.bfloat16
            )
            for name in ("query", "key", "value")
        }
        query, key, value = (
            torch.from_numpy(array.astype(np.float32))
            .double()
            .reshape(shape[0], shape[1], heads, shape[2] // heads)
            .transpose(1, 2)
            for array in arrays.values()
        )
        with torch.inference_mode():
            oracle = ((query @ key.transpose(-1, -2)) * scale).softmax(-1) @ value
            rounded = (
                oracle.to(torch.bfloat16)
                .transpose(1, 2)
                .reshape(captures[prefix + "attention_context"].shape)
                .float()
                .numpy()
            )
            oracle = oracle.transpose(1, 2).reshape(rounded.shape).numpy()
        results = {}
        for mode, selector, expected_kernel in (
            ("flash", 1, "FLASH_ATTENTION"),
            ("efficient", 2, "EFFICIENT_ATTENTION"),
            ("math", 16, "MATH"),
        ):
            capfd.readouterr()
            session = ort.InferenceSession(
                str(path),
                providers=[("CUDAExecutionProvider", {"sdpa_kernel": str(selector)})],
            )
            assert session.get_providers()[0] == "CUDAExecutionProvider"
            binding = session.io_binding()
            values = {
                name: ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                    array, int(ir.DataType.BFLOAT16)
                )
                for name, array in arrays.items()
            }
            for name, tensor in values.items():
                binding.bind_ortvalue_input(name, tensor)
            actual = np.empty(shape, dtype=ml_dtypes.bfloat16)
            binding.bind_output(
                "context",
                device_type="cpu",
                device_id=0,
                element_type=int(ir.DataType.BFLOAT16),
                shape=shape,
                buffer_ptr=actual.ctypes.data,
            )
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            actual = actual.astype(np.float32).reshape(rounded.shape)
            del binding, session
            captured = capfd.readouterr()
            dispatch = [line for line in captured.out.splitlines() if "SdpaKernel=" in line]
            assert (
                len(dispatch) == 1
                and dispatch[0].partition("SdpaKernel=")[2].split("\x1b", 1)[0].strip()
                == expected_kernel
            ), captured.out + captured.err
            if mode == "flash":
                np.testing.assert_array_equal(actual, captures[prefix + "attention_context"])
            results[f"ort_{mode}"] = actual
        for mode, backend in (
            ("efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("math", SDPBackend.MATH),
        ):
            with torch.inference_mode(), sdpa_kernel(backend):
                actual = (
                    torch.nn.functional.scaled_dot_product_attention(
                        query.to("cuda", torch.bfloat16),
                        key.to("cuda", torch.bfloat16),
                        value.to("cuda", torch.bfloat16),
                        scale=scale,
                    )
                    .transpose(1, 2)
                    .reshape(rounded.shape)
                )
            results[f"torch_{mode}"] = actual.float().cpu().numpy()
        report[prefix.removesuffix("_")] = {
            mode: {
                "vs_fp64": _vision_error_metrics(actual.astype(np.float64), oracle),
                "vs_rounded_fp64": _vision_error_metrics(actual, rounded),
                "vs_torch_efficient": _vision_error_metrics(
                    actual, results["torch_efficient"]
                ),
            }
            for mode, actual in results.items()
        }
    (capture_dir / "same_qkv_kernel_accuracy.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for label, results in report.items():
        for mode, metrics in results.items():
            print(f"Same-QKV {label}/{mode} vs rounded FP64: {metrics['vs_rounded_fp64']}")


def _compare_ort_attention_dispatch(capture_dir, capfd):
    """Change only ORT attention dispatch; verify actual kernels and capture integrity."""
    import onnxruntime as ort

    manifest = json.loads((capture_dir / "checkpoints.json").read_text(encoding="utf-8"))
    checkpoints = manifest["checkpoints"]
    shapes = {item["onnx_value"]: tuple(item["onnx_shape"]) for item in checkpoints.values()}
    with np.load(capture_dir / "inputs.npz") as inputs:
        pixels, grid = inputs["pixel_values"], inputs["grid_thw"]
    with np.load(capture_dir / "onnx_checkpoints.npz") as inputs:
        baseline = dict(inputs)
    with np.load(capture_dir / "torch_checkpoints.npz") as inputs:
        reference = dict(inputs)
    with np.load(capture_dir / "encoder_kernel_checkpoints.npz") as inputs:
        kernel_reference = dict(inputs)
    with np.load(capture_dir / "norm1_reset_checkpoints.npz") as inputs:
        norm1_reference = dict(inputs)
    with np.load(capture_dir / "both_norms_reset_checkpoints.npz") as inputs:
        both_norms_reference = dict(inputs)
    layer_count = sum(label.removeprefix("block_").isdigit() for label in checkpoints)
    report = {}
    for mode, selector, expected_kernel in (
        ("flash", 1, "FLASH_ATTENTION"),
        ("efficient", 2, "EFFICIENT_ATTENTION"),
        ("math", 16, "MATH"),
    ):
        providers = [("CUDAExecutionProvider", {"sdpa_kernel": str(selector)})]
        capfd.readouterr()
        session = ort.InferenceSession(
            str(capture_dir / "vision_checkpoints.onnx"), providers=providers
        )
        assert session.get_provider_options()["CUDAExecutionProvider"]["sdpa_kernel"] == str(
            selector
        )
        outputs = _run_bf16_vision_outputs(session, pixels, grid, shapes)
        del session
        actual = {
            label: outputs[item["onnx_value"]].reshape(baseline[label].shape)
            for label, item in checkpoints.items()
        }
        session = ort.InferenceSession(manifest["source"], providers=providers)
        original = _run_bf16_vision(session, pixels, grid, baseline["projector"].shape)
        del session
        np.testing.assert_array_equal(actual["projector"], original)
        captured = capfd.readouterr()
        dispatch = [line for line in captured.out.splitlines() if "SdpaKernel=" in line]
        assert len(dispatch) == 2 * layer_count, captured.out + captured.err
        assert all(
            line.partition("SdpaKernel=")[2].split("\x1b", 1)[0].strip() == expected_kernel
            for line in dispatch
        ), dispatch
        (capture_dir / f"attention_dispatch_{mode}.log").write_text(
            "\n".join(dispatch), encoding="utf-8"
        )
        for label in ("block_00_query", "block_00_key", "block_00_value"):
            np.testing.assert_array_equal(actual[label], baseline[label])
        if mode == "flash":
            for label in checkpoints:
                np.testing.assert_array_equal(actual[label], baseline[label])
        if mode == "efficient":
            for label in (
                "embeddings",
                "block_00_norm1",
                "block_00_query",
                "block_00_key",
                "block_00_value",
                "block_00_attention_context",
                "block_00_attention_projection",
                "block_00_attention_residual",
            ):
                np.testing.assert_array_equal(
                    actual[label], norm1_reference[label], err_msg=label
                )
            for label, expected in both_norms_reference.items():
                if label == "embeddings" or label.startswith("block_00"):
                    np.testing.assert_array_equal(actual[label], expected, err_msg=label)
        report[mode] = {
            "sdpa_kernel": selector,
            "observed_kernel": expected_kernel,
            "vs_upstream": {
                label: _vision_error_metrics(output, reference[label])
                for label, output in actual.items()
            },
            "vs_default": {
                label: _vision_error_metrics(output, baseline[label])
                for label, output in actual.items()
            },
            "vs_kernel_replay": {
                label: _vision_error_metrics(actual[label], expected)
                for label, expected in kernel_reference.items()
            },
            "vs_norm1_reset": {
                label: _vision_error_metrics(actual[label], expected)
                for label, expected in norm1_reference.items()
            },
            "vs_both_norms_reset": {
                label: _vision_error_metrics(actual[label], expected)
                for label, expected in both_norms_reference.items()
            },
        }
        np.savez(capture_dir / f"ort_{mode}_checkpoints.npz", **actual)
    (capture_dir / "ort_attention_dispatch.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for mode, result in report.items():
        print(f"ORT {mode} projector vs upstream: {result['vs_upstream']['projector']}")
        print(
            f"ORT {mode} projector vs kernel replay: {result['vs_kernel_replay']['projector']}"
        )


def test_skip_layer_norm_probe(tmp_path):
    """Collect evidence; passing does not assert fused normalization accuracy."""
    _run_skip_layer_norm_probe(tmp_path)


def test_skip_layer_norm_near_constant_accuracy(tmp_path):
    """Require fused BF16 normalization to match the rounded FP64 stress oracle."""
    _run_skip_layer_norm_probe(tmp_path)
    with np.load(tmp_path / "near_constant_normalization_outputs.npz") as arrays:
        actual = arrays["skip_bf16"]
        expected = torch.from_numpy(arrays["oracle"]).bfloat16().float().numpy()
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-3,
        rtol=1e-2,
        err_msg=(
            "CUDA SkipLayerNormalization fails the near-constant variance accuracy check. "
            "This reproduces the kernel defect, not the full Cosmos parity failure."
        ),
    )


def _run_skip_layer_norm_probe(tmp_path):
    """Isolate CUDA normalization from residual addition using a captured real input."""
    import ml_dtypes
    import onnx_ir as ir
    import onnxruntime as ort

    capture_path = os.environ.get("COSMOS3_EDGE_CAPTURE_DIR")
    if not capture_path:
        pytest.skip(
            "Set COSMOS3_EDGE_CAPTURE_DIR to a completed bf16_vision_capture directory"
        )
    capture_dir = Path(capture_path)
    manifest = json.loads((capture_dir / "checkpoints.json").read_text(encoding="utf-8"))
    model = ir.load(manifest["source"])
    target = manifest["checkpoints"]["block_00_norm1"]["onnx_value"]
    source = next(
        node for node in model.graph if any(value.name == target for value in node.outputs)
    )
    epsilon = source.attributes["epsilon"].as_float()
    gamma, beta = (
        value.const_value.numpy().astype(np.float32) for value in source.inputs[2:4]
    )
    with np.load(capture_dir / "onnx_checkpoints.npz") as arrays:
        residual = arrays["embeddings"].reshape(1, -1, gamma.size)
        captured = arrays["block_00_norm1"].reshape(residual.shape)
    del source, model
    ort.preload_dlls()
    constant_offset = np.full((1, 1, gamma.size), 128, dtype=np.float32)
    constant_offset[..., 0] = 129
    report = {}
    for case, hidden, weight, bias in (
        ("captured", residual, gamma, beta),
        ("captured_unit_affine", residual, np.ones_like(gamma), np.zeros_like(beta)),
        ("near_constant", constant_offset, np.ones_like(gamma), np.zeros_like(beta)),
        (
            "near_constant_centered",
            constant_offset - 128,
            np.ones_like(gamma),
            np.zeros_like(beta),
        ),
    ):
        precise = torch.from_numpy(hidden).double()
        with torch.inference_mode():
            variance, mean = torch.var_mean(precise, dim=-1, correction=0, keepdim=True)
            oracle = (precise - mean) * torch.rsqrt(variance + epsilon)
            oracle = (
                oracle * torch.from_numpy(weight).double() + torch.from_numpy(bias).double()
            )
            rounded = oracle.to(torch.bfloat16).float().numpy()
            torch_output = (
                torch.nn.functional.layer_norm(
                    torch.from_numpy(hidden).cuda().bfloat16(),
                    (gamma.size,),
                    torch.from_numpy(weight).cuda().bfloat16(),
                    torch.from_numpy(bias).cuda().bfloat16(),
                    epsilon,
                )
                .float()
                .cpu()
                .numpy()
            )
        outputs = {"torch": torch_output}
        if case == "captured":
            with torch.inference_mode():
                hidden_cuda = torch.from_numpy(hidden).cuda().bfloat16()
                gamma_cuda = torch.from_numpy(weight).cuda().bfloat16()
                beta_cuda = torch.from_numpy(bias).cuda().bfloat16()
                native, native_mean, native_rstd = torch.native_layer_norm(
                    hidden_cuda, (gamma.size,), gamma_cuda, beta_cuda, epsilon
                )
                np.testing.assert_array_equal(native.float().cpu().numpy(), torch_output)
                centered_cuda = hidden_cuda.float() - native_mean
                products = {
                    "torch_stats_rstd_first": (
                        centered_cuda * native_rstd,
                        gamma_cuda.float(),
                    ),
                    "torch_stats_gamma_first": (
                        centered_cuda * gamma_cuda.float(),
                        native_rstd,
                    ),
                }
                for label, (left, right) in products.items():
                    fused_affine = (
                        left.double() * right.double() + beta_cuda.double()
                    ).float()
                    outputs[label] = fused_affine.bfloat16().float().cpu().numpy()
            np.savez(
                tmp_path / "torch_normalization_statistics.npz",
                mean=native_mean.cpu().numpy(),
                rstd=native_rstd.cpu().numpy(),
            )
        for mode, dtype, operation in (
            ("skip_bf16", ir.DataType.BFLOAT16, "SkipLayerNormalization"),
            ("plain_bf16", ir.DataType.BFLOAT16, "LayerNormalization"),
            ("skip_fp32", ir.DataType.FLOAT, "SkipLayerNormalization"),
            ("plain_fp32", ir.DataType.FLOAT, "LayerNormalization"),
        ):
            numpy_dtype = ml_dtypes.bfloat16 if dtype == ir.DataType.BFLOAT16 else np.float32
            values = [
                ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))
                for name, shape in (
                    ("hidden", hidden.shape),
                    ("skip", hidden.shape),
                    ("weight", weight.shape),
                    ("bias", bias.shape),
                )
            ]
            norm = ir.Node(
                "com.microsoft" if operation == "SkipLayerNormalization" else "",
                operation,
                values
                if operation == "SkipLayerNormalization"
                else [values[0], values[2], values[3]],
                {"epsilon": ir.AttrFloat32("epsilon", epsilon)},
            )
            norm.outputs[0].name = "normalized"
            norm.outputs[0].type = ir.TensorType(dtype)
            norm.outputs[0].shape = ir.Shape(hidden.shape)
            graph = ir.Graph(
                values, norm.outputs, nodes=[norm], opset_imports={"": 24, "com.microsoft": 1}
            )
            path = tmp_path / f"{case}_{mode}.onnx"
            ir.save(ir.Model(graph, ir_version=10), path)
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CUDAExecutionProvider"]
            )
            assert session.get_providers()[0] == "CUDAExecutionProvider"
            binding = session.io_binding()
            feeds = {
                name: np.ascontiguousarray(array, dtype=numpy_dtype)
                for name, array in (
                    ("hidden", hidden),
                    ("skip", np.zeros_like(hidden)),
                    ("weight", weight),
                    ("bias", bias),
                )
            }
            tensors = {
                name: ort.OrtValue.ortvalue_from_numpy_with_onnx_type(array, int(dtype))
                for name, array in feeds.items()
            }
            for name, tensor in tensors.items():
                binding.bind_ortvalue_input(name, tensor)
            output = np.empty(hidden.shape, dtype=numpy_dtype)
            binding.bind_output(
                "normalized",
                device_type="cpu",
                device_id=0,
                element_type=int(dtype),
                shape=hidden.shape,
                buffer_ptr=output.ctypes.data,
            )
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            outputs[mode] = output.astype(np.float32)
            del binding, session
        np.savez(
            tmp_path / f"{case}_normalization_outputs.npz",
            **outputs,
            oracle=oracle.numpy(),
            hidden=hidden,
            weight=weight,
            bias=bias,
        )
        if case == "captured":
            np.testing.assert_array_equal(outputs["skip_bf16"], captured)
        if case == "near_constant":
            np.testing.assert_allclose(outputs["plain_bf16"], rounded, atol=1e-3, rtol=1e-2)
            np.testing.assert_array_equal(outputs["plain_bf16"], outputs["torch"])
            report["stress_values"] = {
                mode: actual.reshape(-1)[:2].tolist() for mode, actual in outputs.items()
            }
            report["stress_values"]["fp64"] = oracle.numpy().reshape(-1)[:2].tolist()
        report[case] = {
            mode: {
                "vs_fp64": _vision_error_metrics(actual.astype(np.float64), oracle.numpy()),
                "rounded_vs_oracle": _vision_error_metrics(
                    torch.from_numpy(actual).bfloat16().float().numpy(), rounded
                ),
                "vs_torch": _vision_error_metrics(actual, torch_output),
            }
            for mode, actual in outputs.items()
        }
        report[case]["conditioning"] = {
            "min_variance": float(variance.min()),
            "max_variance": float(variance.max()),
            "max_second_moment_over_variance": float(
                (precise.square().mean(-1, keepdim=True) / variance).max()
            ),
        }
        if case in ("captured_unit_affine", "near_constant", "near_constant_centered"):
            centered = precise.numpy() - mean.numpy()
            normalized = outputs["skip_fp32"].astype(np.float64)
            slope = (centered * normalized).sum(-1, keepdims=True) / (centered * centered).sum(
                -1, keepdims=True
            )
            inferred_mean = mean.numpy() - normalized.mean(-1, keepdims=True) / slope
            inferred_variance = 1 / slope**2 - epsilon
            report[case]["inferred_skip_statistics"] = {
                "max_mean_error": float(np.abs(inferred_mean - mean.numpy()).max()),
                "max_variance_relative_error": float(
                    (np.abs(inferred_variance - variance.numpy()) / variance.numpy()).max()
                ),
                "min_inferred_variance": float(inferred_variance.min()),
            }
        if case == "near_constant_centered":
            np.testing.assert_allclose(outputs["skip_bf16"], rounded, atol=1e-3, rtol=1e-3)
            with np.load(tmp_path / "near_constant_normalization_outputs.npz") as original:
                report[case]["translation_invariance"] = {
                    mode: _vision_error_metrics(outputs[mode], original[mode])
                    for mode in outputs
                }
    (tmp_path / "skip_layer_norm_probe.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    stress = report["stress_values"]
    print(
        "Diagnostic only (fused stress accuracy is NOT asserted here): "
        f"SkipLayerNorm BF16 first={stress['skip_bf16'][0]}, "
        f"standard LayerNorm BF16 first={stress['plain_bf16'][0]}, "
        f"FP64 first={stress['fp64'][0]:.8f}"
    )
    print(f"SkipLayerNorm probe artifacts: {tmp_path}")


def test_skip_layer_norm_model_control(tmp_path):
    """Compare fused and decomposed normalization without changing BF16 tensor precision."""
    import onnx_ir as ir
    import onnxruntime as ort

    capture_path = os.environ.get("COSMOS3_EDGE_CAPTURE_DIR")
    if not capture_path:
        pytest.skip(
            "Set COSMOS3_EDGE_CAPTURE_DIR to a completed bf16_vision_capture directory"
        )
    capture_dir = Path(capture_path)
    manifest = json.loads((capture_dir / "checkpoints.json").read_text(encoding="utf-8"))
    with np.load(capture_dir / "inputs.npz") as arrays:
        pixels, grid = arrays["pixel_values"], arrays["grid_thw"]
    with np.load(capture_dir / "torch_checkpoints.npz") as arrays:
        reference = arrays["projector"]
    fp32 = np.load(capture_dir / "torch_fp32_reference.npy")
    model = ir.load(manifest["source"])
    replaced = 0
    for node in list(model.graph):
        if (node.domain, node.op_type) != ("com.microsoft", "SkipLayerNormalization"):
            continue
        assert all(
            not value.uses() and value not in model.graph.outputs
            for value in node.outputs[1:3]
        )
        nodes = []
        hidden = node.inputs[0]
        if len(node.inputs) > 4 and node.inputs[4] is not None:
            bias_add = ir.Node("", "Add", [hidden, node.inputs[4]])
            nodes.append(bias_add)
            hidden = bias_add.outputs[0]
        addition = ir.Node("", "Add", [hidden, node.inputs[1]])
        normalized = ir.Node(
            "",
            "LayerNormalization",
            [addition.outputs[0], *node.inputs[2:4]],
            {
                "epsilon": ir.AttrFloat32("epsilon", node.attributes["epsilon"].as_float()),
                "axis": ir.AttrInt64("axis", -1),
            },
        )
        nodes.extend((addition, normalized))
        for replacement in nodes:
            replacement.outputs[0].type = node.outputs[0].type
            replacement.outputs[0].shape = node.outputs[0].shape
        normalized.outputs[0].name = node.outputs[0].name
        old_values, new_values = [node.outputs[0]], [normalized.outputs[0]]
        if len(node.outputs) > 3:
            addition.outputs[0].name = node.outputs[3].name
            old_values.append(node.outputs[3])
            new_values.append(addition.outputs[0])
        ir.convenience.replace_nodes_and_values(
            model.graph, node, [node], nodes, old_values, new_values
        )
        replaced += 1
    assert replaced == 55
    path = tmp_path / "decomposed_skip_layer_norm.onnx"
    ir.save(model, path, external_data="decomposed_skip_layer_norm.onnx.data")
    del model
    ort.preload_dlls()
    report = {}
    for mode, graph_path, disabled in (
        ("original", manifest["source"], []),
        (
            "fused_control",
            manifest["source"],
            ["SkipLayerNormFusion", "BiasSkipLayerNormFusion"],
        ),
        ("decomposed", path, ["SkipLayerNormFusion", "BiasSkipLayerNormFusion"]),
    ):
        options = ort.SessionOptions()
        optimized_path = tmp_path / "optimized.onnx"
        options.optimized_model_filepath = str(optimized_path)
        session = ort.InferenceSession(
            str(graph_path),
            sess_options=options,
            providers=["CUDAExecutionProvider"],
            disabled_optimizers=disabled,
        )
        output = _run_bf16_vision(session, pixels, grid, reference.shape)
        del session
        optimized = ir.load(optimized_path)
        fused_count = sum(node.op_type == "SkipLayerNormalization" for node in optimized.graph)
        assert fused_count == (0 if mode == "decomposed" else replaced)
        del optimized
        report[mode] = {
            "fused_count": fused_count,
            "vs_upstream_bf16": _vision_error_metrics(output, reference),
            "vs_upstream_fp32": _vision_error_metrics(output, fp32),
        }
        np.save(tmp_path / f"{mode}_output.npy", output)
    (tmp_path / "skip_layer_norm_model_control.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"SkipLayerNorm model-control artifacts: {tmp_path}")


def _compare_first_block_norms(capture_dir):
    """Measure same-residual LayerNorm accuracy without replaying CUDA reduction order."""
    import onnx_ir as ir

    manifest = json.loads((capture_dir / "checkpoints.json").read_text(encoding="utf-8"))
    model = ir.load(manifest["source"])
    with np.load(capture_dir / "ort_efficient_checkpoints.npz") as arrays:
        captures = dict(arrays)
    with np.load(capture_dir / "encoder_kernel_checkpoints.npz") as arrays:
        norm1_replay = dict(arrays)
    with np.load(capture_dir / "norm1_reset_checkpoints.npz") as arrays:
        norm2_replay = dict(arrays)
    report = {}
    for label, residual_label, replay in (
        ("block_00_norm1", "embeddings", norm1_replay),
        ("block_00_norm2", "block_00_attention_residual", norm2_replay),
    ):
        name = manifest["checkpoints"][label]["onnx_value"]
        node = next(
            node for node in model.graph if any(value.name == name for value in node.outputs)
        )
        assert node.op_type == "SkipLayerNormalization"
        epsilon = node.attributes["epsilon"].as_float()
        weight, bias = (
            torch.from_numpy(value.const_value.numpy().astype(np.float64))
            for value in node.inputs[2:4]
        )
        hidden = torch.from_numpy(captures[residual_label]).double()
        with torch.inference_mode():
            mean = hidden.mean(-1, keepdim=True)
            variance = (hidden - mean).square().mean(-1, keepdim=True)
            oracle = (hidden - mean) * torch.rsqrt(variance + epsilon) * weight + bias
            rounded = oracle.to(torch.bfloat16).float().numpy()
            hidden = hidden.to("cuda", torch.bfloat16)
            weight, bias = weight.to("cuda", torch.bfloat16), bias.to("cuda", torch.bfloat16)
            actual = torch.nn.functional.layer_norm(
                hidden, (hidden.shape[-1],), weight, bias, epsilon
            )
            np.testing.assert_array_equal(actual.float().cpu().numpy(), replay[label])
            outputs = {"ort": captures[label], "torch_bf16": actual.float().cpu().numpy()}
            hidden, weight, bias = hidden.float(), weight.float(), bias.float()
            outputs["torch_fp32_cast"] = (
                torch.nn.functional.layer_norm(
                    hidden,
                    (hidden.shape[-1],),
                    weight,
                    bias,
                    epsilon,
                )
                .to(torch.bfloat16)
                .float()
                .cpu()
                .numpy()
            )
            mean = hidden.mean(-1, keepdim=True)
            moments_variance = hidden.square().mean(-1, keepdim=True) - mean.square()
            stable_variance, stable_mean = torch.var_mean(
                hidden, dim=-1, correction=0, keepdim=True
            )
            for mode, variant_mean, variant_variance in (
                ("fp32_moments_gamma_first", mean, moments_variance),
                ("fp32_varmean_gamma_first", stable_mean, stable_variance),
            ):
                outputs[mode] = (
                    (
                        weight
                        * (hidden - variant_mean)
                        * torch.rsqrt(variant_variance + epsilon)
                        + bias
                    )
                    .to(torch.bfloat16)
                    .float()
                    .cpu()
                    .numpy()
                )
        mismatch = outputs["ort"] != outputs["torch_bf16"]
        lower = np.minimum(outputs["ort"], outputs["torch_bf16"])[mismatch].astype(np.float64)
        upper = np.maximum(outputs["ort"], outputs["torch_bf16"])[mismatch].astype(np.float64)
        distance = np.abs(oracle.numpy()[mismatch] - (lower + upper) / 2) / (upper - lower)
        adjacent = (
            torch.nextafter(
                torch.from_numpy(lower).to(torch.bfloat16),
                torch.full(lower.shape, float("inf"), dtype=torch.bfloat16),
            )
            .double()
            .numpy()
            == upper
        )
        report[label] = {
            "mismatch_count": int(mismatch.sum()),
            "total_elements": int(mismatch.size),
            "adjacent_bf16_disagreements": int(adjacent.sum()),
            "neither_matches_rounded_fp64_at_disagreement": int(
                np.count_nonzero(
                    (outputs["ort"] != rounded) & (outputs["torch_bf16"] != rounded) & mismatch
                )
            ),
            "ort_matches_rounded_fp64_at_disagreement": int(
                np.count_nonzero((outputs["ort"] == rounded) & mismatch)
            ),
            "torch_matches_rounded_fp64_at_disagreement": int(
                np.count_nonzero((outputs["torch_bf16"] == rounded) & mismatch)
            ),
            "max_midpoint_distance_in_disagreement_gaps": float(distance.max())
            if distance.size
            else 0.0,
            "arithmetic": {
                mode: {
                    "vs_rounded_fp64": _vision_error_metrics(output, rounded),
                    "vs_ort": _vision_error_metrics(output, outputs["ort"]),
                }
                for mode, output in outputs.items()
            },
        }
    (capture_dir / "first_block_normalization.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for label, metrics in report.items():
        print(f"Same-input normalization {label}: {metrics}")


def _compare_vision_activation_precision(capture_dir):
    """Change activation intermediates only, with no captured tensors in inference."""
    import onnx_ir as ir
    import onnxruntime as ort

    manifest = json.loads((capture_dir / "checkpoints.json").read_text(encoding="utf-8"))
    with np.load(capture_dir / "inputs.npz") as arrays:
        pixels, grid = arrays["pixel_values"], arrays["grid_thw"]
    with np.load(capture_dir / "torch_checkpoints.npz") as arrays:
        reference = arrays["projector"]
    with np.load(capture_dir / "onnx_checkpoints.npz") as arrays:
        baseline = arrays["projector"]
    report = {"baseline": _vision_error_metrics(baseline, reference)}
    candidate_path = capture_dir / "activation_precision.onnx"
    optimized_path = capture_dir / "activation_precision_optimized.onnx"
    for mode in ("projector", "encoder", "all"):
        model = ir.load(manifest["source"])
        changed = 0
        for node in list(model.graph):
            if node.op_type != "Gelu" or node.domain != "":
                continue
            is_projector = "projector" in node.outputs[0].name
            if (mode == "projector" and not is_projector) or (
                mode == "encoder" and is_projector
            ):
                continue
            assert node.inputs[0].dtype == ir.DataType.BFLOAT16
            stem = node.outputs[0].name
            cast = ir.Node(
                "",
                "Cast",
                [node.inputs[0]],
                {"to": ir.AttrInt64("to", int(ir.DataType.FLOAT))},
            )
            activation = ir.Node("", "Gelu", [cast.outputs[0]], node.attributes.values())
            downcast = ir.Node(
                "",
                "Cast",
                [activation.outputs[0]],
                {"to": ir.AttrInt64("to", int(ir.DataType.BFLOAT16))},
            )
            for value, name, dtype in (
                (cast.outputs[0], stem + "_fp32_input", ir.DataType.FLOAT),
                (activation.outputs[0], stem + "_fp32", ir.DataType.FLOAT),
                (downcast.outputs[0], stem, ir.DataType.BFLOAT16),
            ):
                value.name = name
                value.type = ir.TensorType(dtype)
                value.shape = node.outputs[0].shape
            ir.convenience.replace_nodes_and_values(
                model.graph,
                node,
                [node],
                [cast, activation, downcast],
                [node.outputs[0]],
                [downcast.outputs[0]],
            )
            changed += 1
        expected_count = {"projector": 1, "encoder": 27, "all": 28}[mode]
        assert changed == expected_count
        ir.save(model, candidate_path, external_data="activation_precision.onnx.data")
        del model
        options = ort.SessionOptions()
        options.optimized_model_filepath = str(optimized_path)
        session = ort.InferenceSession(
            str(candidate_path), sess_options=options, providers=["CUDAExecutionProvider"]
        )
        output = _run_bf16_vision(session, pixels, grid, reference.shape)
        del session
        optimized = ir.load(optimized_path)
        float_activations = sum(
            node.op_type in ("Gelu", "FastGelu") and node.inputs[0].dtype == ir.DataType.FLOAT
            for node in optimized.graph
        )
        assert float_activations == changed, (
            "ORT did not retain the requested activation precision"
        )
        del optimized
        report[mode] = {
            "fp32_activations": changed,
            "vs_upstream": _vision_error_metrics(output, reference),
            "vs_baseline": _vision_error_metrics(output, baseline),
        }
        print(f"FP32 {mode} activation intermediates: {report[mode]}")
    (capture_dir / "activation_precision.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def _compare_saved_graph_fp32(capture_dir):
    """Test the saved graph itself at FP32 using the same rounded inputs and weights."""
    import onnx_ir as ir
    import onnxruntime as ort

    manifest = json.loads((capture_dir / "checkpoints.json").read_text(encoding="utf-8"))
    model = ir.load(manifest["source"])
    values = set(model.graph.inputs) | set(model.graph.initializers.values())
    for node in model.graph:
        values.update(node.outputs)
        if node.op_type == "Cast" and node.attributes["to"].as_int() == int(
            ir.DataType.BFLOAT16
        ):
            node.attributes["to"] = ir.AttrInt64("to", int(ir.DataType.FLOAT))
        assert not any(
            attr.type == ir.AttributeType.TENSOR
            and attr.as_tensor().dtype == ir.DataType.BFLOAT16
            for attr in node.attributes.values()
        ), "BF16 tensor attributes require explicit promotion"
    promoted_weights = 0
    for value in values:
        if value.dtype != ir.DataType.BFLOAT16:
            continue
        if value.const_value is not None:
            value.const_value = ir.Tensor(value.const_value.numpy().astype(np.float32))
            promoted_weights += 1
        value.type = ir.TensorType(ir.DataType.FLOAT)
    assert promoted_weights > 0
    assert all(value.dtype != ir.DataType.BFLOAT16 for value in values)
    path = capture_dir / "saved_graph_fp32.onnx"
    ir.save(model, path, external_data="saved_graph_fp32.onnx.data")
    del model, values
    with np.load(capture_dir / "inputs.npz") as arrays:
        pixels = torch.from_numpy(arrays["pixel_values"]).to(torch.bfloat16).float().numpy()
        grid = arrays["grid_thw"]
    expected = np.load(capture_dir / "torch_fp32_reference.npy")
    session = ort.InferenceSession(
        str(path),
        providers=[("CUDAExecutionProvider", {"use_tf32": "0", "sdpa_kernel": "16"})],
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    actual = session.run(None, {"pixel_values": pixels, "grid_thw": grid})[0]
    metrics = _vision_error_metrics(actual, expected)
    print(f"Saved graph promoted to FP32 vs upstream FP32: {metrics}")
    (capture_dir / "saved_graph_fp32_comparison.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.save(capture_dir / "saved_graph_fp32_output.npy", actual)
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
    with np.load(capture_dir / "gradient_precision_reference.npz") as arrays:
        gradient_pixels = arrays["pixel_values"]
        gradient_fp32 = arrays["fp32"]
        gradient_bf16 = arrays["bf16"]
    actual = session.run(None, {"pixel_values": gradient_pixels, "grid_thw": grid})[0]
    np.testing.assert_allclose(actual, gradient_fp32, atol=1e-3, rtol=1e-3)
    gradient_metrics = {"fp32_parity": _vision_error_metrics(actual, gradient_fp32)}
    del session
    session = ort.InferenceSession(manifest["source"], providers=["CUDAExecutionProvider"])
    actual = _run_bf16_vision(session, gradient_pixels, grid, gradient_fp32.shape)
    gradient_metrics.update(
        {
            "ort_bf16_vs_fp32": _vision_error_metrics(actual, gradient_fp32),
            "torch_bf16_vs_fp32": _vision_error_metrics(gradient_bf16, gradient_fp32),
            "bf16_parity": _vision_error_metrics(actual, gradient_bf16),
        }
    )
    (capture_dir / "gradient_precision_comparison.json").write_text(
        json.dumps(gradient_metrics, indent=2), encoding="utf-8"
    )


def test_bf16_vision_capture(tmp_path, monkeypatch, capfd):
    """Capture checkpoints only; numerical parity is a separate diagnostic."""
    monkeypatch.setenv("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO", "1")
    _check_bf16_vision_pytorch(capture_dir=tmp_path)
    captured = capfd.readouterr()
    dispatch = [line for line in captured.out.splitlines() if "SdpaKernel=" in line]
    assert dispatch, "ORT did not report attention kernel dispatch"
    (tmp_path / "attention_dispatch.log").write_text("\n".join(dispatch), encoding="utf-8")
    print(captured.out)
    print(captured.err)
    _compare_ort_attention_dispatch(tmp_path, capfd)
    _compare_ort_same_qkv(tmp_path, capfd)
    _compare_first_block_norms(tmp_path)
    _compare_vision_activation_precision(tmp_path)
    _compare_saved_graph_fp32(tmp_path)


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
            for index in (
                config.vision_config.num_hidden_layers // 2,
                config.vision_config.num_hidden_layers - 1,
            ):
                layer = visual.encoder.layers[index]
                for label, module in (
                    ("norm1", layer.layer_norm1),
                    ("query", layer.self_attn.q_proj),
                    ("key", layer.self_attn.k_proj),
                    ("value", layer.self_attn.v_proj),
                ):
                    modules[f"block_{index:02d}_{label}"] = module

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

            for index in (
                config.vision_config.num_hidden_layers // 2,
                config.vision_config.num_hidden_layers - 1,
            ):
                handles.append(
                    visual.encoder.layers[index].self_attn.out_proj.register_forward_pre_hook(
                        partial(
                            capture_input,
                            capture,
                            torch_captures,
                            f"block_{index:02d}_attention_context",
                        )
                    )
                )
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

            def capture_projection(callback, captures, label, module, inputs):
                projected = torch.nn.functional.linear(inputs[0], module.weight)
                callback(captures, label, module, inputs, projected)

            for label, module in (
                ("block_00_attention_projection", visual.encoder.layers[0].self_attn.out_proj),
                ("block_00_mlp_up", visual.encoder.layers[0].mlp.fc1),
                ("block_00_mlp_down", visual.encoder.layers[0].mlp.fc2),
            ):
                handles.append(
                    module.register_forward_pre_hook(
                        partial(capture_projection, capture, torch_captures, label)
                    )
                )
            handles.append(
                visual.encoder.layers[0].mlp.fc2.register_forward_pre_hook(
                    partial(capture_input, capture, torch_captures, "block_00_mlp_activation")
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
        if capture_dir is not None:
            rounding = _replay_vision_rounding(
                visual, projector, packed, grid, onnx_captures, device
            )
            (capture_dir / "qkv_rounding_replay.json").write_text(
                json.dumps(rounding, indent=2), encoding="utf-8"
            )
            kernel_outputs = {}
            kernel_rounding = _replay_vision_rounding(
                visual,
                projector,
                packed,
                grid,
                onnx_captures,
                device,
                kernel_arithmetic=True,
                output_captures=kernel_outputs,
            )
            np.savez(capture_dir / "encoder_kernel_checkpoints.npz", **kernel_outputs)
            norm1_outputs = {}
            _replay_vision_rounding(
                visual,
                projector,
                packed,
                grid,
                onnx_captures,
                device,
                kernel_arithmetic=True,
                output_captures=norm1_outputs,
                normalized_captures={"block_00_norm1": onnx_captures["block_00_norm1"]},
            )
            np.savez(capture_dir / "norm1_reset_checkpoints.npz", **norm1_outputs)
            with np.load(capture_dir / "aligned_norms.npz") as arrays:
                aligned_norms = dict(arrays)
            both_norms_outputs = {}
            _replay_vision_rounding(
                visual,
                projector,
                packed,
                grid,
                onnx_captures,
                device,
                kernel_arithmetic=True,
                output_captures=both_norms_outputs,
                normalized_captures=aligned_norms,
            )
            np.savez(capture_dir / "both_norms_reset_checkpoints.npz", **both_norms_outputs)
            with np.load(capture_dir / "aligned_encoder_checkpoints.npz") as arrays:
                aligned_encoder = dict(arrays)
            for mode, reset_inputs in (
                ("local", True),
                ("cumulative", False),
                ("complete", False),
            ):
                norm_outputs = {}
                norm_sources = {
                    label: value
                    for label, value in aligned_encoder.items()
                    if mode == "complete" or (label.startswith("block_") and "_norm" in label)
                }
                norm_metrics = _replay_vision_rounding(
                    visual,
                    projector,
                    packed,
                    grid,
                    aligned_encoder,
                    device,
                    kernel_arithmetic=True,
                    reset_inputs=reset_inputs,
                    normalized_captures=norm_sources,
                    output_captures=norm_outputs,
                    projector_arithmetic=mode == "complete",
                )
                for index in range(config.vision_config.num_hidden_layers):
                    label = f"block_{index:02d}"
                    np.testing.assert_array_equal(
                        norm_outputs[label], aligned_encoder[label], err_msg=label
                    )
                if mode == "complete":
                    for label in (
                        "post_layernorm",
                        "projector_norm",
                        "projector_up",
                        "projector_activation",
                        "projector",
                    ):
                        np.testing.assert_array_equal(
                            norm_outputs[label], aligned_encoder[label], err_msg=label
                        )
                    hidden = torch.from_numpy(aligned_encoder["projector_up"]).to(
                        device, torch.bfloat16
                    )
                    with torch.inference_mode():
                        upstream = torch.nn.functional.gelu(hidden).float().cpu().numpy()
                        precise = hidden.double()
                        oracle = precise * (0.5 * torch.erfc(-precise * (2.0**-0.5)))
                        rounded = oracle.to(torch.bfloat16).float().cpu().numpy()
                        oracle = oracle.cpu().numpy()
                    activation_metrics = {
                        name: {
                            "vs_fp64": _vision_error_metrics(
                                actual.astype(np.float64), oracle
                            ),
                            "vs_rounded_fp64": _vision_error_metrics(actual, rounded),
                        }
                        for name, actual in (
                            ("ort", aligned_encoder["projector_activation"]),
                            ("torch", upstream),
                        )
                    }
                    (capture_dir / "projector_gelu_accuracy.json").write_text(
                        json.dumps(activation_metrics, indent=2), encoding="utf-8"
                    )
                np.savez(capture_dir / f"all_norms_{mode}_checkpoints.npz", **norm_outputs)
                (capture_dir / f"all_norms_{mode}_replay.json").write_text(
                    json.dumps(norm_metrics, indent=2), encoding="utf-8"
                )
            (capture_dir / "encoder_kernel_replay.json").write_text(
                json.dumps(kernel_rounding, indent=2), encoding="utf-8"
            )
            local_rounding = _replay_vision_rounding(
                visual,
                projector,
                packed,
                grid,
                onnx_captures,
                device,
                kernel_arithmetic=True,
                reset_inputs=True,
            )
            (capture_dir / "local_kernel_replay.json").write_text(
                json.dumps(local_rounding, indent=2), encoding="utf-8"
            )
            local_attention = _replay_vision_rounding(
                visual,
                projector,
                packed,
                grid,
                onnx_captures,
                device,
                kernel_arithmetic=True,
                reset_inputs=True,
                reset_attention=True,
            )
            (capture_dir / "local_exact_attention_replay.json").write_text(
                json.dumps(local_attention, indent=2), encoding="utf-8"
            )
            with torch.inference_mode():
                restored_features = visual(
                    packed[0].to(device=device, dtype=precision),
                    torch.from_numpy(grid.reshape(1, 3)).to(device),
                ).last_hidden_state
                restored = projector(restored_features)
            np.testing.assert_array_equal(
                restored.float().cpu().numpy(),
                output.float().cpu().numpy(),
                err_msg="Diagnostic patches changed the restored upstream reference",
            )
            del restored, restored_features
            from torch.nn.attention import SDPBackend, sdpa_kernel

            saved_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                visual.float()
                projector.float()
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    fp32_features = visual(
                        packed[0].to(device=device, dtype=torch.bfloat16).float(),
                        torch.from_numpy(grid.reshape(1, 3)).to(device),
                    ).last_hidden_state
                    fp32_reference = projector(fp32_features).cpu().numpy()
                np.save(capture_dir / "torch_fp32_reference.npy", fp32_reference)
                precision_metrics = {
                    "ort_bf16_vs_fp32": _vision_error_metrics(
                        onnx_captures["projector"], fp32_reference
                    ),
                    "torch_bf16_vs_fp32": _vision_error_metrics(
                        results[f"{device}/{precision}/{attention}"], fp32_reference
                    ),
                }
                (capture_dir / "fp32_reference_comparison.json").write_text(
                    json.dumps(precision_metrics, indent=2), encoding="utf-8"
                )
                del fp32_features
                ramp = torch.linspace(0, 1, 256)
                horizontal = ramp.unsqueeze(0).expand(256, -1)
                vertical = horizontal.T
                image = torch.stack(
                    (horizontal, vertical, 1 - horizontal * vertical)
                ).unsqueeze(0)
                gradient, gradient_h, gradient_w = patchify_images(
                    _normalized(image), patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
                )
                assert (gradient_h, gradient_w) == (grid_h, grid_w)
                gradient_pixels = gradient[0].to(device=device, dtype=torch.bfloat16)
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    gradient_features = visual(
                        gradient_pixels.float(),
                        torch.from_numpy(grid.reshape(1, 3)).to(device),
                    ).last_hidden_state
                    gradient_fp32 = projector(gradient_features).cpu().numpy()
                visual.bfloat16()
                projector.bfloat16()
                with torch.inference_mode(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    gradient_features = visual(
                        gradient_pixels, torch.from_numpy(grid.reshape(1, 3)).to(device)
                    ).last_hidden_state
                    gradient_bf16 = projector(gradient_features).float().cpu().numpy()
                np.savez(
                    capture_dir / "gradient_precision_reference.npz",
                    pixel_values=gradient_pixels.float().cpu().numpy(),
                    fp32=gradient_fp32,
                    bf16=gradient_bf16,
                )
                del gradient_features, gradient_pixels
            finally:
                torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        del visual, projector, module, features, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if capture_dir is not None:
        assert onnx_captures.keys() == torch_captures.keys()
        assert len(torch_captures) == config.vision_config.num_hidden_layers + 25
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
        attention_replay = _replay_vision_attention(
            onnx_captures, reference, config, bf16_device
        )
        (capture_dir / "attention_backend_replay.json").write_text(
            json.dumps(attention_replay, indent=2), encoding="utf-8"
        )
        import onnx_ir as ir

        options = ort.SessionOptions()
        optimized_path = capture_dir / "without_bias_fusion.onnx"
        options.optimized_model_filepath = str(optimized_path)
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CUDAExecutionProvider"],
            disabled_optimizers=["BiasSkipLayerNormFusion"],
        )
        control = _run_bf16_vision(session, packed[0].numpy(), grid, output_shape)
        del session
        optimized = ir.load(optimized_path)
        fused_bias_count = sum(
            node.op_type == "SkipLayerNormalization"
            and len(node.inputs) > 4
            and node.inputs[4] is not None
            for node in optimized.graph
        )
        assert fused_bias_count == 0, "BiasSkipLayerNormFusion was not disabled"
        control_metrics = _vision_error_metrics(control, torch_captures["projector"])
        print(f"Disabled bias fusion vs upstream: {control_metrics}")
        (capture_dir / "without_bias_fusion.json").write_text(
            json.dumps(control_metrics, indent=2), encoding="utf-8"
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
