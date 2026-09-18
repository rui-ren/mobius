---
name: debugging-onnx-parity
description: >
  Use when debugging ONNX export numerical parity, divergent intermediate
  activations, BF16/FP16 rounding, or suspected ONNX Runtime CUDA kernel bugs
  versus PyTorch or Hugging Face. Isolate operators with identical-input replay,
  verify instrumentation integrity, distinguish inherited from local errors,
  construct independent correctness checks, and validate fixes end to end.
---

# Debugging ONNX Numerical Parity

## Core method

Compare equivalent computations using identical inputs and parameters:

```text
Captured ONNX operation input + verified parameters
    |                                  |
    v                                  v
ONNX operation                    PyTorch equivalent
    |                                  |
captured output                      replay
    |__________________________________|
                   compare
```

Weights and activations are different evidence sources. Load reference weights
from the original Hugging Face checkpoint and verify their values against ONNX
initializers, accounting for transposes, packing, casts, and quantization.
Capture runtime activations from ONNX. Do not call activations checkpoint weights.
Loading ONNX weights into both sides alone cannot validate the export's weight mapping.

Replay locates a local difference; an independent correctness check establishes
whether it is a defect. Neither side is automatically ground truth. Agreement
on one input does not establish universal correctness.

## Procedure

### 1. Freeze a reproducible failure

- Record input arrays, preprocessing, masks, positions, cache state, checkpoint
  revision, graph and external-data identities, opset, dtypes, shapes, GPU,
  runtime build revisions, provider options, and optimization settings.
- Keep the original reference and acceptance tolerances unchanged. Run twice
  to check repeatability before interpreting tiny numerical differences.
- Save artifacts outside rotating pytest directories. Resolve source paths
  absolutely. Never reuse a capture directory as pytest `--basetemp`: it is cleared.
- Start with one failing sample; reserve independent samples for validation.

### 2. Localize from broad boundaries to operators

Compare embeddings, blocks, and final output, then inspect the earliest suspicious
region. Sample middle/final blocks when useful to measure accumulation. This is
coarse-to-fine localization, not strict binary search: differences can cancel or
reappear, so matching a midpoint does not certify earlier nodes.

Record shapes, finite values, mean/max absolute error, relative L2 error (handle
zero reference norm), exact-match fraction, and failures of the agreed elementwise
tolerance. Mean absolute error is not a percentage. Separate the first bitwise
difference from the first tolerance failure and from demonstrated task degradation.

### 3. Validate capture integrity before attribution

- Instrument a copy with `onnx_ir`; expose selected existing values as outputs.
  Do not use explicit protobuf APIs in mobius.
- Run the unchanged graph as a control. Require exact final-output agreement
  where execution is deterministic; otherwise establish nondeterministic bounds
  before attributing differences. Inspect optimized graphs and actual dispatch.
- Extra outputs can prevent bias, residual, or Gemm fusion. If execution changes,
  reject the capture as evidence of the original path and choose another boundary.
- Use PyTorch forward hooks for matching module outputs, and pre-hooks for inputs.
  Remove hooks and restore temporary patches/options even on failure.
- Compare semantic boundaries, not names: PyTorch Linear may map to MatMul + Add.
  Saved tensor names may retain an unfused op name after their producer is fused.

### 4. Replay with identical inputs

For each suspect operation, list all operands and attributes. Feed its captured
ONNX inputs into the reference computation; do not recompute upstream layers.
Preserve dtype, shape, layout, epsilon, axes, scale, masks, cache, and bias semantics.
First run the unchanged reference arithmetic, then vary one factor at a time.

Minimal Q-projection example, assuming arrays and verified HF tensors are loaded:

```python
hidden = torch.from_numpy(captures["block_00_norm1"]).to("cuda", torch.bfloat16)
weight = reference_weights["q_proj.weight"].to("cuda", torch.bfloat16)
bias = reference_weights["q_proj.bias"].to("cuda", torch.bfloat16)
actual = captures["block_00_query"]
with torch.inference_mode():
    upstream = torch.nn.functional.linear(hidden, weight, bias)
    separate = torch.nn.functional.linear(hidden, weight) + bias
replay = separate.float().cpu().numpy()
assert actual.shape == replay.shape
np.testing.assert_array_equal(actual, replay)
```

This exact assertion encodes an observed replay property, not a universal demand
that all valid kernels be bitwise identical. Preserve and report `upstream` too:
matching a modified arithmetic variant does not establish unchanged HF parity.
Float32 storage of BF16 captures is lossless; casting back to BF16 restores the
captured values. It does not mean the original model ran in Float32.

Interpretation:
- Same input, same output: no local discrepancy on this sample.
- Same input, different output: investigate local semantics and arithmetic.
- Different upstream inputs: downstream mismatch alone cannot indict that operator.

### 5. Separate the competing explanations

Test one variable at a time: fused versus separate bias, weight layout, residual
addition order, intermediate rounding, accumulation precision, reduction algorithm,
activation approximation, and attention backend. Verify actual kernel dispatch;
accepted provider options do not prove a requested backend ran.

Build a minimal ONNX subgraph and require it to reproduce the original captured
operator output. A standalone graph with different dispatch is a separate control.
Use FP64 on identical rounded operands when feasible, or another independent
reference. FP32 end-to-end agreement is evidence, not proof that BF16 is correct.

Substituting captured norm/context outputs can test how differences propagate.
Label this as attribution only: checkpoint substitution is never a production
fix or an unmodified end-to-end parity pass.

### 6. Demonstrate a mechanism, then test relevance

Inspect source at the installed runtime revision. Derive a discriminating input
or property from a falsifiable hypothesis, rather than constructing arbitrary
stress cases until something fails. Examples include normalization translation
invariance (subject to representability), attention mask boundaries, and cache
prefill/decode equivalence. Check multiple inputs and algorithmic branch boundaries.

For variance computed as E[x^2] - E[x]^2, large offsets with small variance expose
cancellation. A centered-input control discriminates this from ordinary output
rounding. Distinguish directly measured internal statistics from values inferred
from the output. Do not claim exact kernel reproduction from an approximate replay.

After a synthetic defect is demonstrated, check whether the original model input
has the necessary conditions. Keep these claims separate:
1. Local difference found.
2. Independent defect reproduced.
3. Defect shown to cause the original failure.
4. Original failure fixed and regressions checked.

### 7. Separate diagnostics from correctness tests

- A capture/probe may pass while recording severe errors. Say this explicitly
  in its docstring, printed output, and user-facing results.
- Add a separate accuracy test that asserts the required output against the
  independent oracle and fails on the unfixed runtime. Show the exact command,
  failing assertion, actual/expected values, and required environment.
- Do not relax tolerances, assert the known bad value as correctness, or mark a
  defect fixed because a data-collection test passes.
- Keep diagnostic APIs/helpers reusable instead of duplicating model-loading code.

### 8. Fix the owner and close the loop

Patch only the evidenced owner: preprocessing, weight mapping, exporter, rewrite,
runtime metadata, or kernel. Run the focused check immediately after the edit.
The same failing regression must pass without changing inputs or acceptance criteria.
Then validate the untouched original workflow, independent inputs, relevant shapes,
dtypes/providers, shared callers, and performance. A runtime-wide fix needs a wider
regression matrix than a model-local change. Preserve unrelated work. Request
authorization before external builds/downloads, scope expansion, or commits.

## Worked evidence: Cosmos3-Edge

The investigation used ORT 1.30.0 revision f2c39fe2f and Torch 2.14.0+cu130 on
an RTX 4060 Laptop GPU. These numbers are observations, not portable guarantees.

- Patch projection and positional sum matched exactly. First norm differed in
  80/294912 values. That alone did not establish a kernel defect.
- Same captured norm1 input plus separate BF16 matmul/bias reproduced Q/K/V
  exactly. Fused PyTorch linear differed. Same-QKV efficient attention matched
  across runtimes; original attention backends differed.
- After aligning other arithmetic, substituting both norm outputs made block 0
  exact; extending to all encoder norms made all blocks exact in controlled replay.
  This attributed drift but was not a production parity fix.
- Isolated norm stress: 1152 values, all 128 except one 129, zero skip, unit gamma,
  zero beta, epsilon 1e-6. FP64 first output ~33.90685, standard ORT/Torch BF16 34,
  fused ORT BF16 2024. Centering by 128 restored accuracy; fused FP32 also failed.
- This exposed unstable raw-moment variance. The real first-norm input was well
  conditioned, and replacing all 55 fused norms did not restore Cosmos parity.
  Do not present the synthetic defect as the proven Cosmos root cause or a fix.

Existing implementation: [Cosmos diagnostics](../../../tests/cosmos3_edge_integration_test.py).
`test_bf16_vision_capture` collects the full-model evidence;
`test_skip_layer_norm_probe` collects isolated evidence;
`test_skip_layer_norm_near_constant_accuracy` asserts synthetic correctness;
`test_skip_layer_norm_model_control` measures full-model replacement effects.
The isolated tests currently require `COSMOS3_EDGE_CAPTURE_DIR` from a completed
capture; they are not independent of those artifacts. Inspect current code before
running them. No ORT kernel patch was validated in this investigation.

## Agent reporting and related skills

Maintain a compact ledger: observation, hypothesis, disconfirming experiment,
result, decision, artifact paths, remaining uncertainty. If an experiment fails
to discriminate, choose a new discriminating check instead of repeating controls.
Stop with an explicit prerequisite/budget blocker when execution is unavailable.
Report status as localized, defect reproduced, fixed, or blocked; distinguish
synthetic accuracy, model parity, and downstream task quality.

Use [writing-tests](../writing-tests/SKILL.md) for repository test conventions,
[debugging-multimodal](../debugging-multimodal/SKILL.md) for pipeline boundaries,
and [attention-optimization](../attention-optimization/SKILL.md) for CUDA dispatch.