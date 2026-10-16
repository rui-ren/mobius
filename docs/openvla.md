# OpenVLA: experimental ONNX export

Mobius exports **`openvla/openvla-7b`** as three portable FP32 ONNX components.
The standard-ONNX package passes one frozen full-weight **CPU** reference sample.
This is experimental conversion support, **not a qualified robot policy**.
H100 execution still fails the strict prefill-logit gate.

## Export

Use checkpoint revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`.
Review the model implementation and OpenVLA/Llama licenses, then download and
independently verify its three safetensors shards and configuration assets.
Remote-code execution is not required.

```python
import torch
from mobius import build

torch.set_num_threads(4)
package = build(
    "/absolute/pinned/openvla-checkpoint",
    dtype="f32",
    execution_provider="onnx-standard",
    trust_remote_code=False,
)
package.save("/absolute/fresh/openvla-onnx", max_workers=4, progress_bar=False)
```

Allow roughly **15 GB for weights and 30 GB for the FP32 export**, plus temporary
files and substantial CPU RAM. The graphs use **opset 24**; demonstrated execution
uses **ONNX Runtime 1.31**, not older opset-21 runtimes.

Use `onnx-standard` for the validated CPU baseline. The `default` profile's fused
residual-normalization path fails the full-checkpoint CPU logit gate; the `cpu`
profile and other provider-specific rewrites are unqualified. Matching actions
or successfully loading a graph does not make these profiles interchangeable.

## Exported package

Each component has a `model.onnx` and `model.onnx.data`:

| Component | Inputs | Outputs |
|---|---|---|
| `vision_encoder` | FP32 pixels `[B,6,224,224]` | Projected features `[B,256,4096]` |
| `embedding` | INT64 text tokens and image features | Embeddings with image patches after BOS |
| `decoder` | Embeddings, INT64 mask/positions, 32 layers of past KV state | Logits and updated KV state |

The caller owns preprocessing, prompts, generation, request-local caches and
action decoding. **ORT-GenAI architecture integration is not provided.**
The package uses standard ONNX operators, without custom domains or local functions.

## What has been validated

For reviewed implementation `f049872a1e240a360e9120e606abf5fe2a75c627`,
one full-weight CPU sample passes all 133 captured source boundaries, 128 cache
outputs and seven exact own-cache action tokens/values. Matched-operand ONNX
reference replay also passes all 1,339 nodes / 1,501 outputs.

The gate is **`abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`**, not absolute
error alone. Per-node replay is not original-class tracing of every primitive.

For the separate randomized tiny-model CPU fixture:

```bash
MOBIUS_OPENVLA_REFERENCE=/absolute/independent/tiny-fixture \
python -m pytest tests/integration/openvla_test.py -q
```

That fixture requires independently generated references and does not establish
full-checkpoint parity. The generic multimodal golden runner remains explicitly
blocked pending image/reference and action-generation integration.

## Remaining limits

**H100 qualification remains blocked** despite matching actions; it is separate
from the passing CPU baseline. A specific ORT kernel defect has not been proven.
Other precisions, quantization and non-default/scaled RoPE are unsupported or
unqualified. Quantization and non-default/scaled RoPE requests fail explicitly.

There is no representative robot-task evaluation, held-out certification,
latency qualification or production admission. Any Olive optimization,
quantization or provider rewrite must earn fresh numerical and task-quality
evidence. **Keep production and release gates closed.**
