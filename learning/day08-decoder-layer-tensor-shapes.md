# Day 8: Decoder-Layer Tensor Shapes

> Working-branch check: this coding-agent checkout is on
> `copilot/resolve-decoder-layer-tensor-shapes`, not
> `ruiren/tensorRT-mobius`.

Use standard multi-head attention (MHA), without a KV cache, with:

- `B`: batch size
- `S`: sequence length
- `V`: vocabulary size
- `H`: hidden size
- `N_h`: number of attention heads
- `D_h`: head dimension
- `I`: MLP intermediate size

The concrete configuration is `B=2`, `S=16`, `H=512`, `N_h=8`, `D_h=64`,
and `I=2048`. Its model-width contract is

```text
H = N_h * D_h = 8 * 64 = 512
```

Mobius stores each standard-MHA projection weight as `[out_features,
in_features] = [H, H]`. `Linear.forward()` transposes it for the matrix
multiplication, so `[B, S, H] @ [H, H]` produces `[B, S, H]`.

## Tensor derivation

| Tensor or stage | Symbolic shape | Concrete shape | Elements |
|---|---:|---:|---:|
| Integer token IDs | `[B, S]` | `[2, 16]` | 32 |
| Embedding output / decoder input | `[B, S, H]` | `[2, 16, 512]` | 16,384 |
| Attention-normalized input | `[B, S, H]` | `[2, 16, 512]` | 16,384 |
| Packed Q projection | `[B, S, H]` | `[2, 16, 512]` | 16,384 |
| Packed K projection | `[B, S, H]` | `[2, 16, 512]` | 16,384 |
| Packed V projection | `[B, S, H]` | `[2, 16, 512]` | 16,384 |
| Q by head | `[B, N_h, S, D_h]` | `[2, 8, 16, 64]` | 16,384 |
| K by head | `[B, N_h, S, D_h]` | `[2, 8, 16, 64]` | 16,384 |
| V by head | `[B, N_h, S, D_h]` | `[2, 8, 16, 64]` | 16,384 |
| Transposed K | `[B, N_h, D_h, S]` | `[2, 8, 64, 16]` | 16,384 |
| Attention scores `Q @ K^T` | `[B, N_h, S, S]` | `[2, 8, 16, 16]` | 4,096 |
| Softmax probabilities `P` | `[B, N_h, S, S]` | `[2, 8, 16, 16]` | 4,096 |
| Context by head `P @ V` | `[B, N_h, S, D_h]` | `[2, 8, 16, 64]` | 16,384 |
| Merged heads | `[B, S, H]` | `[2, 16, 512]` | 16,384 |
| Attention output projection | `[B, S, H]` | `[2, 16, 512]` | 16,384 |
| First residual output | `[B, S, H]` | `[2, 16, 512]` | 16,384 |
| MLP-normalized input | `[B, S, H]` | `[2, 16, 512]` | 16,384 |
| `gate_proj` output | `[B, S, I]` | `[2, 16, 2048]` | 65,536 |
| Activated gate branch | `[B, S, I]` | `[2, 16, 2048]` | 65,536 |
| `up_proj` branch | `[B, S, I]` | `[2, 16, 2048]` | 65,536 |
| Gated activation `act(gate) * up` | `[B, S, I]` | `[2, 16, 2048]` | 65,536 |
| MLP output after `down_proj` | `[B, S, H]` | `[2, 16, 512]` | 16,384 |
| Second residual / decoder output | `[B, S, H]` | `[2, 16, 512]` | 16,384 |

The embedding is an ONNX `Gather`: IDs `[B, S]` select rows from the
embedding table `[V, H]`, appending the row width and producing `[B, S, H]`.
Both layer normalizations operate over the last dimension and preserve that
shape.

For Q, K, and V, the packed-to-head transformation is:

```text
[B, S, H] -> [B, S, N_h, D_h] -> [B, N_h, S, D_h]
```

It is a reshape followed by a transpose, so the element count does not
change. Scaled dot-product attention then has the contracts

```text
scores:  [B, N_h, S, D_h] @ [B, N_h, D_h, S]
         -> [B, N_h, S, S]
P:       softmax(scores / sqrt(D_h)) -> [B, N_h, S, S]
context: [B, N_h, S, S] @ [B, N_h, S, D_h]
         -> [B, N_h, S, D_h]
```

The inverse transpose and reshape merge the context heads into `[B, S, H]`.
The output projection uses another `[H, H]` weight and therefore also returns
`[B, S, H]`.

Mobius passes packed Q/K/V directly to the opset-24 `Attention` operator and
sets `q_num_heads=N_h` and `kv_num_heads=N_h`. The per-head tensors, score
matrix, probabilities, and head merge above describe that fused operator's
logical MHA computation rather than separate values in the Mobius ONNX graph.
A TensorRT attention tactic may likewise avoid materializing every logical
temporary.

## Pre-norm decoder trace

`DecoderLayer._forward_pre_norm()` preserves `[B, S, H]` at both residual
boundaries:

```text
residual_0 = hidden_states                         [B, S, H]
attn_input = input_layernorm(hidden_states)        [B, S, H]
attn_output = self_attn(attn_input)                [B, S, H]
hidden_states = residual_0 + attn_output           [B, S, H]  # residual 1

residual_1 = hidden_states                         [B, S, H]
mlp_input = post_attention_layernorm(hidden_states) [B, S, H]
mlp_output = down_proj(act(gate_proj(mlp_input))
                       * up_proj(mlp_input))        [B, S, H]
hidden_states = residual_1 + mlp_output            [B, S, H]  # residual 2
```

The first addition combines the original decoder input and projected
attention output. The second combines the first residual output and the MLP
output. Every operand is `[2, 16, 512]`, so the decoder-layer output remains
`[2, 16, 512]`.

## Largest activations and sequence growth

The three widest named activations are tied:

1. activated gate branch: 65,536 elements
2. up branch: 65,536 elements
3. gated activation: 65,536 elements

The `gate_proj` result before activation is an additional equal-size
temporary. At this configuration, each is larger than the 16,384-element
model-width tensors and the 4,096-element score or probability tensor.

Scores and probabilities each contain `B * N_h * S^2` elements. Their size
therefore grows quadratically with sequence length, while the packed Q/K/V,
context, and MLP activations grow linearly with `S`. This makes the logical
`[B, N_h, S, S]` tensors increasingly important to TensorRT tactic selection
and optimization-profile memory budgets even when a fused kernel can reduce
their physical storage.
