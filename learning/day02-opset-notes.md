# Day 2: ONNX Domains, Opsets, Schemas, and Attributes

> Working-branch check: `.git/HEAD` points to
> `refs/heads/copilot/onnx-tensorrt-domains-schema-attributes` in this coding-agent
> checkout, not `refs/heads/ruiren/tensorRT-mobius`.

- **Domain:** A domain is an operator namespace. The standard ONNX namespace is
  normally encoded as `""`, while a name such as `com.microsoft` identifies a
  separate vendor namespace. The same operator name in two domains does not
  necessarily have the same contract or implementation.
- **Opset:** `src/mobius/_constants.py` sets `OPSET_VERSION = 24`. An opset import
  assigns one schema-version contract to each imported domain. All nodes from that
  domain in a graph must therefore be valid under that imported contract; changing
  only the import number can make an existing node mean something different or use
  inputs that the selected schema does not define.
- **Schema:** The schema selected by a node's domain, operator name, and imported
  opset defines its ordered inputs and outputs, allowed data types, attributes,
  defaults, and validation rules. Schema validity does not guarantee that every
  TensorRT parser version implements that operator contract.
- **Attribute:** An attribute is configuration stored on a node, not a tensor value
  flowing through a graph edge. It specializes the schema-defined operation, while
  a tensor input is an ordered value that can be produced by another node or exposed
  as a graph input.
- **Mobius's opset-24 requirement:** The static-cache path uses opset-24
  `TensorScatter` to update preallocated key/value caches and the opset-24 Attention
  input `nonpad_kv_seqlen`. `_maybe_apply_opset_lowering()` recursively preserves
  opset 24 when it finds either a standard-domain `TensorScatter` node or a populated
  Attention input at index `6`; merely relabeling such a graph as opset 23 would not
  rewrite those operations into an equivalent older contract.
- **TensorRT risk:** A nonstandard-domain node may be checker-valid because its
  vendor schema is available to the checker, yet fail TensorRT parsing when TensorRT
  has no importer or plugin for that domain and operator. Even standard-domain nodes
  can be incompatible when the TensorRT version does not support their imported
  opset, schema revision, data types, or attribute combination.
- **Robotics relevance:** Keeping these contracts precise avoids parser surprises
  when dynamic perception and KV-cache graphs are converted into deterministic
  TensorRT engines for edge robots.

## Attention tensor inputs

`_apply_attention()` supplies these ordered tensor operands to ONNX Attention:

1. Index `0`: `query`
2. Index `1`: `key` (the updated full key cache in static-cache mode)
3. Index `2`: `value` (the updated full value cache in static-cache mode)
4. Index `3`: optional `attn_mask`
5. Index `4`: optional `past_key`
6. Index `5`: optional `past_value`
7. Index `6`: optional `nonpad_kv_seqlen`, used by Mobius in static-cache mode

`nonpad_kv_seqlen` is a tensor input, not an attribute. In static-cache mode,
Mobius leaves inputs `4` and `5` empty because the full caches are already supplied
through `key` and `value`.

## Attention attributes

`_apply_attention()` sets these schema-defined Attention attributes:

- `q_num_heads`: the number of query heads
- `kv_num_heads`: the number of key/value heads
- `scale`: the query-key score scale
- `softcap`: the optional score soft cap
- `is_causal`: whether Attention applies causal masking

In particular, `q_num_heads`, `kv_num_heads`, `scale`, and `is_causal` configure the
Attention node; they do not occupy tensor input positions.

This analysis uses only ONNX graph metadata and TensorRT parser compatibility; it
does not use an inference runtime dependency or API.
