# Activation Schema - Alpha-3 Phase E1

## Standard Format (cross-runtime compatible)

```yaml
schema_version: 1

model_spec:
  name: qwen25-0.5b
  arch: qwen2
  hidden_size: 896
  n_head: 14
  n_kv_head: 2
  head_dim: 64
  intermediate_size: 4864
  rope_theta: 1000000.0
  context_length: 32768

activations:
  - stage: block0_input
    tensor:
      dtype: f32
      shape: [1, 896]
    stats:
      min: ...
      max: ...
      mean: ...
      l2_norm: ...

  - stage: block0_rms_attn
    tensor:
      dtype: f32
      shape: [896]
    ...

  - stage: block0_q
    tensor:
      dtype: f32
      shape: [14, 64]
    ...

  - stage: block0_q_after_rope
    tensor:
      dtype: f32
      shape: [14, 64]
    ...

  - stage: block0_attn_scores
    tensor:
      dtype: f32
      shape: [14, seq_len]
    ...

  - stage: block0_attn_probs
    tensor:
      dtype: f32
      shape: [14, seq_len]
    ...

  - stage: block0_attn_out
    tensor:
      dtype: f32
      shape: [896]
    ...

  - stage: block0_output
    tensor:
      dtype: f32
      shape: [896]
    ...
```

## File Format

Binary: Raw f32 little-endian
Metadata: JSON/YAML with shape, stats, sha256

## Runtime Compatibility

- `llama.cpp` - requires `--dump-activations` patch
- `vLLM` - export via `model.forward()` hooks
- `HF Transformers` - use `output_hidden_states=True`
- `hacedle` - use `ActivationDump` struct