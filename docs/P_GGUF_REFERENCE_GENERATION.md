# GGUF Reference Generation Guide

## Prerequisites

Python 3 + gguf package:
```bash
pip install gguf numpy
```

## Step 1: Export Reference Tensors

```bash
cd t:\hace\engine\hace\fem\hacedle

# Export all block0 tensors to reference format
python scripts/export_gguf_tensors.py "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" reference
```

Output:
```
reference/
├── model_metadata.json
├── blk_0_attn_q_weight.npy
├── blk_0_attn_k_weight.npy
├── blk_0_attn_v_weight.npy
└── ...
```

## Step 2: Generate Golden Bundle

```bash
# Create standardized golden bundle
python scripts/generate_golden_bundle.py qwen25-0.5b "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
```

Output:
```
golden/qwen2505b/
├── model_spec.sio
├── block0/
│   ├── q.npy
│   ├── k.npy
│   ├── v.npy
│   ├── q_meta.json
│   └── ...
└── manifest.json
```

## Step 3: Run Parity Tests

```bash
cargo test --test llama_cpp_parity --features std -- --nocapture
```

## Reference Data Format

### Tensor Binary (.bin)
- Raw f32 little-endian
- Shape stored in accompanying .meta.yaml

### Model Spec (.sio)
```yaml
arch: qwen2
hidden_size: 896
n_head: 14
rope_theta: 1000000.0
context_length: 32768
```

### Manifest
```json
{
  "model": "qwen25-0.5b",
  "version": "1.0",
  "files": {
    "block0_q": "block0/q.npy",
    "block0_k": "block0/k.npy"
  }
}
```