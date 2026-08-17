# GGUF Parity Test Suite

This test suite validates HACE inference pipeline against llama.cpp reference implementation.

## Models Required

Place GGUF models in `D:/host/llama-models/`:
- `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (preferred - smallest)
- `Phi-3-mini-4k-instruct-Q4_K_M.gguf` (alternative)
- `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` (alternative)

## Test Execution Order

**CRITICAL**: Tests must pass in order. Do NOT proceed to next test if current fails.

### Test 1: Q4_K Dequant Parity
```bash
cargo test --test q4k_parity --features std -- --nocapture
```

**Pass Criteria:**
- `max_abs_error: < 1e-6` vs llama.cpp dequant
- `mean_abs_error: < 1e-7`

If FAIL → STOP. Do not run Block0 test.

### Test 2: RMSNorm Parity
```bash
cargo test --test rmsnorm_parity --features std -- --nocapture
```

**Pass Criteria:** `max_abs_error: < 1e-6`

### Test 3: RoPE Parity
```bash
cargo test --test rope_parity --features std -- --nocapture
```

**Pass Criteria:** 
- Extract rope_theta, rope_scaling from GGUF metadata
- Verify correct pair-wise/split-half pattern

### Test 4: Block0 Parity
```bash
cargo test --test block0_parity --features std -- --nocapture
```

**Pass Criteria:** `cosine_similarity: > 0.99999`

Only run after Test 1-3 pass.

### Test 5: KV Cache Parity
```bash
cargo test --test kv_parity --features std -- --nocapture
```

**Pass Criteria:** `max_abs_error: < 1e-6`, shape matches expected

### Test 6: Logits Parity
```bash
cargo test --test logits_parity --features std -- --nocapture
```

**Pass Criteria:**
- `top1_token: identical`
- `top5_tokens: identical`
- `cosine_similarity: > 0.9999`

## Generating Reference Data with llama.cpp

```bash
# Build llama.cpp with tensor dump support
cd /path/to/llama.cpp
make

# Dump tensor (need custom modification or use debug build)
./llama-cli --model D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --dump-tensor blk.0.attn_q.weight

# Alternative: use gguf Python package
pip install gguf
python scripts/dump_tensor.py D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf blk.0.attn_q.weight ref_q4k.npy
```

## Status Summary

```yaml
artifact_completion: 95%
runtime_parity: 15-20%
A3-10_status: NOT_READY
next_gate: GGUF_PARITY
```

**CRD Directive**: No mock tensors for A3-10. Use real GGUF data only.