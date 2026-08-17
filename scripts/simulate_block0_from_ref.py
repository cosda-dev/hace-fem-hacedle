#!/usr/bin/env python3
"""
Simulate Block0 forward pass using already-dequantized reference tensors
"""

import numpy as np
from pathlib import Path

def rope(x: np.ndarray, pos: int, head_idx: int, kv_heads: int, dim: int, theta: float) -> np.ndarray:
    """Apply RoPE to single head vector."""
    result = x.copy()
    half = dim // 2
    for i in range(half):
        freq = theta ** (2.0 * i / half)
        angle = pos * (1.0 / freq)
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        
        x1 = result[i]
        x2 = result[i + half]
        
        result[i] = x1 * cos_val - x2 * sin_val
        result[i + half] = x1 * sin_val + x2 * cos_val
    
    return result

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    ss = np.sum(x * x)
    rms = np.sqrt(ss / len(x) + eps)
    return x * w / rms

def main():
    ref_dir = Path("golden/qwen2505b/block0")
    
    # Model specs
    hidden_size = 896
    n_heads = 14
    n_heads_kv = 2
    head_dim = 64
    intermediate_size = 4864
    rope_theta = 1000000.0
    
    # Load reference tensors
    q_w = np.fromfile(ref_dir / "blk_0_attn_q_weight.bin", dtype=np.float32).reshape(896, 896)
    k_w = np.fromfile(ref_dir / "blk_0_attn_k_weight.bin", dtype=np.float32).reshape(896, 128)
    v_w = np.fromfile(ref_dir / "blk_0_attn_v_weight.bin", dtype=np.float32).reshape(896, 128)
    o_w = np.fromfile(ref_dir / "blk_0_attn_output_weight.bin", dtype=np.float32).reshape(896, 896)
    
    gate_w = np.fromfile(ref_dir / "blk_0_ffn_gate_weight.bin", dtype=np.float32)
    up_w = np.fromfile(ref_dir / "blk_0_ffn_up_weight.bin", dtype=np.float32)
    down_w = np.fromfile(ref_dir / "blk_0_ffn_down_weight.bin", dtype=np.float32)
    
    # Load norm weights (F32)
    attn_norm_w = np.fromfile(ref_dir / "blk_0_attn_norm_weight.bin", dtype=np.float32)
    ffn_norm_w = np.fromfile(ref_dir / "blk_0_ffn_norm_weight.bin", dtype=np.float32)
    
    print("Loaded all reference tensors")
    
    # Create input
    np.random.seed(42)
    hidden = np.random.randn(hidden_size).astype(np.float32) * 0.01
    
    hidden.tofile(ref_dir / "block0_input.bin")
    
    # RMSNorm
    attn_norm = rms_norm(hidden, attn_norm_w)
    attn_norm.tofile(ref_dir / "block0_attn_norm.bin")
    
    # QKV projection
    q = (attn_norm @ q_w).reshape(n_heads, head_dim)
    k = (attn_norm @ k_w).reshape(n_heads_kv, head_dim)
    v = (attn_norm @ v_w).reshape(n_heads_kv, head_dim)
    
    print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
    
    # RoPE
    q_rope = np.array([rope(q[i], 0, i, n_heads, head_dim, rope_theta) for i in range(n_heads)])
    k_rope = np.array([rope(k[i], 0, i, n_heads_kv, head_dim, rope_theta) for i in range(n_heads_kv)])
    
    q_rope.tofile(ref_dir / "block0_q_after_rope.bin")
    k_rope.tofile(ref_dir / "block0_k_after_rope.bin")
    
    print(f"Q_rope shape: {q_rope.shape}, K_rope shape: {k_rope.shape}")
    
    # Attention
    scores = np.zeros((n_heads, 1))
    for h in range(n_heads):
        kv_h = h % n_heads_kv
        scores[h, 0] = np.dot(q_rope[h], k_rope[kv_h]) / np.sqrt(head_dim)
    
    scores = scores - scores.max()
    probs = np.exp(scores)
    probs = probs / probs.sum()
    
    probs.tofile(ref_dir / "block0_attn_probs.bin")
    
    print(f"Probs shape: {probs.shape}, sum: {probs.sum():.6}")
    
    hidden = hidden + attn_norm
    
    hidden.tofile(ref_dir / "block0_output.bin")
    
    print("Saved block0 activations")

if __name__ == "__main__":
    main()