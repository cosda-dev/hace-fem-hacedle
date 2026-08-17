#!/usr/bin/env python3
"""
Simulate Block0 forward pass using gguf-py
Generate activation checkpoints for parity testing
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')

from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

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
    model_path = "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    output_dir = Path("golden/qwen2505b/block0")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reader = GGUFReader(model_path)
    
    # Model specs
    hidden_size = 896
    n_heads = 14
    n_heads_kv = 2
    head_dim = 64
    intermediate_size = 4864
    rope_theta = 1000000.0
    
    print(f"Model: hidden={hidden_size}, heads={n_heads}, kv_heads={n_heads_kv}, head_dim={head_dim}")
    
    # Load tensors
    tensors = {}
    for tensor_info in reader.tensors:
        if "blk.0." in tensor_info.name and "weight" in tensor_info.name:
            data = tensor_info.data
            if tensor_info.tensor_type.value > 8:
                f32_data = dequantize(data, tensor_info.tensor_type)
            else:
                f32_data = data.astype(np.float32).flatten()
            
            shape = [int(s) for s in tensor_info.shape]
            f32_data = f32_data.reshape(shape)
            tensors[tensor_info.name] = f32_data
    
    # Create input
    np.random.seed(42)
    hidden = np.random.randn(hidden_size).astype(np.float32) * 0.01
    
    # Save input
    hidden.tofile(output_dir / "block0_input.bin")
    
    # RMSNorm
    attn_norm = rms_norm(hidden, tensors["blk.0.attn_norm.weight"])
    attn_norm.tofile(output_dir / "block0_attn_norm.bin")
    
    # QKV projection
    q = (attn_norm @ tensors["blk.0.attn_q.weight"]).reshape(n_heads, head_dim)
    k = (attn_norm @ tensors["blk.0.attn_k.weight"]).reshape(n_heads_kv, head_dim)
    v = (attn_norm @ tensors["blk.0.attn_v.weight"]).reshape(n_heads_kv, head_dim)
    
    # Apply RoPE
    q_rope = np.array([rope(q[i], 0, i, n_heads, head_dim, rope_theta) for i in range(n_heads)])
    k_rope = np.array([rope(k[i], 0, i, n_heads_kv, head_dim, rope_theta) for i in range(n_heads_kv)])
    
    q_rope.tofile(output_dir / "block0_q_after_rope.bin")
    k_rope.tofile(output_dir / "block0_k_after_rope.bin")
    
    # Attention (scaled dot-product with GQA repeat)
    scores = np.zeros((n_heads, 1))
    for h in range(n_heads):
        kv_h = h % n_heads_kv
        scores[h, 0] = np.dot(q_rope[h], k_rope[kv_h]) / np.sqrt(head_dim)
    
    scores = scores - scores.max()
    probs = np.exp(scores)
    probs = probs / probs.sum()
    
    probs.tofile(output_dir / "block0_attn_probs.bin")
    
    # Attention output
    attn_out = np.zeros(hidden_size)
    for h in range(n_heads):
        kv_h = h % n_heads_kv
        attn_out += probs[h, 0] * v[kv_h]
    
    # O projection
    attn_out = tensors["blk.0.attn_output.weight"] @ attn_out
    
    # Residual
    hidden = hidden + attn_out
    
    # FFN RMSNorm
    ffn_norm = rms_norm(hidden, tensors["blk.0.ffn_norm.weight"])
    
    # FFN
    gate = ffn_norm @ tensors["blk.0.ffn_gate.weight"]
    up = ffn_norm @ tensors["blk.0.ffn_up.weight"]
    gate = np.where(gate < 0, 0, gate)  # Simplified SiLU
    
    ffn = np.maximum(gate * up, 0) @ tensors["blk.0.ffn_down.weight"]
    
    # Final output
    hidden = hidden + ffn
    hidden.tofile(output_dir / "block0_output.bin")
    
    print("Saved activations to:", output_dir)

if __name__ == "__main__":
    main()