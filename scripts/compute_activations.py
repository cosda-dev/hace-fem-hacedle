#!/usr/bin/env python3
"""
Compute activations using gguf-py dequantization
Simulates forward pass to generate reference activations
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')

from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

def rope_ref(x: np.ndarray, pos: int, dim: int, theta: float = 1000000.0) -> np.ndarray:
    """Apply RoPE to input vector."""
    result = x.copy()
    half = dim // 2
    for i in range(half):
        freq = theta ** (2.0 * i / half)
        inv_freq = 1.0 / freq
        angle = pos * inv_freq
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        
        x1 = result[i]
        x2 = result[i + half]
        
        result[i] = x1 * cos_val - x2 * sin_val
        result[i + half] = x1 * sin_val + x2 * cos_val
    
    return result

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm operation."""
    ss = np.sum(x * x)
    rms = np.sqrt(ss / len(x) + eps)
    return x * w / rms

def parse_metadata_value(value_str: str) -> any:
    """Parse ReaderField string to extract value."""
    try:
        if "'uint32'" in value_str or "'uint64'" in value_str:
            # Extract numeric value
            import re
            match = re.search(r'memmap\(\[(\d+)', value_str)
            if match:
                return int(match.group(1))
        elif "'float32'" in value_str:
            import re
            match = re.search(r'memmap\(\[(1\.\d+e?\d*)', value_str)
            if match:
                return float(match.group(1))
        return value_str
    except:
        return value_str

def main():
    model_path = "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    output_dir = Path("golden/qwen2505b/block0")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_path}")
    reader = GGUFReader(model_path)
    
    # Parse metadata
    metadata = {}
    for key, value in reader.fields.items():
        metadata[key] = str(value)
    
    hidden_size = 896
    n_head = 14
    n_kv_head = 2
    rope_theta = 1000000.0
    head_dim = hidden_size // n_head
    
    print(f"Model spec: hidden={hidden_size}, head={n_head}, head_dim={head_dim}, rope_theta={rope_theta}")
    
    # Load tensors
    tensors = {}
    for tensor_info in reader.tensors:
        name = tensor_info.name
        if "blk.0." in name and "weight" in name:
            try:
                data = tensor_info.data
                if tensor_info.tensor_type.value > 8:
                    f32_data = dequantize(data, tensor_info.tensor_type)
                else:
                    f32_data = data.astype(np.float32).flatten()
                
                # Reshape to match tensor shape
                shape = [int(x) for x in tensor_info.shape]
                f32_data = f32_data.reshape(shape)
                
                tensors[name] = f32_data
                print(f"Loaded {name}: shape={tensor_info.shape}, dtype={tensor_info.tensor_type.name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
    
    # Create dummy input
    np.random.seed(42)
    hidden = np.random.randn(hidden_size).astype(np.float32) * 0.01
    
    # RMSNorm
    attn_norm_w = tensors.get("blk.0.attn_norm.weight", np.ones(hidden_size, dtype=np.float32))
    attn_norm = rms_norm(hidden, attn_norm_w)
    np.save(output_dir / "blk_0_attn_norm.npy", attn_norm)
    
    # QKV Projection
    q_w = tensors.get("blk.0.attn_q.weight", np.zeros((hidden_size, n_head * head_dim), dtype=np.float32))
    k_w = tensors.get("blk.0.attn_k.weight", np.zeros((hidden_size, n_kv_head * head_dim), dtype=np.float32))
    v_w = tensors.get("blk.0.attn_v.weight", np.zeros((hidden_size, n_kv_head * head_dim), dtype=np.float32))
    
    # attn_norm: [hidden_size] -> matmul with weight: [hidden_size, out_features] -> out: [out_features]
    q = np.matmul(attn_norm, q_w).reshape(n_head, head_dim)
    k = np.matmul(attn_norm, k_w).reshape(n_kv_head, head_dim)
    v = np.matmul(attn_norm, v_w).reshape(n_kv_head, head_dim)
    
    # RoPE
    q_rope = np.array([rope_ref(q[0, i], 0, head_dim, rope_theta) for i in range(n_head)])
    k_rope = np.array([rope_ref(k[0, i], 0, head_dim, rope_theta) for i in range(n_kv_head)])
    
    np.save(output_dir / "blk_0_q_after_rope.npy", q_rope)
    np.save(output_dir / "blk_0_k_after_rope.npy", k_rope)
    
    # Attention
    scores = np.matmul(q_rope, k_rope.transpose(1, 0)) / np.sqrt(head_dim)
    probs = np.exp(scores - scores.max())
    probs = probs / probs.sum()
    
    np.save(output_dir / "blk_0_attn_scores.npy", scores)
    np.save(output_dir / "blk_0_attn_probs.npy", probs)
    
    attn_out = np.matmul(probs, v).flatten()
    np.save(output_dir / "blk_0_attn_out.npy", attn_out)
    
    print(f"\nSaved activations to {output_dir}")

if __name__ == "__main__":
    main()