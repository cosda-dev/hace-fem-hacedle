#!/usr/bin/env python3
"""T6.5.2: Attention Score Closure - Compute Q@K^T / sqrt(d)"""

import numpy as np
from pathlib import Path
import json
import hashlib

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def save_operator(name: str, tensor: np.ndarray, out_dir: Path):
    tensor = tensor.astype(np.float32)
    tensor_bytes = tensor.tobytes()
    
    (out_dir / f"{name}.bin").write_bytes(tensor_bytes)
    
    meta = {
        "operator_id": name,
        "layer_id": 0,
        "tensor_shape": list(tensor.shape),
        "dtype": "f32",
        "sha256": sha256_bytes(tensor_bytes),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))

def main():
    golden_dir = Path("golden/block0_operators")
    
    # Load RoPE outputs
    q_rope = np.frombuffer((golden_dir / "05_rope_q.bin").read_bytes(), dtype=np.float32)
    k_rope = np.frombuffer((golden_dir / "06_rope_k.bin").read_bytes(), dtype=np.float32)
    
    head_dim = 64
    n_head = 14
    n_head_kv = 2
    
    # Reshape: Q -> [14, 64], K -> [2, 64]
    q = q_rope.reshape(n_head, head_dim)  # [14, 64]
    k = k_rope.reshape(n_head_kv, head_dim)  # [2, 64]
    
    # GQA expand K to match Q heads
    # heads 0-6 use KV head 0, heads 7-13 use KV head 1
    k_expanded = np.zeros((n_head, head_dim), dtype=np.float32)
    for i in range(n_head):
        kv_idx = i // (n_head // n_head_kv)  # 0 for i<7, 1 for i>=7
        k_expanded[i] = k[kv_idx]
    
    print(f"K expanded shape: {k_expanded.shape}")
    
    # Compute attention scores: Q @ K^T / sqrt(head_dim)
    # scores[i, j] = sum over dim: q[i] * k[j]
    scores = np.zeros((n_head, n_head), dtype=np.float32)
    for i in range(n_head):
        for j in range(n_head):
            scores[i, j] = np.sum(q[i] * k_expanded[j]) / np.sqrt(head_dim)
    
    # Flatten for storage
    scores_flat = scores.flatten()
    save_operator("07_attention_scores", scores_flat, golden_dir)
    
    print(f"Attention scores shape: {scores.shape}")
    print(f"Saved 07_attention_scores.bin")

if __name__ == "__main__":
    main()