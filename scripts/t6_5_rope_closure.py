#!/usr/bin/env python3
"""T6.5.1: RoPE Closure - Generate 05_rope_q.bin, 06_rope_k.bin"""

import numpy as np
from pathlib import Path
import json
import hashlib

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def apply_rope(input_tensor: np.ndarray, pos: int, dim: int, theta: float = 1000000.0) -> np.ndarray:
    """Apply RoPE pairwise rotation - input [n_heads * head_dim], output same"""
    n_total = input_tensor.shape[0]
    n_heads = n_total // dim
    out = np.zeros_like(input_tensor)
    
    for h in range(n_heads):
        x = input_tensor[h * dim : (h + 1) * dim]
        for i in range(dim // 2):
            freq = theta ** (2.0 * i / dim)
            angle = pos / freq
            cos_val = np.cos(angle)
            sin_val = np.sin(angle)
            
            x0 = x[i]
            x1 = x[i + dim // 2]
            
            out[h * dim + i] = x0 * cos_val - x1 * sin_val
            out[h * dim + i + dim // 2] = x0 * sin_val + x1 * cos_val
    
    return out

def save_operator(name: str, tensor: np.ndarray, out_dir: Path):
    tensor = tensor.astype(np.float32)
    tensor_bytes = tensor.tobytes()
    
    # Save tensor
    (out_dir / f"{name}.bin").write_bytes(tensor_bytes)
    
    # Save metadata
    meta = {
        "operator_id": name,
        "layer_id": 0,
        "tensor_shape": list(tensor.shape),
        "dtype": "f32",
        "sha256": sha256_bytes(tensor_bytes),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std())
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))

def main():
    golden_dir = Path("golden/block0_operators")
    golden_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Q/K projections
    q_proj = np.frombuffer((golden_dir / "02_q_proj.bin").read_bytes(), dtype=np.float32)
    k_proj = np.frombuffer((golden_dir / "03_k_proj.bin").read_bytes(), dtype=np.float32)
    
    print(f"Q shape: {q_proj.shape}")  # Should be [896]
    print(f"K shape: {k_proj.shape}")  # Should be [128]
    
    head_dim = 64
    n_head = 14
    
    # Reshape for RoPE: Q -> [14, 64], K -> [2, 64]
    q_reshaped = q_proj.reshape(n_head, head_dim)
    k_reshaped = k_proj.reshape(2, head_dim)
    
    # Apply RoPE at position 0
    q_rope = apply_rope(q_proj, pos=0, dim=head_dim)
    k_rope = apply_rope(k_proj, pos=0, dim=head_dim)
    
    print(f"\nQ rope shape: {q_rope.shape}")
    print(f"K rope shape: {k_rope.shape}")
    
    # Save
    save_operator("05_rope_q", q_rope, golden_dir)
    save_operator("06_rope_k", k_rope, golden_dir)
    
    print("\nSaved 05_rope_q.bin, 06_rope_k.bin")

if __name__ == "__main__":
    main()