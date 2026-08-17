#!/usr/bin/env python3
"""T6.5.4: Attention Output Closure - weighted sum V"""

import numpy as np
from pathlib import Path
import json

def save_operator(name: str, tensor: np.ndarray, out_dir: Path):
    tensor = tensor.astype(np.float32)
    tensor_bytes = tensor.tobytes()
    
    (out_dir / f"{name}.bin").write_bytes(tensor_bytes)
    
    meta = {
        "operator_id": name,
        "layer_id": 0,
        "tensor_shape": list(tensor.shape),
        "dtype": "f32",
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))

def main():
    golden_dir = Path("golden/block0_operators")
    
    # Load softmax and V projections
    softmax = np.frombuffer((golden_dir / "08_softmax.bin").read_bytes(), dtype=np.float32)
    v_proj = np.frombuffer((golden_dir / "04_v_proj.bin").read_bytes(), dtype=np.float32)
    
    n_head = 14
    head_dim = 64
    n_head_kv = 2
    
    softmax = softmax.reshape(n_head, n_head)  # [14, 14]
    v = v_proj.reshape(n_head_kv, head_dim)  # [2, 64]
    
    # GQA expand V
    v_expanded = np.zeros((n_head, head_dim), dtype=np.float32)
    for i in range(n_head):
        kv_idx = i // (n_head // n_head_kv)
        v_expanded[i] = v[kv_idx]
    
    # Weighted sum: output[i] = sum_j softmax[i,j] * v[j]
    attn_output = softmax @ v_expanded  # [14, 64]
    
    print(f"Attention output shape: {attn_output.shape}")
    print(f"Output stats: min={attn_output.min():.6f}, max={attn_output.max():.6f}")
    
    save_operator("09_attention_output", attn_output.flatten(), golden_dir)
    print("Saved 09_attention_output.bin")

if __name__ == "__main__":
    main()