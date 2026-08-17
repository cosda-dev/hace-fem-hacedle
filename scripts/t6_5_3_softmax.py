#!/usr/bin/env python3
"""T6.5.3: Softmax Closure - Compute attention weights"""

import numpy as np
from pathlib import Path
import json

def save_operator(name: str, tensor: np.ndarray, out_dir: Path, extra_meta=None):
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
    if extra_meta:
        meta.update(extra_meta)
    
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))

def main():
    golden_dir = Path("golden/block0_operators")
    
    # Load attention scores
    scores = np.frombuffer((golden_dir / "07_attention_scores.bin").read_bytes(), dtype=np.float32)
    scores = scores.reshape(14, 14)  # [n_head, n_head]
    
    # Apply softmax to each row
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    softmax_out = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    # Verify softmax invariant
    row_sums = softmax_out.sum(axis=1)
    print(f"Softmax row sums (should be ~1.0): {row_sums[:5]}")
    print(f"Max deviation from 1.0: {np.abs(row_sums - 1.0).max():.10e}")
    
    # Check entropy (optional)
    entropy = -np.sum(softmax_out * np.log(softmax_out + 1e-10), axis=1)
    print(f"Entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")
    
    # Save
    softmax_flat = softmax_out.flatten()
    extra = {
        "stats": {
            "sum_max_deviation": float(np.abs(row_sums - 1.0).max()),
            "entropy_mean": float(entropy.mean()),
            "entropy_std": float(entropy.std())
        }
    }
    save_operator("08_softmax", softmax_flat, golden_dir, extra)
    
    print(f"Saved 08_softmax.bin, 08_softmax.json")

if __name__ == "__main__":
    main()