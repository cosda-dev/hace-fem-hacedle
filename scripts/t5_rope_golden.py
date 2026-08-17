#!/usr/bin/env python3
"""T5: Generate RoPE golden values from gguf-py"""

import numpy as np
from pathlib import Path
import json

def apply_rope_pairwise(x: np.ndarray, pos: int, theta: float = 1000000.0) -> np.ndarray:
    """Apply RoPE pairwise rotation - matching llama.cpp implementation"""
    dim = x.shape[-1]
    out = np.zeros_like(x)
    
    for i in range(dim // 2):
        freq = theta ** (2.0 * i / dim)
        inv_freq = 1.0 / freq
        angle = pos * inv_freq
        
        x0 = x[i]
        x1 = x[i + dim // 2]
        
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        
        out[i] = x0 * cos_val - x1 * sin_val
        out[i + dim // 2] = x0 * sin_val + x1 * cos_val
    
    return out

def main():
    # RoPE params for Qwen2.5
    hidden_size = 896
    n_head = 14
    head_dim = 64  # 896 / 14 = 64
    theta = 1000000.0
    
    # Create test input - simple sequential values
    # In real model: Q projection output
    out = Path("parity_test/rope")
    out.mkdir(parents=True, exist_ok=True)
    
    positions = [0, 1, 128, 1024]
    
    for pos in positions:
        # Create simple test vector
        x = np.arange(head_dim, dtype=np.float32)
        x = (x - head_dim/2) / head_dim  # normalize around 0
        
        x_rope = apply_rope_pairwise(x, pos, theta)
        
        # Save golden
        golden = out / f"pos{pos}.bin"
        golden.write_bytes(x_rope.tobytes())
        
        print(f"Position {pos}:")
        print(f"  Input (first 8): {x[:8]}")
        print(f"  Output (first 8): {x_rope[:8]}")
        
        # Compute inv_freq
        inv_freq_first = 1.0 / (theta ** (2.0 * 0 / head_dim))
        inv_freq_last = 1.0 / (theta ** (2.0 * 63 / head_dim))
        print(f"  inv_freq[0]: {inv_freq_first:.10e}")
        print(f"  inv_freq[63]: {inv_freq_last:.10e}")
    
    # Generate inv_freq schedule
    inv_freq = 1.0 / (theta ** (2.0 * np.arange(head_dim // 2) / head_dim))
    inv_path = out / "inv_freq.bin"
    inv_path.write_bytes(inv_freq.astype(np.float32).tobytes())
    
    # Save metadata
    meta = {
        "hidden_size": hidden_size,
        "n_head": n_head, 
        "head_dim": head_dim,
        "theta": theta,
        "positions_tested": positions,
        "inv_freq_first": float(inv_freq[0]),
        "inv_freq_last": float(inv_freq[-1])
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()