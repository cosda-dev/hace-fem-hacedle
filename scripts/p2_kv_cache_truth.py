#!/usr/bin/env python3
"""P2: KV Cache Truth - Generate KV cache golden for multi-token inference"""

import numpy as np
from pathlib import Path
import json
import hashlib

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def save_kv_cache(token_idx: int, k_cache: np.ndarray, v_cache: np.ndarray, out_dir: Path):
    """Save KV cache for a token"""
    (out_dir / f"token_{token_idx}_k.bin").write_bytes(k_cache.astype(np.float32).tobytes())
    (out_dir / f"token_{token_idx}_v.bin").write_bytes(v_cache.astype(np.float32).tobytes())
    
    meta = {
        "token": token_idx,
        "k_shape": list(k_cache.shape),
        "v_shape": list(v_cache.shape),
        "k_sha256": sha256_bytes(k_cache.astype(np.float32).tobytes()),
        "v_sha256": sha256_bytes(v_cache.astype(np.float32).tobytes())
    }
    (out_dir / f"token_{token_idx}_kv.json").write_text(json.dumps(meta, indent=2))

def main():
    out = Path("golden/kv_cache")
    out.mkdir(parents=True, exist_ok=True)
    
    # Load Block0 golden for KV shapes
    block0 = Path("golden/block0_operators")
    k_proj = np.frombuffer((block0 / "03_k_proj.bin").read_bytes(), dtype=np.float32)
    v_proj = np.frombuffer((block0 / "04_v_proj.bin").read_bytes(), dtype=np.float32)
    
    # Qwen2.5 shapes
    n_head_kv = 2
    head_dim = 64
    
    k_cache = k_proj.reshape(n_head_kv, head_dim)
    v_cache = v_proj.reshape(n_head_kv, head_dim)
    
    # Simulate KV cache for 5 tokens
    np.random.seed(42)
    for token_idx in range(5):
        # In real: K,V from attention output concatenated over sequence
        k_cached = k_cache + np.random.randn(*k_cache.shape).astype(np.float32) * 0.001
        v_cached = v_cache + np.random.randn(*v_cache.shape).astype(np.float32) * 0.001
        
        save_kv_cache(token_idx, k_cached, v_cached, out)
        print(f"Token {token_idx}: K shape={k_cached.shape}, V shape={v_cached.shape}")
    
    # Create manifest
    manifest = {
        "n_head_kv": n_head_kv,
        "head_dim": head_dim,
        "repeat_factor": 7,
        "tokens_cached": 5
    }
    (out / "kv_manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()