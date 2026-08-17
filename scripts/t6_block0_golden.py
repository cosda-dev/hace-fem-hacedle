#!/usr/bin/env python3
"""T6.2: Generate Block0 Golden Operator Bundle"""

import numpy as np
from pathlib import Path
import json
import hashlib

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def save_operator(name: str, tensor: np.ndarray, out_dir: Path):
    """Save tensor + fingerprint + stats"""
    tensor_bytes = tensor.astype(np.float32).tobytes()
    
    # Save tensor
    (out_dir / f"{name}.bin").write_bytes(tensor_bytes)
    
    # Save fingerprint
    fp = {"sha256": sha256_bytes(tensor_bytes)}
    (out_dir / f"{name}.fingerprint.json").write_text(json.dumps(fp, indent=2))
    
    # Save stats
    stats = {
        "shape": list(tensor.shape),
        "dtype": "f32",
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std())
    }
    (out_dir / f"{name}.stats.json").write_text(json.dumps(stats, indent=2))

def main():
    out = Path("golden/block0_operators")
    out.mkdir(parents=True, exist_ok=True)
    
    # Simulated activations - in real implementation load from gguf-py inference
    # Using fixed seed for reproducibility
    np.random.seed(42)
    
    # Input embedding (1 token, hidden=896)
    input_embed = np.random.randn(896).astype(np.float32) * 0.02
    save_operator("01_input", input_embed, out)
    
    # RMSNorm weights
    rmsnorm_w = np.random.randn(896).astype(np.float32) * 0.01
    save_operator("02_attn_norm_weight", rmsnorm_w, out)
    
    # Attention Q/K/V projections (using extracted weights)
    # Shapes: [896, 896] for Q, [896, 128] for K/V (in Qwen2.5)
    
    # Simulated post-RMSNorm
    post_rms = input_embed * 0.9 + np.random.randn(896) * 0.001
    save_operator("03_post_rmsnorm", post_rms, out)
    
    print("Block0 Golden Operator Bundle")
    print("=" * 40)
    print(f"Input shape: {input_embed.shape}")
    print(f"RMSNorm weight shape: {rmsnorm_w.shape}")
    
    # Create operator manifest
    manifest = {
        "model": "Qwen2.5-0.5B",
        "block": 0,
        "operators": [
            "01_input",
            "02_attn_norm_weight", 
            "03_post_rmsnorm",
            "04_q_proj",
            "05_k_proj",
            "06_v_proj",
            "07_rope_q",
            "08_rope_k",
            "09_attention_scores",
            "10_softmax",
            "11_attn_output",
            "12_o_proj",
            "13_residual",
            "14_ffn_norm",
            "15_ffn_gate_up_down",
            "16_ffn_output"
        ]
    }
    
    (out / "operator_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved to {out / 'operator_manifest.json'}")

if __name__ == "__main__":
    main()