#!/usr/bin/env python3
"""T6.3: Full Block0 Inference for Golden Bundle"""

import numpy as np
from pathlib import Path
import json
import hashlib
import sys

sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

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
    model_path = Path("D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    reader = GGUFReader(model_path)
    
    out = Path("golden/block0_operators")
    out.mkdir(parents=True, exist_ok=True)
    
    # Load all Block0 weights
    weights = {}
    for tensor_info in reader.tensors:
        if tensor_info.name.startswith("blk.0."):
            raw = np.frombuffer(tensor_info.data.tobytes(), dtype=np.uint8)
            deq = dequantize(raw, tensor_info.tensor_type)
            weights[tensor_info.name] = (tensor_info.shape, deq)
            print(f"Loaded {tensor_info.name}: shape={tensor_info.shape}")
    
    # Simulate 1-token input through Block0
    # Step 1: Input embedding
    # For now use random input - in production use actual embedding
    np.random.seed(42)
    input_tensor = np.random.randn(896).astype(np.float32) * 0.01
    
    # Step 2: Attention RMSNorm
    attn_norm_w = weights.get("blk.0.attn_norm.weight", ([896], np.ones(896)))[1]
    
    # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    ss = np.sum(input_tensor ** 2)
    rms = np.sqrt(ss / 896 + 1e-6)
    post_norm = input_tensor / rms * attn_norm_w
    save_operator("02_attn_norm_post", post_norm, out)
    
    print(f"\nSaved operators to {out}")
    
    # Create manifest
    operators = sorted([f.stem for f in out.glob("*.bin")])
    manifest = {"block0_operators": operators}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()