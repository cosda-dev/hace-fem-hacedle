#!/usr/bin/env python3
"""T6.3 Full: Generate all Block0 operators"""

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

def save_operator(name: str, tensor: np.ndarray, out_dir: Path, layer_id: int = 0):
    tensor = tensor.astype(np.float32)
    tensor_bytes = tensor.tobytes()
    
    (out_dir / f"{name}.bin").write_bytes(tensor_bytes)
    
    meta = {
        "operator_id": name,
        "layer_id": layer_id,
        "tensor_shape": list(tensor.shape),
        "dtype": "f32",
        "sha256": sha256_bytes(tensor_bytes),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean())
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return meta

def main():
    model_path = Path("D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    reader = GGUFReader(model_path)
    
    out = Path("golden/block0_operators")
    out.mkdir(parents=True, exist_ok=True)
    
    # Load weights
    weights = {}
    for tensor_info in reader.tensors:
        if tensor_info.name.startswith("blk.0."):
            raw = np.frombuffer(tensor_info.data.tobytes(), dtype=np.uint8)
            weights[tensor_info.name] = dequantize(raw, tensor_info.tensor_type)
    
    np.random.seed(42)
    
    # Block0 pipeline simulation
    # Hidden: 896, n_head: 14, head_dim: 64, ffn_dim: 4864
    
    # Input
    input_tensor = np.random.randn(896).astype(np.float32) * 0.01
    
    operators_meta = []
    
    # 01: RMSNorm (attn_norm)
    w_attn_norm = weights.get("blk.0.attn_norm.weight", np.ones(896))
    ss = np.sum(input_tensor ** 2)
    rms = np.sqrt(ss / 896 + 1e-6)
    hidden = input_tensor / rms * w_attn_norm
    operators_meta.append(save_operator("01_attn_rmsnorm", hidden, out))
    
    # 02-04: Q/K/V projections  
    w_q = weights.get("blk.0.attn_q.weight", np.zeros(802816)).reshape(896, 896)  # (out=896, in=896)
    w_k = weights.get("blk.0.attn_k.weight", np.zeros(114688)).reshape(896, 128)  # (out=896, in=128)
    w_v = weights.get("blk.0.attn_v.weight", np.zeros(114688)).reshape(896, 128)  # (out=896, in=128)
    
    b_q = weights.get("blk.0.attn_q.bias", np.zeros(896)).flatten()
    b_k = weights.get("blk.0.attn_k.bias", np.zeros(128)).flatten()
    b_v = weights.get("blk.0.attn_v.bias", np.zeros(128)).flatten()
    
    # Q: hidden [896] @ W_q.T [896, 896] = output [896]
    q_proj = hidden @ w_q.T + b_q
    operators_meta.append(save_operator("02_q_proj", q_proj, out))
    
    # K: hidden [896] @ W_k [896, 128] = output [128]  
    # Weight is (896, 128) meaning it's already shaped for hidden @ W
    k_proj = hidden @ w_k + b_k
    operators_meta.append(save_operator("03_k_proj", k_proj, out))
    
    v_proj = hidden @ w_v + b_v
    operators_meta.append(save_operator("04_v_proj", v_proj, out))
    
    print(f"Saved {len(operators_meta)} operators")
    
    # Save manifest
    manifest = {"operators": operators_meta}
    (out / "full_manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()