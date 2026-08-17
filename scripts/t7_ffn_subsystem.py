#!/usr/bin/env python3
"""T7: FFN Subsystem - operators 12-18"""

import numpy as np
from pathlib import Path
import hashlib

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def save_operator(name: str, tensor: np.ndarray, out_dir: Path):
    tensor = tensor.astype(np.float32)
    tensor_bytes = tensor.tobytes()
    
    (out_dir / f"{name}.bin").write_bytes(tensor_bytes)
    
    import json
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
    
    # Load residual from attention
    residual = np.frombuffer((golden_dir / "11_residual.bin").read_bytes(), dtype=np.float32)
    
    # Load FFN weights
    import sys
    sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')
    from gguf.gguf_reader import GGUFReader
    from gguf.quants import dequantize
    
    reader = GGUFReader('D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf')
    
    # Load FFN weights
    ffn_w = {}
    for t in reader.tensors:
        if t.name.startswith("blk.0.ffn_"):
            arr = dequantize(np.frombuffer(t.data.tobytes(), dtype=np.uint8), t.tensor_type)
            ffn_w[t.name] = arr
            print(f"Loaded {t.name}: {arr.shape}")
    
    # 12. FFN RMSNorm
    ffn_norm_w = ffn_w.get("blk.0.ffn_norm.weight", np.ones(896))
    ss = np.sum(residual ** 2)
    ffn_rms = np.sqrt(ss / 896 + 1e-6)
    post_ffn_norm = residual / ffn_rms * ffn_norm_w
    save_operator("12_ffn_norm", post_ffn_norm, golden_dir)
    
    # 13. Gate projection (4864 output)
    gate_w = ffn_w.get("blk.0.ffn_gate.weight", np.zeros(4864 * 896)).reshape(4864, 896)
    gate_b = ffn_w.get("blk.0.ffn_gate.bias", np.zeros(4864))
    gate_proj = post_ffn_norm @ gate_w.T + gate_b.flatten()
    save_operator("13_gate_proj", gate_proj, golden_dir)
    
    # 14. Up projection
    up_w = ffn_w.get("blk.0.ffn_up.weight", np.zeros(4864 * 896)).reshape(4864, 896)
    up_b = ffn_w.get("blk.0.ffn_up.bias", np.zeros(4864))
    up_proj = post_ffn_norm @ up_w.T + up_b.flatten()
    save_operator("14_up_proj", up_proj, golden_dir)
    
    # 15. SiLU activation (swish)
    silu = gate_proj / (1 + np.exp(-gate_proj))  # x * sigmoid(x)
    save_operator("15_silu", silu, golden_dir)
    
    # 16. Gate * Up (elementwise)
    gate_mul_up = silu * up_proj
    save_operator("16_gate_mul_up", gate_mul_up, golden_dir)
    
    # 17. Down projection
    down_w = ffn_w.get("blk.0.ffn_down.weight", np.zeros(896 * 4864)).reshape(896, 4864)
    down_b = ffn_w.get("blk.0.ffn_down.bias", np.zeros(896))
    down_proj = gate_mul_up @ down_w.T + down_b.flatten()
    save_operator("17_down_proj", down_proj, golden_dir)
    
    # 18. FFN residual
    ffn_residual = residual + down_proj
    save_operator("18_ffn_residual", ffn_residual, golden_dir)
    
    print(f"\nFFN subsystem complete: operators 12-18 saved")

if __name__ == "__main__":
    main()