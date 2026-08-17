#!/usr/bin/env python3
"""T6.5.5: O Projection Closure - output projection + residual"""

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
    
    # Load attention output
    attn_output = np.frombuffer((golden_dir / "09_attention_output.bin").read_bytes(), dtype=np.float32)
    
    # Load weights - attn_output.W
    reader = GGUFReader('D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf')
    
    w_o = None
    b_o_flat = np.zeros(896, dtype=np.float32)
    
    for t in reader.tensors:
        if t.name == "blk.0.attn_output.weight":
            w_o = dequantize(np.frombuffer(t.data.tobytes(), dtype=np.uint8), t.tensor_type)
        if t.name == "blk.0.attn_output.bias":
            b_o = dequantize(np.frombuffer(t.data.tobytes(), dtype=np.uint8), t.tensor_type)
            b_o_flat = b_o.flatten()
    
    if w_o is None:
        w_o = np.zeros(896 * 896, dtype=np.float32)
    
    w_o = w_o.reshape(896, 896)
    
    # O projection: attn_output @ W_o.T + bias
    o_proj = attn_output @ w_o.T + b_o_flat
    
    print(f"O projection shape: {o_proj.shape}")
    save_operator("10_o_proj", o_proj, golden_dir)
    print("Saved 10_o_proj.bin")
    
    # 11. Residual (input + o_proj)
    input_embed = np.frombuffer((golden_dir / "01_input.bin").read_bytes(), dtype=np.float32)
    if input_embed.shape != o_proj.shape:
        input_embed = np.concatenate([input_embed, np.zeros(896 - input_embed.shape[0])])
    
    residual = input_embed + o_proj
    save_operator("11_residual", residual, golden_dir)
    print("Saved 11_residual.bin")
    
    print(f"\nAttention subsystem golden complete: operators 01-11 saved")

if __name__ == "__main__":
    main()