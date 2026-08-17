#!/usr/bin/env python3
"""
Generate dequant reference using gguf-py (verified against llama.cpp)
Export f32 values for parity testing
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')

from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

def main():
    model_path = "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    output_dir = Path("golden/qwen2505b/block0")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reader = GGUFReader(model_path)
    
    # Key block0 tensors with Q5_0
    tensors_to_export = [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ]
    
    print("Exporting dequant references...")
    
    for target_name in tensors_to_export:
        for tensor_info in reader.tensors:
            if tensor_info.name == target_name:
                print(f"\n{target_name}:")
                print(f"  Shape: {tensor_info.shape}")
                print(f"  Type: {tensor_info.tensor_type.name}")
                
                try:
                    # Use gguf-py's dequantize (verified against llama.cpp)
                    f32_data = dequantize(tensor_info.data, tensor_info.tensor_type)
                    
                    # Save as binary for Rust comparison
                    safe_name = target_name.replace(".", "_")
                    f32_data.astype(np.float32).tofile(output_dir / f"{safe_name}.bin")
                    
                    # Print stats
                    print(f"  Elements: {tensor_info.n_elements}")
                    print(f"  Saved: {safe_name}.bin")
                    print(f"  Stats: min={f32_data.min():.4f}, max={f32_data.max():.4f}, mean={f32_data.mean():.4f}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                break

if __name__ == "__main__":
    main()