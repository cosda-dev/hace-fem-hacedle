#!/usr/bin/env python3
"""Export raw tensor weights (dequantized) for reference"""

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
    
    # Qwen2.5-0.5B specs
    hidden_size = 896
    n_head = 14
    n_kv_head = 2
    head_dim = 64
    intermediate_size = 4864
    
    # Load and export weight tensors
    tensors_to_export = [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight", 
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ]
    
    for target_name in tensors_to_export:
        for tensor_info in reader.tensors:
            if tensor_info.name == target_name:
                print(f"Exporting {target_name}...")
                data = tensor_info.data
                
                if tensor_info.tensor_type.value > 8:
                    f32_data = dequantize(data, tensor_info.tensor_type)
                else:
                    f32_data = data.astype(np.float32).flatten()
                
                # Shape from tensor info
                shape = [int(x) for x in tensor_info.shape]
                expected_size = np.prod(shape)
                
                # Handle shape mismatches
                if len(f32_data) != expected_size:
                    print(f"  Warning: size mismatch {len(f32_data)} vs {expected_size}")
                    f32_data = f32_data.flatten()[:expected_size]
                
                f32_data = f32_data.reshape(shape)
                
                # Save as raw binary
                safe_name = target_name.replace(".", "_")
                f32_data.astype(np.float32).tofile(output_dir / f"{safe_name}.bin")
                
                # Save stats
                np.save(output_dir / f"{safe_name}.npy", f32_data)
                break
    
    print("Done")

if __name__ == "__main__":
    main()