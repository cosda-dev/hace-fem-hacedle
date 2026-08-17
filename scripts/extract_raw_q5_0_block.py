#!/usr/bin/env python3
"""P5.3: Extract one Q5_0 block for bit-exact testing"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')

from gguf.gguf_reader import GGUFReader

def main():
    model_path = "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    output_dir = Path("parity_test")
    output_dir.mkdir(exist_ok=True)
    
    reader = GGUFReader(model_path)
    
    # Find Q5_0 tensor
    for tensor_info in reader.tensors:
        if "blk.0.attn_q.weight" in tensor_info.name and tensor_info.tensor_type.value == 12:  # Q5_0
            print(f"Found Q5_0 tensor: {tensor_info.name}")
            print(f"Shape: {tensor_info.shape}")
            print(f"Type: {tensor_info.tensor_type.name}")
            
            # Save raw quantized bytes (first block only)
            raw_bytes = tensor_info.data.tobytes()
            
            # Q5_0: 22 bytes per block, 32 elements
            # Extract first block (32 bytes as multiple of block)
            block_size = 22
            with open(output_dir / "q5_0_block.bin", "wb") as f:
                f.write(raw_bytes[:block_size])
            
            print(f"Saved first Q5_0 block: {block_size} bytes")
            
            # Also save dequantized reference
            from gguf.quants import dequantize
            f32_data = dequantize(tensor_info.data, tensor_info.tensor_type)
            f32_data[:32].astype(np.float32).tofile(output_dir / "q5_0_dequant_ref.bin")
            
            print(f"Reference values (first 32): {f32_data[:32]}")
            break

if __name__ == "__main__":
    main()