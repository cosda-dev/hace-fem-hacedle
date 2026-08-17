#!/usr/bin/env python3
"""Q5_0 Bit-exact Parity Test - Compare Rust vs gguf-py reference"""

import numpy as np
from pathlib import Path
import json

sys_path = 'D:/host/llama.cpp/gguf-py'
import sys
sys.path.insert(0, sys_path)

from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

def main():
    model_path = "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    reader = GGUFReader(model_path)
    
    # Extract Q5_0 tensor
    for tensor_info in reader.tensors:
        if tensor_info.name == "blk.0.attn_q.weight":
            print("Found Q5_0 tensor")
            
            # Reference dequant
            ref = dequantize(tensor_info.data, tensor_info.tensor_type).flatten()
            
            # Test statistics
            print(f"Reference stats: min={ref.min():.6f}, max={ref.max():.6f}, mean={ref.mean():.6f}")
            print(f"Total elements: {len(ref)}")
            
            # Save reference for Rust test
            output_dir = Path("parity_test")
            output_dir.mkdir(exist_ok=True)
            
            # Raw quantized block
            ref.astype(np.float32).tofile(output_dir / "q5_0_reference.bin")
            
            # Random sample comparison
            np.random.seed(42)
            indices = np.random.choice(len(ref), size=1000, replace=False)
            sample = ref[indices]
            
            sample.tofile(output_dir / "q5_0_sample_ref.bin")
            
            with open(output_dir / "q5_0_indices.json", "w") as f:
                json.dump({"indices": indices.tolist()}, f)
            
            print(f"Saved 1000 sample reference values")
            break

if __name__ == "__main__":
    main()