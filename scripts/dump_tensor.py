#!/usr/bin/env python3
"""
Dump GGUF tensor data using llama.cpp for parity comparison.
Usage: python dump_tensor.py <model.gguf> <tensor_name> <output.bin>
"""
import sys
import struct
import numpy as np

def print_usage():
    print("Usage: python dump_tensor.py <model.gguf> <tensor_name> <output.bin>")
    print("Example: python dump_tensor.py Qwen2.5-0.5B-Instruct-Q4_K_M.gguf blk.0.attn_q.weight output.bin")
    sys.exit(1)

def main():
    if len(sys.argv) != 4:
        print_usage()
    
    model_path = sys.argv[1]
    tensor_name = sys.argv[2]
    output_path = sys.argv[3]
    
    print(f"Loading model: {model_path}")
    print(f"Target tensor: {tensor_name}")
    
    try:
        import gguf
    except ImportError:
        print("ERROR: gguf package not installed. Install with: pip install gguf")
        sys.exit(1)
    
    try:
        reader = gguf.GGUFReader(model_path)
    except Exception as e:
        print(f"ERROR: Cannot load model: {e}")
        sys.exit(1)
    
    tensors = reader.tensors
    for name in tensors:
        if tensor_name in name:
            data = reader.tensor(name)
            print(f"Found tensor: {name}")
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"First 10 values: {data.flatten()[:10]}")
            
            np.save(output_path, data)
            print(f"Saved to: {output_path}")
            return
    
    print(f"ERROR: Tensor {tensor_name} not found")

if __name__ == "__main__":
    main()