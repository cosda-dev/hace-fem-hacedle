#!/usr/bin/env python3
"""T1: Q8_0 Bit Exact Parity - Extract raw Q8_0 bytes and dequantize for comparison"""

import json
import struct
import numpy as np
from pathlib import Path

def main():
    try:
        import sys
        sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')
        from gguf.gguf_reader import GGUFReader
        from gguf.quants import dequantize
    except ImportError:
        print("ERROR: gguf-py not found at D:/host/llama.cpp/gguf-py")
        return
    
    model_path = Path("D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    reader = GGUFReader(model_path)
    
    # Find Q8_0 tensors (blk.0.attn_v.weight is typically Q8_0)
    target_name = "blk.0.attn_v.weight"
    
    for tensor_info in reader.tensors:
        if tensor_info.name == target_name:
            print(f"Found {target_name}")
            print(f"  Shape: {tensor_info.shape}")
            print(f"  Type: {tensor_info.tensor_type.name}")
            
            # Get raw quantized data
            raw_data = np.frombuffer(tensor_info.data.tobytes(), dtype=np.uint8)
            print(f"  Raw bytes: {len(raw_data)}")
            print(f"  n_bytes field: {tensor_info.n_bytes}")
            
            # Dequantize using gguf-py (reference)
            f32_ref = dequantize(raw_data, tensor_info.tensor_type)
            print(f"  Dequantized: {f32_ref.shape}")
            
            # Save both raw and dequantized
            out_dir = Path("parity_test/q8_0")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw bytes (first block only for bit-exact test)
            # Q8_0 block = 34 bytes (2 f16 + 32 i8)
            # Extract first complete block (34 bytes)
            first_block_raw = raw_data[:34]  # First 34 bytes = first block
            with open(out_dir / "blk0_attn_v_q8_0_first_block.raw", "wb") as f:
                f.write(first_block_raw.tobytes())
            
            # Save first block dequantized (32 f32 values)
            first_block_f32 = f32_ref[:32]
            with open(out_dir / "blk0_attn_v_q8_0_first_block.bin", "wb") as f:
                f.write(first_block_f32.astype(np.float32).tobytes())
            
            # Full tensor dequantized for comparison
            with open(out_dir / "blk0_attn_v_q8_0_full.bin", "wb") as f:
                f.write(f32_ref.astype(np.float32).tobytes())
            
            # Generate stats report
            report = {
                "tensor": "blk.0.attn_v.weight",
                "quant": "Q8_0",
                "raw_bytes_per_block": 34,
                "block_size": 32,
                "shape": [int(s) for s in tensor_info.shape],
                "total_elements": int(np.prod(tensor_info.shape)),
                "reference_stats": {
                    "min": float(f32_ref.min()),
                    "max": float(f32_ref.max()),
                    "mean": float(f32_ref.mean())
                },
                "status": "extracted"
            }
            
            with open(out_dir / "q8_0_parity_report.json", "w") as f:
                json.dump(report, f, indent=2)
            
            print("\nQ8_0 Parity Report:")
            print(json.dumps(report, indent=2))
            return
    
    print(f"Tensor {target_name} not found")

if __name__ == "__main__":
    main()