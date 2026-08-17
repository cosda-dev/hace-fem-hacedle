#!/usr/bin/env python3
"""T2: Q5_0 Bit Exact - Extract first block for parity test"""

import json
import numpy as np
from pathlib import Path

def main():
    try:
        import sys
        sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')
        from gguf.gguf_reader import GGUFReader
        from gguf.quants import dequantize
    except ImportError:
        print("ERROR: gguf-py not found")
        return
    
    model_path = Path("D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    if not model_path.exists():
        print(f"ERROR: Model not found")
        return
    
    reader = GGUFReader(model_path)
    
    # blk.0.attn_q.weight is typically Q5_0 in Qwen2.5
    target_name = "blk.0.attn_q.weight"
    
    out_dir = Path("parity_test/q5_0")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for tensor_info in reader.tensors:
        if tensor_info.name == target_name:
            print(f"Found {target_name}")
            print(f"  Shape: {tensor_info.shape}")
            print(f"  Type: {tensor_info.tensor_type.name}")
            
            raw_data = np.frombuffer(tensor_info.data.tobytes(), dtype=np.uint8)
            f32_ref = dequantize(raw_data, tensor_info.tensor_type)
            
            # Q5_0: 32 elements per block, 22 bytes
            first_block_raw = raw_data[:22]
            
            with open(out_dir / "blk0_attn_q_q5_0_first_block.raw", "wb") as f:
                f.write(first_block_raw.tobytes())
            
            first_block_f32 = f32_ref[:32]
            with open(out_dir / "blk0_attn_q_q5_0_first_block.bin", "wb") as f:
                f.write(first_block_f32.astype(np.float32).tobytes())
            
            # Q5_0 manual dequant
            d = np.frombuffer(first_block_raw[:2].tobytes(), dtype='<f2')[0].astype(np.float32)
            qh = int.from_bytes(first_block_raw[2:6].tobytes(), 'little')
            qs = first_block_raw[6:]
            
            print(f"\nQ5_0 Block layout:")
            print(f"  d (f16 scale): {d}")
            print(f"  qh (4 bytes): {qh:032b}")
            print(f"  qs (first 8): {list(qs[:8])}")
            
            report = {
                "tensor": target_name,
                "quant": "Q5_0",
                "block_bytes": 22,
                "block_elements": 32,
                "d": float(d),
                "qh_bits": bin(qh),
                "reference_stats": {
                    "min": float(first_block_f32.min()),
                    "max": float(first_block_f32.max()),
                    "mean": float(first_block_f32.mean())
                }
            }
            
            with open(out_dir / "q5_0_block_info.json", "w") as f:
                json.dump(report, f, indent=2)
            
            print(f"\nQ5_0 First Block Reference Stats:")
            print(f"  min: {first_block_f32.min():.6f}")
            print(f"  max: {first_block_f32.max():.6f}")
            print(f"  mean: {first_block_f32.mean():.6f}")
            return
    
    print(f"Tensor {target_name} not found in model")

if __name__ == "__main__":
    main()