#!/usr/bin/env python3
"""T3: Q6_K Bit Exact - Extract first block for parity test"""

import json
import numpy as np
from pathlib import Path

try:
    import sys
    sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')
    from gguf.gguf_reader import GGUFReader
    from gguf.quants import dequantize
except ImportError:
    print("ERROR: gguf-py not found")
    exit(1)

model_path = Path("D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
if not model_path.exists():
    print(f"ERROR: Model not found")
    exit(1)

reader = GGUFReader(model_path)

# blk.0.ffn_down.weight is typically Q6_K in Qwen2.5
target_name = "blk.0.ffn_down.weight"

out_dir = Path("parity_test/q6_k")
out_dir.mkdir(parents=True, exist_ok=True)

for tensor_info in reader.tensors:
    if tensor_info.name == target_name:
        print(f"Found {target_name}")
        print(f"  Shape: {tensor_info.shape}")
        print(f"  Type: {tensor_info.tensor_type.name}")
        
        raw = np.frombuffer(tensor_info.data.tobytes(), dtype=np.uint8)
        f32_ref = dequantize(raw, tensor_info.tensor_type)
        
        # Q6_K: 256 elements per block, 210 bytes
        first_block = raw[:210]
        
        with open(out_dir / "blk0_ffn_down_q6_k_first_block.raw", "wb") as f:
            f.write(first_block.tobytes())
        
        first_block_f32 = f32_ref[:256]
        with open(out_dir / "blk0_ffn_down_q6_k_first_block.bin", "wb") as f:
            f.write(first_block_f32.astype(np.float32).tobytes())
        
        print(f"\nQ6_K First Block Reference Stats:")
        print(f"  min: {first_block_f32.min():.6f}")
        print(f"  max: {first_block_f32.max():.6f}")
        
        report = {
            "tensor": target_name,
            "quant": "Q6_K",
            "block_bytes": 210,
            "block_elements": 256,
            "reference_stats": {
                "min": float(first_block_f32.min()),
                "max": float(first_block_f32.max()),
                "mean": float(first_block_f32.mean())
            }
        }
        with open(out_dir / "q6_k_block_info.json", "w") as f:
            json.dump(report, f, indent=2)
        
        break

print(f"Tensor {target_name} not found in model")