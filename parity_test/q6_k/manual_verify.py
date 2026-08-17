#!/usr/bin/env python3
"""T3: Q6_K Manual verification using gguf-py directly"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')
from gguf.quants import Q6_K

def main():
    raw_path = Path('parity_test/q6_k/blk0_ffn_down_q6_k_first_block.raw')
    ref_path = Path('parity_test/q6_k/blk0_ffn_down_q6_k_first_block.bin')
    
    raw = raw_path.read_bytes()
    ref = np.frombuffer(ref_path.read_bytes(), dtype=np.float32)
    
    # Use gguf-py directly
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(1, 210)
    result = Q6_K.dequantize_blocks(blocks)
    
    errors = np.abs(result - ref)
    max_err = float(errors.max())
    mean_err = float(errors.mean())
    
    print("Q6_K Manual Verification Report")
    print("=" * 40)
    print(f"Result (first 8): {result[0, :8]}")
    print(f"Reference (first 8): {ref[:8]}")
    print(f"max_abs_error: {max_err:.15e}")
    print(f"mean_abs_error: {mean_err:.15e}")
    print(f"Status: {'PASS' if max_err < 1e-6 else 'FAIL'}")

if __name__ == "__main__":
    main()