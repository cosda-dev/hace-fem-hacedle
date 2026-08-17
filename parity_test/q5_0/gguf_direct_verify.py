#!/usr/bin/env python3
"""Verify Q5_0 using gguf-py directly on single block"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')
from gguf.constants import GGMLQuantizationType
from gguf.quants import Q5_0

def main():
    # Load our raw block
    raw = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.raw').read_bytes()
    raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape(1, 22)
    
    # Use gguf-py dequantize_blocks directly
    dequant = Q5_0.dequantize_blocks(raw_arr)
    result = dequant[0]
    
    # Load reference
    ref_bytes = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.bin').read_bytes()
    ref = np.frombuffer(ref_bytes, dtype=np.float32)
    
    print("gguf-py direct dequant result (first 8):", result[:8])
    print("reference (first 8):", ref[:8])
    
    errors = np.abs(result - ref)
    max_err = float(errors.max())
    mean_err = float(errors.mean())
    
    print(f"\nmax_abs_error: {max_err:.15e}")
    print(f"mean_abs_error: {mean_err:.15e}")
    print(f"Status: {'PASS' if max_err < 1e-6 else 'FAIL'}")

if __name__ == "__main__":
    main()