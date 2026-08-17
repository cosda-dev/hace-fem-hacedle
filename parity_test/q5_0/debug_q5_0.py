#!/usr/bin/env python3
"""Debug Q5_0 extraction"""

import struct
import numpy as np
from pathlib import Path

raw = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.raw').read_bytes()
ref_bytes = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.bin').read_bytes()
ref_values = [struct.unpack('<f', ref_bytes[i*4:(i+1)*4])[0] for i in range(32)]

print("=== Q5_0 Debug ===\n")

# Extract components
d = np.frombuffer(raw[0:2], dtype='<f2')[0].astype(np.float32)
print(f"d (f16 scale): {d}")

qh_full = int.from_bytes(raw[2:6], 'little')
print(f"qh (uint32): {qh_full} = {bin(qh_full)}")

qs = raw[6:]
print(f"qs bytes: {[hex(b) for b in qs]}")

print("\n=== Element-by-element ===")
print("j | qh_bit | qs_byte | ql_nibble | q_combined | result")
results = []
for j in range(32):
    qh_bit = (qh_full >> j) & 1
    qs_byte_idx = j // 2
    qs_byte = qs[qs_byte_idx]
    ql = qs_byte & 0x0F if j % 2 == 0 else (qs_byte >> 4)
    q = ql + (qh_bit << 4)
    result = (q - 16.0) * d
    results.append(result)
    print(f"{j:2d} | {qh_bit:7d} | {hex(qs_byte):6s} | {ql:11d} | {q:11d} | {result:.8f}")

print(f"\nReference: {[f'{v:.8f}' for v in ref_values[:8]]}")
print(f"Manual:    {[f'{v:.8f}' for v in results[:8]]}")

# Try alternative: what if d should be positive?
print("\n=== Try absolute d ===")
d_abs = abs(d)
results_abs = [(ql + ((qh_full >> j) & 1) * 16 - 16.0) * d_abs for j in range(32) for ql in [qs[j//2] & 0x0F if j%2==0 else (qs[j//2] >> 4)]]

# Actually check formula from gguf-py quants.py
print("\n=== Using gguf-py directly ===")
try:
    import sys
    sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')
    from gguf.quants import dequantize
    raw_arr = np.frombuffer(raw, dtype=np.uint8)
    # This gives us 32 values
    direct = dequantize(raw_arr, 12)  # GGMLQuantizationType.Q5_0 = 12
    print(f"Direct gguf-py: {direct[:8]}")
    print(f"Reference:      {ref_values[:8]}")
except Exception as e:
    print(f"Error: {e}")