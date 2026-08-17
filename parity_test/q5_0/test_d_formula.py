#!/usr/bin/env python3
"""Test different d formulas"""

import struct
import numpy as np
from pathlib import Path

raw = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.raw').read_bytes()
ref_bytes = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.bin').read_bytes()
ref = np.frombuffer(ref_bytes, dtype=np.float32)

d = np.frombuffer(raw[0:2], dtype='<f2')[0].astype(np.float32)
qh = int.from_bytes(raw[2:6], 'little')
qs = raw[6:22]

print(f"Original d: {d}")
print(f"Abs d: {abs(d)}")

# Try both formulas
for use_abs in [False, True]:
    d_use = abs(d) if use_abs else d
    print(f"\n=== Using d = {d_use:.8f} ({'abs' if use_abs else 'original'}) ===")
    
    for sign_flip in [1, -1]:
        d_final = d_use * sign_flip
        values = []
        
        for j in range(32):
            qh_bit = (qh >> j) & 1
            qs_byte = qs[j // 2]
            if j % 2 == 0:
                ql = qs_byte & 0x0F
            else:
                ql = (qs_byte >> 4) & 0x0F
            
            q = (ql + (qh_bit << 4)) - 16
            values.append(d_final * q)
        
        errors = [abs(m - r) for m, r in zip(values, ref)]
        max_err = max(errors)
        
        if max_err < 0.01:
            print(f"  sign_flip={sign_flip}: max_err={max_err:.6e} - CLOSE!")
            if max_err < 1e-6:
                print(f"  *** PASS! ***")
                print(f"  First 8 values: {[f'{v:.8f}' for v in values[:8]]}")