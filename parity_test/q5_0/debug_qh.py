#!/usr/bin/env python3
"""Debug Q5_0 qh bit extraction"""

import numpy as np
from pathlib import Path

raw = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.raw').read_bytes()
ref_bytes = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.bin').read_bytes()
ref = np.frombuffer(ref_bytes, dtype=np.float32)

d = np.frombuffer(raw[0:2], dtype='<f2')[0].astype(np.float32)
qh = int.from_bytes(raw[2:6], 'little')
qs = np.frombuffer(raw[6:22], dtype=np.uint8)

print(f"d = {d}")
print(f"qh bits: {bin(qh)}")

# From gguf-py: qh = np.packbits(q.reshape((1,32)) >> 4, bitorder="little")
# So qh bit j corresponds to bit 4 of original q value at position j
# This means qh gives us the high bit (bit 4) of the 5-bit value

# Try: for each position j, the 5-bit value is:
# bits 0-3 from qs nibble, bit 4 from qh bit

print("\nElement analysis:")
for j in range(32):
    # qs nibble (bits 0-3)
    byte_idx = j // 2
    if j % 2 == 0:
        ql = qs[byte_idx] & 0x0F
    else:
        ql = (qs[byte_idx] >> 4) & 0x0F
    
    # qh bit (bit 4)
    qh_bit = (qh >> j) & 1
    
    # Combined 5-bit value (0-31)
    q = ql + (qh_bit << 4)
    
    # Center at -16 to 15
    val = (q - 16) * d
    
    print(f"j={j:2d}: qs_nibble={ql:2d}, qh_bit={qh_bit}, q={q:2d}, val={val:12.8f}, ref={ref[j]:12.8f}, match={abs(val-ref[j])<1e-10}")