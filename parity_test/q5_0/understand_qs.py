#!/usr/bin/env python3
"""Understand Q5_0 qs layout from quantize logic"""

import numpy as np
from pathlib import Path

raw = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.raw').read_bytes()
ref_bytes = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.bin').read_bytes()
ref = np.frombuffer(ref_bytes, dtype=np.float32)

d = np.frombuffer(raw[0:2], dtype='<f2')[0].astype(np.float32)
qh = int.from_bytes(raw[2:6], 'little')
qs = np.frombuffer(raw[6:22], dtype=np.uint8)

print(f"d = {d}")
print(f"\nqs bytes: {qs}")

# From quantize:
# qs = q.reshape((n_blocks, 2, cls.block_size // 2))
# qs = (qs[..., 0, :] & 0x0F) | (qs[..., 1, :] << 4)
# 
# This means: for 32 values, qs is reshaped to (2, 16)
# qs[0, 0..15] contains low nibbles
# qs[1, 0..15] contains high nibbles, then combined

# So actual layout:
# qs[0:16] has low nibbles of values 0-15 and 16-31? No wait...

# Let me think again. qs is 16 bytes total.
# After reshape (2, 16), we get:
# qs[..., 0, :] = 16 bytes (low nibbles)
# qs[..., 1, :] = 16 bytes (high nibbles)

# But that's 32 bytes total, not 16!
# The trick is: | combines them into single byte per pair

# So qs layout is actually INTERLEAVED:
# For j in 0..15:
#   qs[j] = low_nibble(j) | (low_nibble(j+16) << 4)
# Wait no, that's still wrong

# Let me try: each qs byte contains 2 values
# qs[0] = nibble(j=0) | (nibble(j=1) << 4) 
# qs[1] = nibble(j=2) | (nibble(j=3) << 4)
# ...

print("\n=== Trying interleaved interpretation ===")
for j in range(16):
    combined = qs[j]
    low_j = combined & 0x0F      # for position j*2
    high_j = (combined >> 4) & 0x0F  # for position j*2+1
    print(f"qs[{j}] = {qs[j]:02x}: pos {j*2}={low_j}, pos {j*2+1}={high_j}")