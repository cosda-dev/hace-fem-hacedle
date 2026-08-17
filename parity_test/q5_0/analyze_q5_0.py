#!/usr/bin/env python3
"""Analyze Q5_0 format - understand reshape behavior"""

import numpy as np
from pathlib import Path

raw = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.raw').read_bytes()
qs = np.frombuffer(raw[6:22], dtype=np.uint8)

print("qs bytes:", qs)
print("qs shape:", qs.shape)

# gguf-py does: ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> np.array([0, 4], ...)
# block_size = 32, block_size // 2 = 16
# For single block (n_blocks=1): reshape to (1, -1, 1, 16) = (1, 1, 1, 16) hoặc (1, 1, 1, 16)?
# qs has 16 elements, so reshape to (1, 1, 1, 16)

# Try reshape
ql = qs.reshape(1, 1, 1, 16).view(np.uint8)  # (1, 1, 1, 16)
print("After reshape:", ql.shape, ql.view())

# Shift by [0, 4] - this creates 2 copies
shift_arr = np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
ql_shifted = ql >> shift_arr
print("After shift:", ql_shifted.shape)

# Wait, this doesn't match because 16 bytes != 32 values
# Let me check actual layout from gguf-py source

# From source:
# qs = q.reshape((n_blocks, 2, cls.block_size // 2)) = (1, 2, 16)
# qs = (qs[..., 0, :] & 0x0F) | (qs[..., 1, :] << 4)

# This means qs is reorganized BEFORE this step!
# Actually quantize does: qs = q.reshape(...).view + packing

print("\n=== Correct interpretation ===")
# qs contains packed 4-bit values for 32 elements
# Each byte has 2 nibbles, so 16 bytes = 32 values

for j in range(32):
    byte_idx = j // 2
    if j % 2 == 0:
        nibble = qs[byte_idx] & 0x0F
    else:
        nibble = (qs[byte_idx] >> 4) & 0x0F
    print(f"j={j}: byte={qs[byte_idx]:02x}, nibble={nibble:04b}")