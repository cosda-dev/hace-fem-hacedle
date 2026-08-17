#!/usr/bin/env python3
"""Trace exactly like gguf-py dequantize_blocks"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')

raw = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.raw').read_bytes()
ref_bytes = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.bin').read_bytes()
ref = np.frombuffer(ref_bytes, dtype=np.float32)

# Match gguf-py exactly
blocks = np.frombuffer(raw, dtype=np.uint8).reshape(1, 22)
n_blocks = blocks.shape[0]

# np.hsplit
d, rest = np.hsplit(blocks, [2])
qh, qs = np.hsplit(rest, [4])

print(f"d shape: {d.shape}, values: {d}")
print(f"qh shape: {qh.shape}, values: {qh}")
print(f"qs shape: {qs.shape}, values: {qs}")

d = d.view(np.float16).astype(np.float32)[0, 0]
qh = qh.view(np.uint32)[0, 0]

print(f"\nd float32: {d}")
print(f"qh uint32: {qh}, bin: {bin(qh)}")

# qh shift - this is the key!
qh_shifted = np.array([i for i in range(32)], dtype=np.uint32)
qh_bits = (qh.reshape(1) >> qh_shifted.reshape(1, 32)) & 1
print(f"\nqh_bits shape: {qh_bits.shape}")
print(f"qh_bits: {qh_bits}")

# ql extraction
ql = qs.reshape(n_blocks, -1, 1, 16) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
ql = (ql & 0x0F).reshape(n_blocks, -1)
print(f"\nql shape: {ql.shape}")
print(f"ql: {ql}")

# Combine
combined = (ql | (qh_bits & 1) << 4).astype(np.int8) - 16
print(f"\ncombined shape: {combined.shape}")
print(f"combined: {combined}")

result = d * combined
print(f"\nresult: {result}")

errors = np.abs(result - ref)
print(f"\nErrors: max={errors.max()}, mean={errors.mean()}")
print(f"Status: {'PASS' if errors.max() < 1e-6 else 'FAIL'}")