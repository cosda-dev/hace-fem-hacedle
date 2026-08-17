#!/usr/bin/env python3
"""Final Q5_0 verification - copy exact gguf-py logic"""

import struct
import numpy as np
from pathlib import Path

raw = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.raw').read_bytes()
ref_bytes = Path('parity_test/q5_0/blk0_attn_q_q5_0_first_block.bin').read_bytes()
ref = np.frombuffer(ref_bytes, dtype=np.float32)

# Exact gguf-py logic
blocks = np.frombuffer(raw, dtype=np.uint8).reshape(1, 22)

d, rest = np.hsplit(blocks, [2])
qh, qs = np.hsplit(rest, [4])

d_f32 = d.view(np.float16).astype(np.float32)[0, 0]
qh_u32 = qh.view(np.uint32)[0, 0]

# qh shift
qh_bits = ((qh_u32.reshape(1) >> np.array([i for i in range(32)], dtype=np.uint32).reshape(1, 32)) & 1).astype(np.uint8)

# ql extraction - this is the tricky part
ql = (qs.reshape(1, 16) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)
ql = ql.reshape(1, 32)

# Combine
combined = (ql | (qh_bits << np.uint8(4))) - np.int8(16)

# Result
result = d_f32 * combined.astype(np.float32)

print("Q5_0 Final Verification")
print("=" * 40)
print(f"Result (first 8): {result[0, :8]}")
print(f"Reference (first 8): {ref[:8]}")

errors = np.abs(result - ref)
max_err = float(errors.max())
mean_err = float(errors.mean())

print(f"\nmax_abs_error: {max_err:.15e}")
print(f"mean_abs_error: {mean_err:.15e}")
print(f"Status: {'PASS' if max_err < 1e-6 else 'FAIL'}")

# Save report
from pathlib import Path
import json
report = {
    "tensor": "blk.0.attn_q.weight",
    "quant": "Q5_0",
    "block_index": 0,
    "max_abs_error": max_err,
    "mean_abs_error": mean_err,
    "cosine_similarity": float(np.dot(result, ref) / (np.linalg.norm(result) * np.linalg.norm(ref))),
    "status": "PASS" if max_err < 1e-6 else "FAIL"
}
Path('parity_test/q5_0/manual_verify_report.json').write_text(json.dumps(report, indent=2))