#!/usr/bin/env python3
"""T2: Q5_0 Manual verification"""

import struct
import numpy as np
from pathlib import Path

def f16_to_f32(h: int) -> float:
    h_bytes = h.to_bytes(2, 'little')
    arr = np.frombuffer(h_bytes, dtype='<f2')[0]
    return float(arr.astype(np.float32))

def dequant_q5_0_manual(raw_bytes: bytes) -> list:
    """Q5_0: d(2) + qh(4) + qs(16) = 22 bytes per block, 32 values
    Key insight: each qs byte contributes to 2 consecutive positions via broadcast shift
    """
    d = f16_to_f32(int.from_bytes(raw_bytes[0:2], 'little'))
    
    qh_full = int.from_bytes(raw_bytes[2:6], 'little')
    qs = np.frombuffer(raw_bytes[6:22], dtype=np.uint8)
    
    # Broadcast shift: each qs byte shifted by both 0 and 4
    ql = (qs.reshape(16) & 0x0F).astype(np.int8)  # low nibble for all 16 positions
    
    # Actually need to interleave: positions 0-15 use low nibble, positions 16-31 use high nibble
    values = []
    for j in range(32):
        qh_bit = (qh_full >> j) & 1
        
        # For j in 0..15: use low nibble of qs[j]
        # For j in 16..31: use high nibble of qs[j-16]
        qs_idx = j if j < 16 else j - 16
        if j < 16:
            ql = qs[qs_idx] & 0x0F
        else:
            ql = (qs[qs_idx] >> 4) & 0x0F
        
        combined = ql + (qh_bit << 4)
        values.append(d * (combined - 16))
    
    return values

def main():
    parity_dir = Path("parity_test/q5_0")
    raw_path = parity_dir / "blk0_attn_q_q5_0_first_block.raw"
    ref_path = parity_dir / "blk0_attn_q_q5_0_first_block.bin"
    
    raw = raw_path.read_bytes()
    ref_bytes = ref_path.read_bytes()
    ref_values = [struct.unpack('<f', ref_bytes[i*4:(i+1)*4])[0] for i in range(32)]
    
    manual_values = dequant_q5_0_manual(raw)
    
    errors = [abs(m - r) for m, r in zip(manual_values, ref_values)]
    max_err = max(errors)
    mean_err = sum(errors) / len(errors)
    
    dot = sum(m * r for m, r in zip(manual_values, ref_values))
    norm_m = sum(v**2 for v in manual_values) ** 0.5
    norm_r = sum(v**2 for v in ref_values) ** 0.5
    cos_sim = dot / (norm_m * norm_r) if norm_m > 0 and norm_r > 0 else 1.0
    
    report = f"""Q5_0 Manual Verification Report
================================
Raw bytes: {len(raw)} (expected 22)
Reference values: {len(ref_values)}

max_abs_error: {max_err:.15e}
mean_abs_error: {mean_err:.15e}
cosine_similarity: {cos_sim:.15f}

Status: {'PASS' if max_err < 1e-6 else 'FAIL'}

First 8 manual: {[f'{v:.8f}' for v in manual_values[:8]]}
First 8 ref:    {[f'{v:.8f}' for v in ref_values[:8]]}
"""
    
    print(report)
    
    (parity_dir / "manual_verify_report.txt").write_text(report)
    
    import json
    json_report = {
        "tensor": "blk.0.attn_q.weight",
        "quant": "Q5_0",
        "block_index": 0,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "cosine_similarity": cos_sim,
        "status": "PASS" if max_err < 1e-6 else "FAIL"
    }
    (parity_dir / "manual_verify_report.json").write_text(json.dumps(json_report, indent=2))
    
    return max_err < 1e-6

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)