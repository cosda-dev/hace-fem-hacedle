#!/usr/bin/env python3
"""Manual Q8_0 verification - run both gguf-py and manual dequant"""

import struct
import numpy as np
from pathlib import Path

def f16_to_f32(h: int) -> float:
    """Convert f16 to f32 via numpy"""
    h_bytes = h.to_bytes(2, 'little')
    arr = np.frombuffer(h_bytes, dtype='<f2')[0]  # f2 = float16 little-endian
    return float(arr.astype(np.float32))

def dequant_q8_0_manual(raw_bytes: bytes) -> list:
    """Manual dequant - verify Rust logic"""
    d = f16_to_f32(int.from_bytes(raw_bytes[0:2], 'little'))
    values = []
    for j in range(32):
        v = int.from_bytes(raw_bytes[2+j:3+j], 'little', signed=True)
        values.append(v * d)
    return values

def main():
    parity_dir = Path("parity_test/q8_0")
    raw_path = parity_dir / "blk0_attn_v_q8_0_first_block.raw"
    ref_path = parity_dir / "blk0_attn_v_q8_0_first_block.bin"
    
    raw = raw_path.read_bytes()
    ref_bytes = ref_path.read_bytes()
    ref_values = [struct.unpack('<f', ref_bytes[i*4:(i+1)*4])[0] for i in range(32)]
    
    manual_values = dequant_q8_0_manual(raw)
    
    # Compute metrics
    errors = [abs(m - r) for m, r in zip(manual_values, ref_values)]
    max_err = max(errors)
    mean_err = sum(errors) / len(errors)
    
    # Cosine similarity
    dot = sum(m * r for m, r in zip(manual_values, ref_values))
    norm_m = sum(v**2 for v in manual_values) ** 0.5
    norm_r = sum(v**2 for v in ref_values) ** 0.5
    cos_sim = dot / (norm_m * norm_r) if norm_m > 0 and norm_r > 0 else 1.0
    
    report = f"""Q8_0 Manual Verification Report
================================
Raw bytes: {len(raw)} (expected 34)
Reference values: {len(ref_values)}
Manual values: {len(manual_values)}

max_abs_error: {max_err:.15e}
mean_abs_error: {mean_err:.15e}
cosine_similarity: {cos_sim:.15f}

Status: {'PASS' if max_err < 1e-6 else 'FAIL'}

First 8 manual: {manual_values[:8]}
First 8 ref:    {ref_values[:8]}
"""
    
    print(report)
    
    # Save report
    (parity_dir / "manual_verify_report.txt").write_text(report)
    
    # JSON report
    import json
    json_report = {
        "tensor": "blk.0.attn_v.weight",
        "quant": "Q8_0",
        "block_index": 0,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "cosine_similarity": cos_sim,
        "status": "PASS" if max_err < 1e-6 else "FAIL",
        "manual_values": manual_values,
        "reference_values": ref_values
    }
    (parity_dir / "manual_verify_report.json").write_text(json.dumps(json_report, indent=2))
    
    return max_err < 1e-6

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)