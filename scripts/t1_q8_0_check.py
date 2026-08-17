#!/usr/bin/env python3
"""Quick verification that Q8_0 test data exists"""

import json
from pathlib import Path

parity_dir = Path("parity_test/q8_0")
raw_path = parity_dir / "blk0_attn_v_q8_0_first_block.raw"
ref_path = parity_dir / "blk0_attn_v_q8_0_first_block.bin"

print("T1 Q8_0 Test Data Check")
print("=" * 40)

if not raw_path.exists():
    print("ERROR: Missing raw bytes file")
    exit(1)
    
raw_data = raw_path.read_bytes()
print(f"Raw bytes: {len(raw_data)} bytes (expected 34)")

if len(raw_data) != 34:
    print("ERROR: Wrong raw bytes size")
    exit(1)

if not ref_path.exists():
    print("ERROR: Missing reference f32 file")
    exit(1)

import struct
ref_data = ref_path.read_bytes()
ref_values = [struct.unpack('<f', ref_data[i*4:(i+1)*4])[0] for i in range(8)]
print(f"Reference f32 (first 8): {ref_values}")

print("\nReady for Rust parity test")