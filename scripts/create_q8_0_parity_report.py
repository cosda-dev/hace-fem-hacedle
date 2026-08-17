#!/usr/bin/env python3
"""Task A: Q8_0 bit-exact parity verification"""

import json
import struct
from pathlib import Path

def main():
    # Q8_0 layout: 34 bytes per block
    # 2 bytes f16 scale + 32 bytes u8 values
    
    # Test data: simple block
    scale_bytes = struct.pack('<e', 1.0)  # f16 for 1.0
    values = bytes([0, 32, 64, 96, 128, 160, 192, 224] * 4)  # 8-bit values
    
    block = bytearray(scale_bytes + values)
    
    output_dir = Path("parity_test")
    output_dir.mkdir(exist_ok=True)
    
    # Save raw block
    with open(output_dir / "q8_0_block.bin", "wb") as f:
        f.write(block)
    
    # Expected dequant values
    expected = []
    scale = 1.0
    for v in values:
        expected.append((v - 128) * scale)  # Q8_0: values are signed 8-bit centered at 128
    
    # Save expected
    with open(output_dir / "q8_0_expected.bin", "wb") as f:
        for val in expected:
            f.write(struct.pack('<f', val))
    
    print(f"Q8_0 test block created: {len(block)} bytes, {len(expected)} values")
    print(f"Expected values: {expected[:8]}")

if __name__ == "__main__":
    main()