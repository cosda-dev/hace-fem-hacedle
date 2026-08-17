#!/usr/bin/env python3
"""Extract RAW Q5_0 block bytes for bit-exact test"""

import sys
from pathlib import Path
sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')

from gguf.gguf_reader import GGUFReader
import numpy as np

def main():
    reader = GGUFReader("D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
    
    for tensor_info in reader.tensors:
        if tensor_info.name == "blk.0.attn_q.weight":
            # Q5_0: 22 bytes per block, 32 elements
            raw = tensor_info.data.tobytes()
            
            # Extract 10 blocks
            blocks = []
            for i in range(min(10, len(raw) // 22)):
                block = raw[i*22:(i+1)*22]
                blocks.append(block)
            
            # Save
            out = Path("parity_test")
            out.mkdir(exist_ok=True)
            
            for i, block in enumerate(blocks):
                with open(out / f"q5_0_block_{i}.bin", "wb") as f:
                    f.write(block)
            
            print(f"Extracted {len(blocks)} Q5_0 blocks")
            break

if __name__ == "__main__":
    main()