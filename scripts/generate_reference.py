#!/usr/bin/env python3
"""
Generate reference data for GGUF parity tests using llama.cpp GGML.
This creates binary files for comparison with HACE implementation.

Usage:
    python generate_reference.py <model.gguf> [output_dir]
"""

import sys
import os
import struct
import numpy as np

try:
    from gguf import GGUFReader
except ImportError:
    print("ERROR: gguf package required. Install with: pip install gguf")
    sys.exit(1)

def f16_to_f32(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    frac = h & 0x3FF
    if exp == 0:
        return 0.0
    f32_exp = (exp - 15 + 127)
    f32_frac = frac / 1024.0
    result = (1.0 + f32_frac) * (2.0 ** f32_exp)
    return -result if sign else result

def dequant_q4k(data, shape):
    """Dequantize Q4_K tensor using llama.cpp logic."""
    BLOCK_SIZE = 256
    BYTES_PER_BLOCK = 144
    
    numel = np.prod(shape) if len(shape) > 0 else len(data) // BYTES_PER_BLOCK * BLOCK_SIZE
    output = np.zeros(numel, dtype=np.float32)
    
    blocks = len(data) // BYTES_PER_BLOCK
    elements = min(len(output), blocks * BLOCK_SIZE)
    
    for block_idx in range(blocks):
        block_offset = block_idx * BYTES_PER_BLOCK
        if block_offset + BYTES_PER_BLOCK > len(data):
            break
        
        block = data[block_offset:block_offset + BYTES_PER_BLOCK]
        
        scale = f16_to_f32(struct.unpack_from('<H', block, 0)[0])
        min_val = f16_to_f32(struct.unpack_from('<H', block, 2)[0])
        
        for j in range(BLOCK_SIZE):
            idx = block_idx * BLOCK_SIZE + j
            if idx >= elements:
                break
            
            scale_idx = 8 + (j // 32)
            scale_val = f16_to_f32(struct.unpack_from('<H', block, scale_idx * 2)[0])
            
            q_offset = 16 + (j // 2)
            q = block[q_offset] & 0xF if j % 2 == 0 else block[q_offset] >> 4
            
            output[idx] = (q * scale_val) + min_val
    
    return output.reshape(shape)

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_reference.py <model.gguf> [output_dir]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "reference"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model: {model_path}")
    reader = GGUFReader(model_path)
    
    # Extract metadata
    metadata = {}
    for key, _ in reader.fields.items():
        if key in ['rope_theta', 'rope_scaling', 'context_length']:
            metadata[key] = reader.field(key)
    
    print(f"Metadata: {metadata}")
    
    # Dump tensor layers
    tensors_to_dump = [
        'token_embd.weight',
        'blk.0.attn_q.weight',
        'blk.0.attn_k.weight',
        'blk.0.attn_v.weight',
        'blk.0.attn_output.weight',
        'blk.0.ffn_gate.weight',
        'blk.0.ffn_up.weight',
        'blk.0.ffn_down.weight',
        'blk.0.attn_norm.weight',
        'blk.0.ffn_norm.weight',
        'output.weight',
    ]
    
    for tensor_name in tensors_to_dump:
        try:
            data = reader.tensor(tensor_name)
            np.save(os.path.join(output_dir, tensor_name.replace('.', '_')), data)
            print(f"Dumped: {tensor_name} -> {tensor_name.replace('.', '_')}.npy")
        except KeyError:
            print(f"Tensor not found: {tensor_name}")
    
    print(f"\nReference files saved to: {output_dir}")

if __name__ == "__main__":
    main()