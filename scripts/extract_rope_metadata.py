#!/usr/bin/env python3
"""P5.2: Extract complete RoPE metadata from GGUF"""

import sys
import json
from pathlib import Path

sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')

from gguf.gguf_reader import GGUFReader

def extract_rope_metadata(model_path: str, output_path: str):
    reader = GGUFReader(model_path)
    
    rope_keys = [
        "qwen2.rope.freq_base",
        "qwen2.rope.scaling_type", 
        "qwen2.rope.scaling_factor",
        "qwen2.rope.scaling_orig_ctx_len",
    ]
    
    rope_metadata = {}
    
    for key, value in reader.fields.items():
        key_str = str(key)
        if "rope" in key_str.lower():
            rope_metadata[key_str] = str(value)
    
    # Also check standard GGUF keys
    for key in ["rope.freq_base", "rope.scaling.factor", "rope.scaling.origin"]:
        if key in reader.fields:
            rope_metadata[key] = str(reader.fields[key])
    
    print("RoPE Metadata Found:")
    for k, v in rope_metadata.items():
        print(f"  {k}: {v}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rope_metadata, f, indent=2)
    
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    extract_rope_metadata(
        "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "golden/rope_metadata.json"
    )