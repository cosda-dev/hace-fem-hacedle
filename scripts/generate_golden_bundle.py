#!/usr/bin/env python3
"""
Generate Golden Bundle from GGUF + llama.cpp activations
Phase E6: SKB (Sealed Knowledge Bundle) creation
"""

import os
import json
import struct
import hashlib
from pathlib import Path

def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def compute_stats(f32_data: list) -> dict:
    if not f32_data:
        return {"min": 0, "max": 0, "mean": 0, "std": 0, "l2_norm": 0}
    
    import numpy as np
    arr = np.array(f32_data, dtype=np.float32)
    
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "l2_norm": float(np.linalg.norm(arr))
    }

def create_bundle(model_name: str, gguf_path: str, output_dir: str = "golden"):
    bundle_dir = Path(output_dir) / model_name.lower().replace("-", "").replace(".", "")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "block0").mkdir(exist_ok=True)
    
    manifest = {
        "model": model_name,
        "version": "1.0",
        "source": "llama.cpp",
        "files": {}
    }
    
    # Load GGUF model
    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: pip install gguf")
        return
    
    print(f"Loading: {gguf_path}")
    reader = GGUFReader(gguf_path)
    
    # Extract and save model metadata
    metadata = {}
    for key, value in reader.fields.items():
        if key in ["rope_theta", "rope_scaling", "context_length", "head_count", 
                   "head_count_kv", "embedding_length", "block_count"]:
            metadata[key] = str(value)
    
    with open(bundle_dir / "model_spec.sio", "w") as f:
        json.dump(metadata, f, indent=2)
    
    manifest["files"]["model_spec"] = "model_spec.sio"
    
    # Save tensor activations (block0)
    tensors = ["attn_q", "attn_k", "attn_v", "attn_output", 
               "ffn_gate", "ffn_up", "ffn_down"]
    
    for tensor_name in tensors:
        try:
            data = reader.tensor(f"blk.0.{tensor_name}.weight")
            output = data.flatten().astype('float32')
            
            # Save binary
            out_path = bundle_dir / "block0" / f"{tensor_name}.bin"
            output.tofile(out_path)
            
            # Save metadata
            stats = compute_stats(output.tolist())
            meta = {
                "shape": list(data.shape),
                "stats": stats,
                "sha256": compute_sha256(output.tobytes())
            }
            
            with open(bundle_dir / "block0" / f"{tensor_name}.json", "w") as f:
                json.dump(meta, f, indent=2)
            
            manifest["files"][f"block0_{tensor_name}"] = f"block0/{tensor_name}.bin"
            
        except KeyError:
            print(f"Tensor not found: blk.0.{tensor_name}.weight")
    
    # Save manifest
    with open(bundle_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Bundle created: {bundle_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python generate_golden_bundle.py <model_name> <gguf_path>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    gguf_path = sys.argv[2]
    
    create_bundle(model_name, gguf_path)