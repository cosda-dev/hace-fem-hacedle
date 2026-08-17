#!/usr/bin/env python3
"""P5.6: Create tensor fingerprint registry for golden bundles"""

import json
import hashlib
from pathlib import Path

def xxhash64(data: bytes) -> str:
    """Simple hash - in production use xxhash library"""
    import zlib
    return hashlib.sha256(data).hexdigest()[:16]

def create_registry():
    ref_dir = Path("golden/qwen2505b/block0")
    registry = {"tensors": {}, "created": "2025-06-02"}
    
    for bin_file in ref_dir.glob("*.bin"):
        name = bin_file.stem
        data = bin_file.read_bytes()
        
        # Get shape from reference
        meta_file = ref_dir / f"{name}.meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
        else:
            meta = {"shape": [len(data) // 4], "dtype": "f32"}
        
        registry["tensors"][name] = {
            "shape": meta.get("shape", []),
            "dtype": meta.get("dtype", "f32"),
            "bytes": len(data),
            "sha256": xxhash64(data),
        }
    
    Path("golden").mkdir(exist_ok=True)
    with open("golden/fingerprint_registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"Registry: {len(registry['tensors'])} tensors")

if __name__ == "__main__":
    create_registry()