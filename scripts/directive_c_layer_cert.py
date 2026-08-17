#!/usr/bin/env python3
"""Directive C: Layer Certification - Tạo layer_*.yaml cho 24 layers"""

import numpy as np
from pathlib import Path
import json
import hashlib

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]

def main():
    layer_dir = Path("golden/layer_replay")
    cert_dir = Path("golden/layer_certification")
    cert_dir.mkdir(parents=True, exist_ok=True)
    
    # Create certification for each layer
    for layer in range(24):
        layer_path = layer_dir / f"layer{layer}"
        
        cert = {
            "layer_id": layer,
            "status": "GOLDEN_GENERATED",
            "attention": {
                "cosine": "PENDING_RUNTIME",  # To be filled by Rust
                "max_abs_error": "N/A"
            },
            "ffn": {
                "cosine": "PENDING_RUNTIME",
                "max_abs_error": "N/A"
            },
            "residual": {
                "cosine": "PENDING_RUNTIME",
                "max_abs_error": "N/A"
            },
            "golden_files": []
        }
        
        # List golden files
        if layer_path.exists():
            cert["golden_files"] = [f.name for f in layer_path.glob("*.bin")]
        
        with open(cert_dir / f"layer_{layer}.yaml", "w") as f:
            import yaml
            yaml.dump(cert, f)
    
    # Master certification
    master = {
        "total_layers": 24,
        "layers_verified": 0,
        "gate": "all_layers_pass == true"
    }
    with open(cert_dir / "certification.yaml", "w") as f:
        import yaml
        yaml.dump(master, f)
    
    print(f"Created layer certification for layers 0-23")

if __name__ == "__main__":
    main()