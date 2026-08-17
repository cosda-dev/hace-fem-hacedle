#!/usr/bin/env python3
"""
Export GGUF tensors to reference format for parity testing
Uses llama.cpp gguf-py for accurate dequantization
"""

import sys
import os
import json
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')

from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

def export_model(model_path: str, output_dir: str = "reference"):
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    
    print(f"Loading model: {model_path}")
    reader = GGUFReader(model_path)
    
    # Extract metadata
    metadata = {}
    for key, value in reader.fields.items():
        metadata[key] = str(value)
    
    with open(output / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata extracted: {len(metadata)} fields")
    
    # Find block0 tensors
    tensors_to_export = []
    for tensor_info in reader.tensors:
        tensor_name = tensor_info.name
        if "blk.0." in tensor_name and "weight" in tensor_name:
            tensors_to_export.append(tensor_info)
    
    print(f"Found {len(tensors_to_export)} block0 weight tensors")
    
    # Export each tensor as f32 binary
    for tensor_info in tensors_to_export[:10]:
        try:
            tensor_name = tensor_info.name
            print(f"Exporting: {tensor_name}")
            
            # Get tensor data (numpy array)
            data = tensor_info.data
            
            if tensor_info.tensor_type.value > 8:  # Quantized
                # Dequantize - only 2 args
                f32_tensor = dequantize(data, tensor_info.tensor_type)
            else:
                f32_tensor = data.astype(np.float32).flatten()
            
            # Save raw binary
            safe_name = tensor_name.replace(".", "_").replace("/", "_")
            output_path = output / f"{safe_name}.bin"
            
            # Convert shape to list properly
            shape_list = [int(x) for x in tensor_info.shape]
            
            f32_tensor.tofile(output_path)
            
            # Save metadata
            with open(output / f"{safe_name}.meta.json", "w") as f:
                json.dump({
                    "shape": shape_list,
                    "dtype": "f32",
                    "min": float(np.min(f32_tensor)),
                    "max": float(np.max(f32_tensor)),
                    "mean": float(np.mean(f32_tensor)),
                    "n_elements": int(tensor_info.n_elements),
                }, f, indent=2)
            
            print(f"  Saved: {output_path} shape={shape_list}")
            
        except Exception as e:
            print(f"Failed to export {tensor_info.name}: {e}")
    
    print(f"Done. Files saved to: {output}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py export_gguf_tensors.py <model.gguf> [output_dir]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "reference"
    
    export_model(model_path, output_dir)