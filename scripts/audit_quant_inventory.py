#!/usr/bin/env python3
"""P0: Quant Inventory Audit - Exact tensor quant types from GGUF"""

import sys
from pathlib import Path
sys.path.insert(0, 'D:/host/llama.cpp/gguf-py')

from gguf.gguf_reader import GGUFReader
import json

def audit_quant_types(model_path: str):
    print(f"Loading: {model_path}")
    reader = GGUFReader(model_path)
    
    # Extract model architecture
    arch = {}
    for key, value in reader.fields.items():
        arch[key] = str(value)
    
    # Key architecture fields
    print("\n=== Model Architecture ===")
    for key in ["qwen2.embedding_length", "qwen2.block_count", "qwen2.attention.head_count", 
                "qwen2.attention.head_count_kv", "qwen2.feed_forward.wide_hidden", 
                "qwen2.rope.freq_base", "qwen2.context_length"]:
        if key in arch:
            print(f"{key}: {arch[key]}")
    
    # Quant type inventory
    quant_inventory = {}
    
    print("\n=== Quant Inventory ===")
    for tensor_info in reader.tensors:
        name = tensor_info.name
        dtype = tensor_info.tensor_type.name
        shape = tensor_info.shape
        
        if dtype not in quant_inventory:
            quant_inventory[dtype] = {"count": 0, "tensors": []}
        
        quant_inventory[dtype]["count"] += 1
        quant_inventory[dtype]["tensors"].append(name)
    
    # Print summary
    for dtype, info in quant_inventory.items():
        print(f"\n{dtype}: {info['count']} tensors")
        # Show first 5 tensors
        for t in info['tensors'][:5]:
            print(f"  - {t}")
        if len(info['tensors']) > 5:
            print(f"  ... and {len(info['tensors']) - 5} more")
    
    # Save inventory
    output = {"architecture": arch, "quant_inventory": quant_inventory}
    Path("golden").mkdir(exist_ok=True)
    with open("golden/quant_inventory.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nSaved to: golden/quant_inventory.json")
    
    # Audit attention dimensions
    print("\n=== Attention Dimension Audit ===")
    n_head = int(arch.get("qwen2.attention.head_count", 14))
    n_head_kv = int(arch.get("qwen2.attention.head_count_kv", 2))
    hidden_size = int(arch.get("qwen2.embedding_length", 896))
    head_dim = hidden_size // n_head
    
    print(f"n_head: {n_head}")
    print(f"n_head_kv: {n_head_kv}")
    print(f"hidden_size: {hidden_size}")
    print(f"head_dim: {head_dim}")
    
    # Check attn_k/v shapes
    for tensor_info in reader.tensors:
        if "blk.0.attn_k.weight" in tensor_info.name or "blk.0.attn_v.weight" in tensor_info.name:
            print(f"{tensor_info.name} shape: {tensor_info.shape}")
            # Expected: [hidden_size, n_head_kv * head_dim]
            expected_kv_dim = n_head_kv * head_dim
            actual_kv_dim = int(tensor_info.shape[1])
            print(f"  Expected KV dim: {expected_kv_dim}, Actual: {actual_kv_dim}")
            if expected_kv_dim == actual_kv_dim:
                print("  ✓ GQA verified!")
            else:
                print("  ✗ GQA mismatch!")

if __name__ == "__main__":
    audit_quant_types("D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")