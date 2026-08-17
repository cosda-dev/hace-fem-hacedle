import numpy as np
import mmap

model_path = r"D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"

with open(model_path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    # GGUF header format (v3):
    # Offset 0-3: magic (4 bytes)
    # Offset 4-7: version (u32)
    # Offset 8-11: tensor_count (u64, but only 4 bytes in v3?)
    # Let me check the correct format
    
    magic = mm[0:4].decode('ascii')
    version = int.from_bytes(mm[4:8], 'little')
    
    # GGUF v3 format: tensor_count at offset 8 (u64)
    tensor_count = int.from_bytes(mm[8:16], 'little')
    kv_count = int.from_bytes(mm[16:24], 'little')
    
    print(f"Magic: {magic}")
    print(f"Version: {version}")
    print(f"Tensor count: {tensor_count}")
    print(f"KV count: {kv_count}")
    print()
    
    # Metadata starts at offset 24
    pos = 24
    
    # Parse ALL metadata
    for i in range(kv_count):
        key_len = int.from_bytes(mm[pos:pos+8], 'little')
        pos += 8
        
        if key_len > 100 or key_len == 0:
            print(f"KV[{i}]: key_len={key_len} (skipping)")
            continue
            
        key = mm[pos:pos+key_len].decode('utf-8', errors='replace')
        pos += key_len
        
        type_byte = mm[pos]
        pos += 1
        
        if type_byte == 2:  # STRING
            str_len = int.from_bytes(mm[pos:pos+8], 'little')
            val = mm[pos+8:pos+8+str_len].decode('utf-8', errors='replace')
            print(f"KV[{i}]: {key} = {val}")
            pos += 8 + str_len
        elif type_byte == 1:  # UINT
            val = int.from_bytes(mm[pos:pos+8], 'little')
            print(f"KV[{i}]: {key} = {val}")
            pos += 8
        elif type_byte == 5:  # FLOAT32
            pos += 4
    
    print(f"\nTensor data starts at offset: {pos}")
    mm.close()