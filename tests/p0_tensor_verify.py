import mmap
import struct

model_path = r"D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"

with open(model_path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    magic = mm[0:4].decode('ascii')
    version = int.from_bytes(mm[4:8], 'little')
    tensor_count = int.from_bytes(mm[8:16], 'little')
    kv_count = int.from_bytes(mm[16:24], 'little')
    
    print(f"Magic: {magic}")
    print(f"Version: {version}")
    print(f"Tensor count: {tensor_count}")
    print(f"KV count: {kv_count}")
    print()
    
    pos = 24
    
    # Skip metadata
    for _ in range(kv_count):
        key_len = int.from_bytes(mm[pos:pos+8], 'little')
        pos += 8
        if key_len > 100: pos += 100; continue
        pos += key_len
        
        type_byte = mm[pos]
        pos += 1
        
        if type_byte == 2:
            str_len = int.from_bytes(mm[pos:pos+8], 'little')
            pos += 8 + str_len
        elif type_byte == 1:
            pos += 8
        elif type_byte == 5:
            pos += 4
    
    print(f"After metadata, pos = {pos}")
    
    # Parse tensors
    found = {}
    for i in range(min(tensor_count, 50)):
        name_len = int.from_bytes(mm[pos:pos+8], 'little')
        pos += 8
        if name_len > 100 or name_len == 0: continue
        
        name = mm[pos:pos+name_len].decode('utf-8', errors='replace')
        pos += name_len
        
        dims_count = int.from_bytes(mm[pos:pos+4], 'little')
        pos += 4
        
        dims = []
        for _ in range(dims_count):
            d = int.from_bytes(mm[pos:pos+8], 'little')
            dims.append(d)
            pos += 8
        
        dtype = int.from_bytes(mm[pos:pos+4], 'little')
        pos += 4
        offset = int.from_bytes(mm[pos:pos+8], 'little')
        pos += 8
        
        if 'token_embd' in name:
            found[name] = {'dims': dims, 'dtype': dtype, 'offset': offset}
            print(f"Found: {name}")
            print(f"  dims: {dims}")
            print(f"  dtype: {dtype} (Q4_K_M=18)")
            print(f"  offset: {offset}")
    
    mm.close()
    
    # Verify
    if 'token_embd.weight' in found:
        t = found['token_embd.weight']
        print("\n✅ P0-TENSOR PASS" if t['dtype'] == 18 and len(t['dims']) == 2 else "❌ FAIL")