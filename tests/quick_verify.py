import struct

model_path = r"D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"

with open(model_path, "rb") as f:
    # GGUF v3 header (28 bytes total)
    magic = f.read(4).decode('ascii')
    version = struct.unpack('<I', f.read(4))[0]
    f.read(4)  # padding (4 bytes)
    tensor_count = struct.unpack('<Q', f.read(8))[0]
    kv_count = struct.unpack('<Q', f.read(8))[0]
    
    print(f"Magic: {magic}")
    print(f"Version: {version}")
    print(f"Tensor count: {tensor_count}")
    print(f"KV count: {kv_count}")
    print()
    print("=== Key Metadata ===")
    
    meta_dict = {}
    for i in range(kv_count):
        key_len = struct.unpack('<Q', f.read(8))[0]
        if key_len > 1000 or key_len == 0:
            continue
        key = f.read(key_len).decode('utf-8', errors='replace')
        type_byte = f.read(1)[0]
        
        if type_byte == 2:  # STRING
            str_len = struct.unpack('<Q', f.read(8))[0]
            if str_len > 10000:
                continue
            val = f.read(str_len).decode('utf-8', errors='replace')
        elif type_byte == 1:  # UINT
            val = struct.unpack('<Q', f.read(8))[0]
        elif type_byte == 5:  # FLOAT32
            val = struct.unpack('<f', f.read(4))[0]
        else:
            val = None
            
        meta_dict[key] = val
    
    # Show key metadata
    keys_to_show = ['architecture', 'block_count', 'head_count', 'head_count_kv', 
                    'embedding_length', 'context_length', 'rope.freq_base', 'vocab_size']
    
    for key in meta_dict:
        for k in keys_to_show:
            if k in key:
                print(f"{key} = {meta_dict[key]}")
                break