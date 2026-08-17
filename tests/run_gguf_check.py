#!/usr/bin/env python3
import struct
import sys

def read_gguf_metadata(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Invalid GGUF magic: {magic}")
        
        version = struct.unpack('<I', f.read(4))[0]
        print(f"GGUF Version: {version}")
        
        f.read(4)  # padding
        
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        print(f"Tensor count: {tensor_count}")
        
        kv_count = struct.unpack('<Q', f.read(8))[0]
        print(f"Metadata entries: {kv_count}")
        
        metadata = {}
        for _ in range(kv_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8', errors='replace')
            
            type_byte = struct.unpack('B', f.read(1))[0]
            
            if type_byte == 0:  # STRING
                str_len = struct.unpack('<Q', f.read(8))[0]
                value = f.read(str_len).decode('utf-8', errors='replace')
            elif type_byte == 1:  # UINT
                value = struct.unpack('<Q', f.read(8))[0]
            elif type_byte == 2:  # FLOAT
                value = struct.unpack('<f', f.read(4))[0]
            else:
                value = None
            
            if value is not None:
                metadata[key] = str(value)
        
        return metadata, tensor_count

def main():
    model_path = r"D:\host\llama-models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"
    
    print("=== GGUF Metadata Golden Test ===")
    print()
    
    try:
        metadata, tensor_count = read_gguf_metadata(model_path)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    expected = {
        "general.architecture": "qwen2",
        "qwen2.block_count": "28",
        "qwen2.attention.head_count": "12",
    }
    
    passed = 0
    failures = []
    
    for key, expected_val in expected.items():
        if key in metadata:
            val = metadata[key]
            if val == expected_val:
                print(f"✅ {key} = {val}")
                passed += 1
            else:
                print(f"❌ {key} = {val} (expected: {expected_val})")
                failures.append(key)
        else:
            print(f"❌ {key} NOT FOUND")
            failures.append(key)
    
    print()
    print(f"=== Summary: {passed}/{len(expected)} passed ===")
    
    if failures:
        print(f"Failures: {failures}")
        sys.exit(1)
    else:
        print("✅ TEST PASSED")

if __name__ == "__main__":
    main()