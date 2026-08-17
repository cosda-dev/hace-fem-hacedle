use std::fs::File;
use std::io::Read;

fn parse_gguf_metadata(path: &str) -> (Vec<(String, String)>, Vec<(String, String)>) {
    let mut file = File::open(path).expect("Failed to open GGUF");
    
    let mut header = [0u8; 24];
    file.read_exact(&mut header).unwrap();
    
    let magic: &[u8; 4] = std::mem::transmute(&header[0..4]);
    assert_eq!(magic, b"GGUF", "Invalid GGUF magic");
    
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    println!("GGUF Version: {}", version);
    
    let tensor_count = u64::from_le_bytes([
        header[12], header[13], header[14], header[15],
        header[16], header[17], header[18], header[19]
    ]);
    println!("Tensor count: {}", tensor_count);
    
    let kv_count = u64::from_le_bytes([
        header[20], header[21], header[22], header[23],
        header[24], header[25], header[26], header[27]
    ]) as usize;
    
    let mut metadata = Vec::new();
    let mut tensors = Vec::new();
    
    for _ in 0..kv_count {
        let mut key_len = [0u8; 8];
        file.read_exact(&mut key_len).unwrap();
        let key_len = u64::from_le_bytes(key_len) as usize;
        
        let mut key = vec![0u8; key_len];
        file.read_exact(&mut key).unwrap();
        let key = String::from_utf8_lossy(&key).to_string();
        
        let mut type_byte = [0u8; 1];
        file.read_exact(&mut type_byte).unwrap();
        
        let value = match type_byte[0] {
            0 => String::new(),
            1 => {
                let mut buf = [0u8; 8];
                file.read_exact(&mut buf).unwrap();
                u64::from_le_bytes(buf).to_string()
            }
            2 => {
                let mut len = [0u8; 8];
                file.read_exact(&mut len).unwrap();
                let len = u64::from_le_bytes(len) as usize;
                let mut buf = vec![0u8; len];
                file.read_exact(&mut buf).unwrap();
                String::from_utf8_lossy(&buf).to_string()
            }
            _ => String::new(),
        };
        
        metadata.push((key, value));
    }
    
    // Parse tensor info
    for _ in 0..tensor_count {
        let mut name_len = [0u8; 8];
        file.read_exact(&mut name_len).unwrap();
        let name_len = u64::from_le_bytes(name_len) as usize;
        
        let mut name = vec![0u8; name_len];
        file.read_exact(&mut name).unwrap();
        let name = String::from_utf8_lossy(&name).to_string();
        
        let mut n_dims = [0u8; 4];
        file.read_exact(&mut n_dims).unwrap();
        let dims = u32::from_le_bytes(n_dims) as usize;
        
        let mut shape = vec![0u64; 4];
        for d in 0..4 {
            let mut dim = [0u8; 8];
            file.read_exact(&mut dim).unwrap();
            shape[d] = u64::from_le_bytes(dim);
        }
        
        let mut dtype = [0u8; 4];
        file.read_exact(&mut dtype).unwrap();
        let dt = u32::from_le_bytes(dtype);
        
        let mut offset = [0u8; 8];
        file.read_exact(&mut offset).unwrap();
        let off = u64::from_le_bytes(offset);
        
        tensors.push((name, dt, off, shape[..dims].to_vec()));
    }
    
    (metadata, tensors)
}

fn main() {
    let model_path = "D:\\host\\llama-models\\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
    let (metadata, tensors) = parse_gguf_metadata(model_path);
    
    println!("\n=== GGUF Metadata Golden Test ===\n");
    
    // Verify expected keys
    let expected = [
        ("general.architecture", "qwen2"),
        ("qwen2.context_length", "32768"),
        ("qwen2.block_count", "28"),
        ("qwen2.attention.head_count", "12"),
    ];
    
    let mut passed = 0;
    let mut failures = Vec::new();
    
    for (key, expected_val) in expected.iter() {
        if let Some(val) = metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v) {
            if val == expected_val {
                println!("✅ {} = {} (expected: {})", key, val, expected_val);
                passed += 1;
            } else {
                println!("❌ {} = {} (expected: {})", key, val, expected_val);
                failures.push(key.to_string());
            }
        } else {
            println!("❌ {} not found", key);
            failures.push(key.to_string());
        }
    }
    
    println!("\n=== Summary ===");
    println!("passed: {}/{}", passed, expected.len());
    println!("failures: {:?}", failures);
    
    if failures.is_empty() {
        println!("\n✅ TEST PASSED: All metadata verified!");
    } else {
        println!("\n❌ TEST FAILED");
        std::process::exit(1);
    }
}