use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

fn load_gguf_metadata(path: &str) -> HashMap<String, String> {
    let mut file = File::open(path).expect("Failed to open GGUF");
    let mut header = [0u8; 24];
    file.read_exact(&mut header).unwrap();
    
    let kv_count = u64::from_le_bytes([
        header[16], header[17], header[18], header[19],
        header[20], header[21], header[22], header[23]
    ]) as usize;
    
    let mut metadata = HashMap::new();
    
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
        
        metadata.insert(key, value);
    }
    
    metadata
}

fn main() {
    let model_path = "D:\\host\\llama-models\\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
    let metadata = load_gguf_metadata(model_path);
    
    println!("=== Qwen2.5 ModelSpec ===");
    
    let arch = metadata.get("general.architecture").cloned().unwrap_or("qwen2".to_string());
    println!("arch={}", arch);
    
    if let Some(vocab) = metadata.get("qwen2.embedding_length") {
        println!("vocab_size={}", vocab);
        println!("hidden_size={}", vocab);
    }
    
    if let Some(layers) = metadata.get("qwen2.block_count") {
        println!("layers={}", layers);
    }
    
    if let Some(heads) = metadata.get("qwen2.attention.head_count") {
        println!("heads={}", heads);
    }
    
    if let Some(kv_heads) = metadata.get("qwen2.attention.head_count_kv") {
        println!("kv_heads={}", kv_heads);
    }
    
    if let Some(ctx) = metadata.get("qwen2.context_length") {
        println!("context_length={}", ctx);
    }
    
    if let Some(theta) = metadata.get("qwen2.rope.freq_base") {
        println!("rope_theta={}", theta);
    }
    
    println!("\n✅ ModelSpec extraction successful!");
}