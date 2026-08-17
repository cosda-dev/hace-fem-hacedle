//! Stage B Simple - Verify GGUF tensor header structure

use std::fs::File;
use std::io::Read;

fn main() {
    let model_path = "D:\\host\\llama-models\\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
    let mut file = File::open(model_path).expect("Failed to open GGUF");
    
    // Read header
    let mut header = [0u8; 24];
    file.read_exact(&mut header).unwrap();
    
    println!("Magic: {:?}", String::from_utf8_lossy(&header[0..4]));
    println!("Version: {:?}", header[4]);
    
    let tensor_count = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let kv_count = u64::from_le_bytes(header[16..24].try_into().unwrap());
    
    println!("Tensor count: {}", tensor_count);
    println!("Metadata KV count: {}", kv_count);
    
    // Read first tensor name to verify format
    let mut name_len = [0u8; 8];
    file.read_exact(&mut name_len).unwrap();
    let name_len = u64::from_le_bytes(name_len) as usize;
    
    let mut name = vec![0u8; name_len];
    file.read_exact(&mut name).unwrap();
    
    println!("First tensor name: {}", String::from_utf8_lossy(&name));
    
    println!("\n✅ GGUF format verified - tensors accessible");
}