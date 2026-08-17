//! Stage A Binary Test - GGUF Load
//! Run: cargo run --bin stage_a

use std::fs::File;
use std::io::Read;

fn main() {
    let model_path = "D:\\host\\llama-models\\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
    
    let mut file = File::open(model_path).expect("Failed to open GGUF file");
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).expect("Failed to read magic");

    if magic == *b"GGUF" {
        println!("✓ GGUF magic verified: {:?}", String::from_utf8_lossy(&magic));
        
        let mut version = [0u8; 4];
        file.read_exact(&mut version).expect("Failed to read version");
        println!("✓ GGUF version: {:?}", version);
        
        let mut tensor_count_bytes = [0u8; 8];
        file.read_exact(&mut tensor_count_bytes).expect("Failed to read tensor count");
        let tensor_count = u64::from_le_bytes(tensor_count_bytes);
        println!("✓ Tensor count: {}", tensor_count);
    } else {
        eprintln!("✗ Invalid GGUF magic: {:?}", magic);
    }
}