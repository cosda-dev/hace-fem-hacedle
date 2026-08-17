// T1: Tensor Projection Verification
// Target: token_embd.weight

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "D:\\host\\llama-models\\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
    let mut f = BufReader::new(File::open(path)?);
    
    // Parse header
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    f.seek_relative(4).unwrap(); // version
    f.seek_relative(4).unwrap(); // padding
    
    let mut tensor_count_bytes = [0u8; 8];
    f.read_exact(&mut tensor_count_bytes)?;
    let tensor_count = u64::from_le_bytes(tensor_count_bytes);
    
    let mut kv_count_bytes = [0u8; 8];
    f.read_exact(&mut kv_count_bytes)?;
    let kv_count = u64::from_le_bytes(kv_count_bytes);
    
    println!("Tensor count: {}", tensor_count);
    println!("KV count: {}", kv_count);
    
    // Skip metadata
    for _ in 0..kv_count {
        let mut key_len = [0u8; 8];
        f.read_exact(&mut key_len)?;
        let kl = u64::from_le_bytes(key_len) as usize;
        f.seek_relative(kl as i64).unwrap();
        
        let mut type_byte = [0u8; 1];
        f.read_exact(&mut type_byte)?;
        
        match type_byte[0] {
            2 => {
                let mut len = [0u8; 8];
                f.read_exact(&mut len)?;
                f.seek_relative(u64::from_le_bytes(len) as i64).unwrap();
            }
            1 => f.seek_relative(8).unwrap(),
            5 => f.seek_relative(4).unwrap(),
            _ => {}
        }
    }
    
    // Read tensor info
    let mut token_embd_offset: u64 = 0;
    let mut token_embd_dims: Vec<u64> = vec![];
    
    for _ in 0..tensor_count {
        let mut name_len = [0u8; 8];
        f.read_exact(&mut name_len)?;
        let nl = u64::from_le_bytes(name_len) as usize;
        
        let mut name = vec![0u8; nl];
        f.read_exact(&mut name)?;
        let name_str = String::from_utf8_lossy(&name);
        
        let mut n_dims = [0u8; 4];
        f.read_exact(&mut n_dims)?;
        let dims_count = u32::from_le_bytes(n_dims) as usize;
        
        let mut dims = vec![0u64; 4];
        for d in 0..dims_count {
            f.read_exact(&mut dims[..])?;
        }
        
        let mut shape: Vec<u64> = Vec::new();
        for i in 0..4 {
            let mut d = [0u8; 8];
            f.read_exact(&mut d)?;
            shape.push(u64::from_le_bytes(d));
        }
        
        f.seek_relative(4).unwrap(); // dtype
        let mut offset = [0u8; 8];
        f.read_exact(&mut offset)?;
        let off = u64::from_le_bytes(offset);
        
        if name_str.contains("token_embd") {
            token_embd_offset = off;
            token_embd_dims = shape.clone();
        }
    }
    
    println!("\n=== T1: Tensor Projection Test ===");
    println!("token_embd.weight offset: {}", token_embd_offset);
    println!("token_embd.weight dims: {:?}", token_embd_dims);
    
    // Expected for Qwen2.5 1.5B: vocab_size=151936, hidden_size=1536
    // dims should be [vocab_size, hidden_size] or [hidden_size]
    if !token_embd_dims.is_empty() {
        println!("✅ T1 PASS: Tensor found");
    }
    
    Ok(())
}