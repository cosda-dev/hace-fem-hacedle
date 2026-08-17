//! Stage B+ Test - Tensor Metadata Reader
//! Parse 338 tensors from Qwen2.5-Coder-1.5B-Q4_K_M.gguf

use std::fs::File;
use std::io::Read;

#[derive(Debug, Clone)]
struct GGUFTensor {
    name: String,
    n_dims: u32,
    shape: Vec<u32>,
    dtype: u32,
    offset: u64,
}

fn main() {
    let model_path = "D:\\host\\llama-models\\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
    let mut file = File::open(model_path).expect("Failed to open GGUF");
    
    // Skip header (magic + version + tensor_count)
    let mut header = [0u8; 24];
    file.read_exact(&mut header).unwrap();
    
    let tensor_count = u64::from_le_bytes([header[8], header[9], header[10], header[11], header[12], header[13], header[14], header[15]]);
    println!("Tensor count from header: {}", tensor_count);
    
    // Read metadata key-value pairs count
    let mut kv_header = [0u8; 8];
    file.read_exact(&mut kv_header).unwrap();
    let kv_count = u64::from_le_bytes(kv_header);
    println!("Metadata KV count: {}", kv_count);
    
    // Skip metadata
    for _ in 0..kv_count {
        let mut key_len = [0u8; 8];
        file.read_exact(&mut key_len).unwrap();
        let key_len = u64::from_le_bytes(key_len) as usize;
        
        let mut key = vec![0u8; key_len];
        file.read_exact(&mut key).unwrap();
        
        let mut type_byte = [0u8; 1];
        file.read_exact(&mut type_byte).unwrap();
        
        let value_type = type_byte[0];
        match value_type {
            0 => {}
            1 => { let mut buf = [0u8; 8]; file.read_exact(&mut buf).unwrap(); }
            2 => { 
                let mut len = [0u8; 8];
                file.read_exact(&mut len).unwrap();
                let len = u64::from_le_bytes(len) as usize;
                let mut buf = vec![0u8; len];
                file.read_exact(&mut buf).unwrap();
            }
            _ => { let mut len = [0u8; 8]; file.read_exact(&mut len).unwrap(); }
        }
    }
    
    // Read tensors
    let mut tensors = Vec::new();
    for i in 0..tensor_count {
        let mut name_len = [0u8; 8];
        file.read_exact(&mut name_len).unwrap();
        let name_len = u64::from_le_bytes(name_len) as usize;
        
        let mut name = vec![0u8; name_len];
        file.read_exact(&mut name).unwrap();
        
        let mut n_dims = [0u8; 4];
        file.read_exact(&mut n_dims).unwrap();
        let dims = u32::from_le_bytes(n_dims);
        
        let mut shape = vec![0u32; 4];
        for d in 0..4 {
            let mut dim = [0u8; 4];
            file.read_exact(&mut dim).unwrap();
            shape[d] = u32::from_le_bytes(dim);
        }
        
        let mut dtype = [0u8; 4];
        file.read_exact(&mut dtype).unwrap();
        let dt = u32::from_le_bytes(dtype);
        
        let mut offset = [0u8; 8];
        file.read_exact(&mut offset).unwrap();
        let off = u64::from_le_bytes(offset);
        
        let tensor = GGUFTensor {
            name: String::from_utf8_lossy(&name).to_string(),
            n_dims: dims,
            shape: shape.into_iter().take(dims as usize).collect(),
            dtype: dt,
            offset: off,
        };
        
        if i < 5 {
            println!("Tensor {}: {} shape:{:?} dtype:{} off:{}", 
                i, tensor.name, tensor.shape, tensor.dtype, tensor.offset);
        }
        
        tensors.push(tensor);
    }
    
    println!("\n✅ Total tensors parsed: {}", tensors.len());
    
    // Find key tensors
    for t in &tensors {
        if t.name.contains("tok_embeddings") {
            println!("✅ Token embeddings: {:?}", t.shape);
        }
        if t.name.contains("output") && t.name.contains("weight") {
            println!("✅ Output layer: {:?}", t.shape);
        }
    }
}