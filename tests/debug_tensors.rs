// Debug: List first 10 tensor names
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "D:\\host\\llama-models\\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
    
    let mut f = BufReader::new(File::open(path)?);
    
    // Skip header (24 bytes)
    f.seek_relative(24)?;
    
    // Read ALL metadata to get position
    let mut pos = 24;
    let mut kv_skipped = 0;
    
    loop {
        let mut key_len = [0u8; 8];
        if f.read_exact(&mut key_len).is_err() { break; }
        let kl = u64::from_le_bytes(key_len) as usize;
        if kl > 500 { break; }
        f.seek_relative(kl as i64)?;
        kv_skipped += 1;
        
        let mut type_byte = [0u8; 1];
        f.read_exact(&mut type_byte)?;
        
        match type_byte[0] {
            0 => {}
            1 => { f.seek_relative(8)?; }
            2 => {
                let mut len = [0u8; 8];
                f.read_exact(&mut len)?;
                f.seek_relative(u64::from_le_bytes(len) as i64)?;
            }
            3..=6 | _ => { f.seek_relative(8)?; }
        }
        
        if kv_skipped >= 100 { break; }
    }
    
    println!("After skipping {} KV entries, pos = {}", kv_skipped, pos);
    
    // Now read first 10 tensors
    for i in 0..10 {
        let mut name_len = [0u8; 8];
        f.read_exact(&mut name_len)?;
        let nl = u64::from_le_bytes(name_len) as usize;
        
        let mut name = vec![0u8; nl.max(1)];
        f.read_exact(&mut name)?;
        
        let mut n_dims = [0u8; 4];
        f.read_exact(&mut n_dims)?;
        let dims_count = u32::from_le_bytes(n_dims) as usize;
        
        f.seek_relative((dims_count * 8) as i64)?; // skip dims
        f.seek_relative(4 + 8)?; // skip dtype + offset
        
        println!("Tensor {}: {}", i, String::from_utf8_lossy(&name));
    }
    
    Ok(())
}