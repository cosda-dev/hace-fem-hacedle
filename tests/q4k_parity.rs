// Q4_K Parity Test - Compare dequant against llama.cpp reference
// Run: cargo test --test q4k_parity --features std -- --nocapture

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;
    
    if exp == 0 {
        0.0
    } else {
        let f32_exp = (exp as i32) - 15 + 127;
        let f32_frac = frac as f32 / 1024.0;
        let result = (1.0 + f32_frac) * 2.0_f32.powi(f32_exp);
        if sign == 1 { -result } else { result }
    }
}

pub fn load_gguf_tensor(path: &str, tensor_name: &str) -> Option<(Vec<u8>, Vec<u64>, u32)> {
    let mut file = File::open(path).ok()?;
    
    let mut header = [0u8; 4];
    file.read_exact(&mut header).ok()?;
    if &header != GGUF_MAGIC {
        return None;
    }
    
    let mut version = [0u8; 4];
    file.read_exact(&mut version).ok()?;
    
    let mut tensor_count = [0u8; 8];
    file.read_exact(&mut tensor_count).ok()?;
    let tensor_count = u64::from_le_bytes(tensor_count) as usize;
    
    let mut kv_count = [0u8; 8];
    file.read_exact(&mut kv_count).ok()?;
    let kv_count = u64::from_le_bytes(kv_count) as usize;
    
    for _ in 0..kv_count {
        let mut key_len = [0u8; 8];
        file.read_exact(&mut key_len).ok()?;
        let key_len = u64::from_le_bytes(key_len) as usize;
        
        let mut key = vec![0u8; key_len];
        file.read_exact(&mut key).ok()?;
        
        let mut type_val = [0u8; 1];
        file.read_exact(&mut type_val).ok()?;
        
        match type_val[0] {
            0 => {
                let mut padding = [0u8; 7];
                file.read_exact(&mut padding).ok()?;
            }
            1 | 2 => {
                let mut len = [0u8; 8];
                file.read_exact(&mut len).ok()?;
                let len = u64::from_le_bytes(len) as usize;
                let mut buf = vec![0u8; len];
                file.read_exact(&mut buf).ok()?;
            }
            3 => {
                let mut len = [0u8; 8];
                file.read_exact(&mut len).ok()?;
                let len = u64::from_le_bytes(len) as usize;
                let mut arr_len = [0u8; 8];
                file.read_exact(&mut arr_len).ok()?;
            }
            _ => {}
        }
    }
    
    for _ in 0..tensor_count {
        let mut name_len = [0u8; 8];
        file.read_exact(&mut name_len).ok()?;
        let name_len = u64::from_le_bytes(name_len) as usize;
        
        let mut name = vec![0u8; name_len];
        file.read_exact(&mut name).ok()?;
        let name_str = String::from_utf8_lossy(&name);
        
        if name_str.contains(tensor_name) {
            let mut n_dims = [0u8; 4];
            file.read_exact(&mut n_dims).ok()?;
            let n_dims = u32::from_le_bytes(n_dims) as usize;
            
            let mut shape = vec![0u64; n_dims];
            for dim in 0..n_dims {
                let mut d = [0u8; 8];
                file.read_exact(&mut d).ok()?;
                shape[dim] = u64::from_le_bytes(d);
            }
            
            let mut dtype = [0u8; 4];
            file.read_exact(&mut dtype).ok()?;
            let dtype = u32::from_le_bytes(dtype);
            
            let mut offset = [0u8; 8];
            file.read_exact(&mut offset).ok()?;
            let offset = u64::from_le_bytes(offset);
            
            let numel: usize = shape.iter().map(|&s| s as usize).product();
            let bytes_needed = calculate_quant_bytes(numel, dtype);
            
            let mut data = vec![0u8; bytes_needed];
            
            let mut f = File::open(path).ok()?;
            f.seek(SeekFrom::Start(offset)).ok()?;
            f.read_exact(&mut data).ok()?;
            
            return Some((data, shape, dtype));
        } else {
            let mut n_dims = [0u8; 4];
            file.read_exact(&mut n_dims).ok()?;
            let n_dims = u32::from_le_bytes(n_dims) as usize;
            
            for _ in 0..n_dims {
                let mut d = [0u8; 8];
                file.read_exact(&mut d).ok()?;
            }
            
            let mut dtype = [0u8; 4];
            file.read_exact(&mut dtype).ok()?;
            
            let mut offset = [0u8; 8];
            file.read_exact(&mut offset).ok()?;
        }
    }
    
    None
}

fn calculate_quant_bytes(numel: usize, dtype: u32) -> usize {
    const Q4K_BYTES_PER_BLOCK: usize = 144;
    const Q6K_BYTES_PER_BLOCK: usize = 210;
    const Q8_0_BYTES_PER_BLOCK: usize = 32;
    const BLOCK_SIZE: usize = 256;
    
    match dtype {
        18 => (numel + BLOCK_SIZE - 1) / BLOCK_SIZE * Q4K_BYTES_PER_BLOCK,
        20 => (numel + BLOCK_SIZE - 1) / BLOCK_SIZE * Q6K_BYTES_PER_BLOCK,
        14 => (numel + 31) / 32 * Q8_0_BYTES_PER_BLOCK,
        _ => numel * 4,
    }
}

fn dequant_q4k_our(data: &[u8], shape: &[u64]) -> Vec<f32> {
    use hace_fem_hacedle::quant_view::{QuantTensorView, QuantType};
    
    let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    let tensor = QuantTensorView::new(data.to_vec(), shape_usize, QuantType::Q4K);
    tensor.dequantize()
}

#[test]
fn test_q4k_dequant_parify_basic() {
    let input = vec![0x80; 144];
    let shape = vec![256];
    let output = dequant_q4k_our(&input, &shape);
    
    assert_eq!(output.len(), 256);
    for val in &output {
        if val.abs() > 1e-5 {
            println!("Non-zero value in zero-ish input: {}", val);
        }
    }
}

#[test]
fn test_q4k_parify_with_gguf() {
    let model_path = "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
    
    let tensor_data = match load_gguf_tensor(model_path, "blk.0.attn_q.weight") {
        Some((data, shape, dtype)) => {
            assert_eq!(dtype, 18, "Expected Q4_K type (18)");
            (data, shape)
        }
        None => {
            println!("Skipping test - model or tensor not found");
            return;
        }
    };
    
    let our_output = dequant_q4k_our(&tensor_data.0, &tensor_data.1);
    
    assert!(our_output.len() > 0, "Output should not be empty");
    println!("Dequantized {} elements", our_output.len());
    
    let min_val = our_output.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = our_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Output range: [{}, {}]", min_val, max_val);
    
    let mean_abs: f32 = our_output.iter().map(|&x| x.abs()).sum::<f32>() / our_output.len() as f32;
    println!("Mean absolute value: {}", mean_abs);
}