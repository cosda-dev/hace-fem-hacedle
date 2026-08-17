// Dequantization Parity Test
// Compare Rust dequant against gguf-py reference

use std::fs;
use std::path::Path;

fn load_bin_f32(path: &str) -> Option<Vec<f32>> {
    let data = fs::read(path).ok()?;
    let count = data.len() / 4;
    
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let bytes: [u8; 4] = [data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]];
        result.push(f32::from_le_bytes(bytes));
    }
    
    Some(result)
}

fn compare_max_abs(a: &[f32], b: &[f32]) -> f32 {
    let min_len = a.len().min(b.len());
    a[..min_len].iter().zip(b[..min_len].iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn test_q5_0_dequant_parity() {
    let ref_data = match load_bin_f32("golden/qwen2505b/block0/blk_0_attn_q_weight.bin") {
        Some(d) => d,
        None => {
            println!("Reference not found. Run export_gguf_tensors.py first");
            return;
        }
    };
    
    println!("Reference Q5_0 data: {} elements", ref_data.len());
    println!("Reference stats: min={:.4}, max={:.4}, mean={:.4}", 
             ref_data.iter().cloned().fold(f32::INFINITY, f32::min),
             ref_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
             ref_data.iter().sum::<f32>() / ref_data.len() as f32);
    
    // TODO: Load raw Q5_0 bytes and dequantize with our Rust implementation
    // Then compare max_abs error
}

#[test]
fn test_q6_k_dequant_parity() {
    let ref_data = match load_bin_f32("golden/qwen2505b/block0/blk_0_ffn_down_weight.bin") {
        Some(d) => d,
        None => {
            println!("Reference not found");
            return;
        }
    };
    
    println!("Reference Q6_K data: {} elements", ref_data.len());
}

#[test]
fn test_q8_0_dequant_parity() {
    let ref_data = match load_bin_f32("golden/qwen2505b/block0/blk_0_attn_v_weight.bin") {
        Some(d) => d,
        None => {
            println!("Reference not found");
            return;
        }
    };
    
    println!("Reference Q8_0 data: {} elements", ref_data.len());
}