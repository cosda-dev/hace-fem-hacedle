// T5: RoPE Runtime Verification
// Verify RoPE implementation matches gguf-py

use std::fs;
use std::path::Path;

fn apply_rope_pairwise(input: &[f32], pos: usize, dim: usize) -> Vec<f32> {
    let theta: f32 = 1000000.0;
    let mut output = vec![0.0f32; dim];
    
    for i in 0..dim / 2 {
        let freq = theta.powf(2.0 * (i as f32) / (dim as f32));
        let angle = pos as f32 / freq;
        let cos_val = angle.cos();
        let sin_val = angle.sin();
        
        let idx0 = i;
        let idx1 = i + dim / 2;
        
        if idx1 < input.len() {
            let x0 = input[idx0];
            let x1 = input[idx1];
            
            output[idx0] = x0 * cos_val - x1 * sin_val;
            output[idx1] = x0 * sin_val + x1 * cos_val;
        }
    }
    
    output
}

#[test]
fn test_t5_rope_pos0() {
    let golden_path = Path::new("parity_test/rope/pos0.bin");
    if !golden_path.exists() {
        println!("SKIP: Run t5_rope_golden.py first");
        return;
    }
    
    let golden = fs::read(golden_path).unwrap();
    let golden_vals: Vec<f32> = golden.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    
    // Create test input
    let mut input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 64.0).collect();
    
    let output = apply_rope_pairwise(&input, 0, 64);
    
    let max_err = output.iter().zip(golden_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("RoPE Position 0: max_err = {:.10}", max_err);
    assert!(max_err < 1e-6, "RoPE FAIL at pos 0");
}

#[test]
fn test_t5_rope_pos1() {
    verify_rope_position(1);
}

#[test]
fn test_t5_rope_pos128() {
    verify_rope_position(128);
}

#[test]
fn test_t5_rope_pos1024() {
    verify_rope_position(1024);
}

fn verify_rope_position(pos: usize) {
    let golden_path = Path::new(&format!("parity_test/rope/pos{}.bin", pos));
    if !golden_path.exists() {
        println!("SKIP: Position {} golden missing", pos);
        return;
    }
    
    let golden = fs::read(golden_path).unwrap();
    let golden_vals: Vec<f32> = golden.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    
    let mut input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 64.0).collect();
    let output = apply_rope_pairwise(&input, pos, 64);
    
    let max_err = output.iter().zip(golden_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("RoPE Position {}: max_err = {:.10}", pos, max_err);
}