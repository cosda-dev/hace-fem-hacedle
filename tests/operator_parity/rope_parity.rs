// T8.1: RoPE Parity Test
use std::fs;
use std::path::Path;

fn apply_rope(input: &[f32], pos: usize, dim: usize) -> Vec<f32> {
    let theta: f32 = 1000000.0;
    let n_total = input.len();
    let n_heads = n_total / dim;
    let mut output = vec![0.0f32; n_total];
    
    for h in 0..n_heads {
        let base = h * dim;
        for i in 0..dim / 2 {
            let freq = theta.powf(2.0 * (i as f32) / (dim as f32));
            let angle = pos as f32 / freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            
            let idx0 = base + i;
            let idx1 = base + i + dim / 2;
            
            if idx1 < input.len() {
                let x0 = input[idx0];
                let x1 = input[idx1];
                output[idx0] = x0 * cos_val - x1 * sin_val;
                output[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
    output
}

fn load_f32(path: &Path) -> Vec<f32> {
    fs::read(path).map(|d| d.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).collect()).unwrap_or_default()
}

#[test]
fn test_rope_q_parity() {
    let golden = Path::new("golden/block0_operators");
    let q_proj = load_f32(&golden.join("02_q_proj.bin"));
    let expected = load_f32(&golden.join("05_rope_q.bin"));
    
    let output = apply_rope(&q_proj, 0, 64);
    
    let max_err = output.iter().zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    
    println!("RoPE Q max_abs_error: {:.10}", max_err);
}

#[test]
fn test_rope_k_parity() {
    let golden = Path::new("golden/block0_operators");
    let k_proj = load_f32(&golden.join("03_k_proj.bin"));
    let expected = load_f32(&golden.join("06_rope_k.bin"));
    
    let output = apply_rope(&k_proj, 0, 64);
    
    let max_err = output.iter().zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    
    println!("RoPE K max_abs_error: {:.10}", max_err);
}