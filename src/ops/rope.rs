// RoPE operator for parity testing - Qwen2.5 specific (theta=1000000)

use crate::alloc_exports::*;

pub fn apply_rope_pairwise(input: &[f32], pos: usize, dim: usize) -> Vec<f32> {
    let theta: f32 = 1000000.0; // Qwen2.5
    let n_total = input.len();
    let n_heads = n_total / dim;
    let mut output = vec![0.0f32; n_total];
    
    for head in 0..n_heads {
        let base = head * dim;
        for i in 0..dim / 2 {
            let freq = theta.powf(2.0 * (i as f32) / (dim as f32));
            let angle = pos as f32 / freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            
            let idx0 = base + i;
            let idx1 = base + i + dim / 2;
            
            if idx1 < n_total {
                let x0 = input[idx0];
                let x1 = input[idx1];
                output[idx0] = x0 * cos_val - x1 * sin_val;
                output[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
    output
}