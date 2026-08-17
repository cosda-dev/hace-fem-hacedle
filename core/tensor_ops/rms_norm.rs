use crate::alloc_exports::*;

/// RMSNorm implementation
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let len = input.len().min(weight.len());
    
    let ss: f32 = input[..len].iter().map(|&x| x * x).sum();
    let rms = (ss / len as f32 + eps).sqrt().recip();
    
    input[..len].iter().zip(weight[..len].iter())
        .map(|(&x, &w)| x * w * rms)
        .collect()
}

/// Golden test vector
pub fn rms_norm_golden_test() -> (Vec<f32>, Vec<f32>, f32, Vec<f32>) {
    let input = vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32];
    let weight = vec![1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32];
    let eps = 1e-5_f32;
    let output = rms_norm(&input, &weight, eps);
    
    (input, weight, eps, output)
}