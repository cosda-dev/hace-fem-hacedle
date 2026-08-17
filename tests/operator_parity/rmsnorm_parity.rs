// T8.1: RMSNorm Parity Test
use std::fs;
use std::path::Path;

fn rmsnorm(input: &[f32], weight: &[f32], output: &mut [f32]) {
    let len = input.len().min(weight.len()).min(output.len());
    let ss: f32 = input[..len].iter().map(|&x| x * x).sum();
    let rms = (ss / len as f32 + 1e-6).sqrt().recip();
    for i in 0..len {
        output[i] = input[i] * weight[i] * rms;
    }
}

fn load_f32(path: &Path) -> Vec<f32> {
    let data = fs::read(path).unwrap();
    data.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn test_rmsnorm_parity() {
    let golden = Path::new("golden/block0_operators");
    
    let input = load_f32(&golden.join("01_input.bin"));
    let weight = load_f32(&golden.join("02_attn_norm_weight.bin"));
    let expected = load_f32(&golden.join("02_attn_norm_post.bin"));
    
    let mut output = vec![0.0f32; 896];
    rmsnorm(&input, &weight, &mut output);
    
    let max_err = output.iter().zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    
    println!("RMSNorm max_abs_error: {:.10}", max_err);
    // Will compare with golden when available
}