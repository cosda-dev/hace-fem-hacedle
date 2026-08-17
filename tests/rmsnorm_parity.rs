// RMSNorm Parity Test - Compare against reference implementation
// Run: cargo test --test rmsnorm_parity --features std -- --nocapture

fn rmsnorm_reference(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let len = input.len().min(weight.len());
    let ss: f32 = input[..len].iter().map(|&x| x * x).sum();
    let rms = (ss / len as f32 + eps).sqrt().recip();
    
    input[..len].iter().zip(weight[..len].iter())
        .map(|(&x, &w)| x * w * rms)
        .collect()
}

#[test]
fn test_rmsnorm_vs_reference() {
    use hace_fem_hacedle::quant_view::NativeBackend;
    
    let test_cases = vec![
        (vec![1.0, 2.0, 3.0, 4.0], vec![1.0; 4]),
        (vec![0.5, -0.5, 0.5, -0.5], vec![1.0; 4]),
        (vec![10.0; 32], vec![1.0; 32]),
        (vec![0.001, 0.002, 0.003, 0.004], vec![1.0; 4]),
    ];
    
    let backend = NativeBackend::new();
    
    for (input, weight) in test_cases {
        let mut our_output = vec![0.0f32; input.len()];
        let ref_output = rmsnorm_reference(&input, &weight, 1e-6);
        
        backend.rmsnorm(&input, &weight, &mut our_output);
        
        for i in 0..input.len() {
            let diff = (our_output[i] - ref_output[i]).abs();
            assert!(diff < 1e-5, "RMSNorm mismatch at {}: our={}, ref={}, diff={}", 
                    i, our_output[i], ref_output[i], diff);
        }
    }
}

#[test]
fn test_rmsnorm_f32_roundtrip() {
    let input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32];
    let weight = vec![1.0f32; 5];
    
    let normalized = rmsnorm_reference(&input, &weight, 1e-6);
    
    let ss_original: f32 = input.iter().map(|&x| x * x).sum();
    let ss_normalized: f32 = normalized.iter().map(|&x| x * x).sum();
    
    let rms_original = (ss_original / 5.0 + 1e-6).sqrt();
    let expected_rms_normalized = 5.0f32 / rms_original;
    
    let actual_rms_normalized = (ss_normalized / 5.0).sqrt();
    
    assert!((actual_rms_normalized - expected_rms_normalized).abs() < 1e-5);
}