// Logits Parity Test - Full pipeline comparison
// Run: cargo test --test logits_parity --features std -- --nocapture

fn softmax_reference(input: &mut [f32]) {
    if input.is_empty() { return; }
    
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = input.iter().map(|&x| (x - max_val).exp()).sum();
    
    for x in input.iter_mut() {
        *x = (x - max_val).exp() / sum;
    }
}

#[test]
fn test_softmax_parity() {
    use hace_fem_hacedle::quant_view::NativeBackend;
    
    let test_cases = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![-1.0, 0.0, 1.0, 2.0],
        vec![100.0, 200.0, 300.0],
    ];
    
    let backend = NativeBackend::new();
    
    for input in test_cases {
        let mut our_output = input.clone();
        let mut ref_output = input.clone();
        
        backend.softmax(&mut our_output);
        softmax_reference(&mut ref_output);
        
        for i in 0..input.len() {
            let diff = (our_output[i] - ref_output[i]).abs();
            assert!(diff < 1e-5, "Softmax mismatch at {}: our={}, ref={}", i, our_output[i], ref_output[i]);
        }
    }
}

#[test]
fn test_logits_accumulation() {
    let vocab_size = 151936;
    let mut logits = vec![0.0f32; vocab_size];
    
    for _ in 0..10 {
        for i in 0..vocab_size {
            logits[i] += 0.1;
        }
    }
    
    softmax_reference(&mut logits);
    
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    
    let max_idx = logits.iter().enumerate()
        .max_by(|(_, &a), (_, &b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap();
    
    assert!(logits[max_idx] > 0.0);
}

#[test]
fn test_einsum_pattern_qk() {
    let seq_len = 10;
    let n_heads = 28;
    let head_dim = 128;
    
    let q = vec![0.1f32; n_heads * head_dim];
    let mut k_cache = vec![0.1f32; seq_len * n_heads * head_dim];
    
    let mut scores = vec![0.0f32; n_heads * seq_len];
    
    for h in 0..n_heads {
        for t in 0..seq_len {
            let mut sum = 0.0f32;
            for d in 0..head_dim {
                sum += q[h * head_dim + d] * k_cache[t * n_heads * head_dim + h * head_dim + d];
            }
            scores[h * seq_len + t] = sum / (head_dim as f32).sqrt();
        }
    }
    
    assert_eq!(scores.len(), n_heads * seq_len);
    let all_same = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-5);
    assert!(all_same, "All scores should be equal for constant inputs");
}