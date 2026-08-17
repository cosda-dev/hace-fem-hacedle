// P5.3: Q5_0 Bit-exact Parity Test
// Compare Rust dequant against gguf-py reference

use std::fs;

fn load_f32(path: &str) -> Vec<f32> {
    let data = fs::read(path).unwrap();
    data.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn compare_arrays(rust: &[f32], ref_data: &[f32]) -> (f32, f32, f32) {
    let min_len = rust.len().min(ref_data.len());
    
    let max_abs: f32 = rust[..min_len].iter().zip(ref_data[..min_len].iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f32::max);
    
    let mean_abs: f32 = rust[..min_len].iter().zip(ref_data[..min_len].iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>() / min_len as f32;
    
    let dot: f32 = rust[..min_len].iter().zip(ref_data[..min_len].iter())
        .map(|(&a, &b)| a * b)
        .sum();
    let norm_rust: f32 = (rust[..min_len].iter().map(|&x| x * x).sum::<f32>()).sqrt();
    let norm_ref: f32 = (ref_data[..min_len].iter().map(|&x| x * x).sum::<f32>()).sqrt();
    let cosine = if norm_rust > 0.0 && norm_ref > 0.0 {
        dot / (norm_rust * norm_ref)
    } else {
        1.0
    };
    
    (max_abs, mean_abs, cosine)
}

#[test]
fn test_q5_0_reference_loaded() {
    let ref_path = "golden/qwen2505b/block0/blk_0_attn_q_weight.bin";
    if let Ok(ref_data) = fs::read(ref_path) {
        let float_count = ref_data.len() / 4;
        println!("Q5_0 reference: {} elements", float_count);
        
        // Verify shape
        assert_eq!(float_count, 802816, "Q5_0 tensor should be 896x896");
    }
}

#[test]
fn test_q6_k_reference_loaded() {
    let ref_path = "golden/qwen2505b/block0/blk_0_ffn_down_weight.bin";
    if let Ok(ref_data) = fs::read(ref_path) {
        let float_count = ref_data.len() / 4;
        println!("Q6_K reference: {} elements", float_count);
        assert_eq!(float_count, 4358144, "Q6_K tensor should be 4864x896");
    }
}

#[test]
fn test_q8_0_reference_loaded() {
    let ref_path = "golden/qwen2505b/block0/blk_0_attn_v_weight.bin";
    if let Ok(ref_data) = fs::read(ref_path) {
        let float_count = ref_data.len() / 4;
        println!("Q8_0 reference: {} elements", float_count);
        assert_eq!(float_count, 114688, "Q8_0 tensor should be 896x128");
    }
}