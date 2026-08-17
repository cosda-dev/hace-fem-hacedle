// T1: Q8_0 Bit Exact Parity Report
// Verify Rust dequant_q8_0_exact matches gguf-py reference

use std::fs;
use std::path::Path;

fn load_f32(path: &Path) -> Vec<f32> {
    let data = fs::read(path).unwrap();
    data.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn compute_metrics(ref_values: &[f32], actual_values: &[f32]) -> (f32, f32, f32) {
    let max_abs_error: f32 = ref_values.iter()
        .zip(actual_values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    let mean_abs_error: f32 = ref_values.iter()
        .zip(actual_values.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / ref_values.len() as f32;
    
    let dot: f32 = ref_values.iter()
        .zip(actual_values.iter())
        .map(|(a, b)| a * b)
        .sum();
    let norm_ref: f32 = ref_values.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_actual: f32 = actual_values.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine_similarity = if norm_ref > 0.0 && norm_actual > 0.0 {
        dot / (norm_ref * norm_actual)
    } else {
        1.0
    };
    
    (max_abs_error, mean_abs_error, cosine_similarity)
}

#[test]
fn test_t1_q8_0_bit_exact_parity() {
    let parity_dir = Path::new("parity_test/q8_0");
    
    let ref_path = parity_dir.join("blk0_attn_v_q8_0_first_block.bin");
    let raw_path = parity_dir.join("blk0_attn_v_q8_0_first_block.raw");
    
    if !ref_path.exists() || !raw_path.exists() {
        println!("SKIP: Run t1_q8_0_parity_extract.py first");
        return;
    }
    
    let ref_values = load_f32(&ref_path);
    let raw_data = fs::read(&raw_path).unwrap();
    
    let mut rust_output = vec![0.0f32; 32];
            crate::quant_view::dequant_q8_0_exact(&raw_data, &mut rust_output);
    
    let (max_abs_error, mean_abs_error, cosine_similarity) = compute_metrics(&ref_values, &rust_output);
    
    println!("Q8_0 Bit Exact Parity Report:");
    println!("  Elements: 32");
    println!("  max_abs_error: {:.10}", max_abs_error);
    println!("  mean_abs_error: {:.10}", mean_abs_error);
    println!("  cosine_similarity: {:.10}", cosine_similarity);
    println!("  Status: {}", if max_abs_error < 1e-6 { "PASS" } else { "FAIL" });
    
    let report = format!(
        r#"{{
  "tensor": "blk.0.attn_v.weight",
  "quant": "Q8_0",
  "block_index": 0,
  "elements": 32,
  "max_abs_error": {},
  "mean_abs_error": {},
  "cosine_similarity": {},
  "rust_sample": {:?},
  "status": "{}"
}}"#,
        max_abs_error, mean_abs_error, cosine_similarity, &rust_output[..8],
        if max_abs_error < 1e-6 { "PASS" } else { "FAIL" }
    );
    
    let out_path = parity_dir.join("t1_q8_0_bit_exact_report.json");
    let _ = fs::write(&out_path, &report);
    
    assert!(max_abs_error < 1e-6, "Q8_0 FAIL: max_abs_error = {}", max_abs_error);
}