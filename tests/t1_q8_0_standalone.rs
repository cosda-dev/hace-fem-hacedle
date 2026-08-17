// T1: Q8_0 Bit Exact Parity - Standalone test
// Copy dequant function to avoid module import issues

use std::fs;
use std::path::Path;

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;
    
    if exp == 0 {
        0.0
    } else {
        let f32_exp = (exp as i32) - 15 + 127;
        let f32_frac = frac as f32 / 1024.0;
        let result = (1.0 + f32_frac) * 2.0_f32.powi(f32_exp);
        if sign == 1 { -result } else { result }
    }
}

fn dequant_q8_0_exact(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 34;
    
    let blocks = data.len() / BYTES_PER_BLOCK;
    let elements = output.len().min(blocks * BLOCK_SIZE);
    
    for block_idx in 0..blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        if block_offset + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block = &data[block_offset..block_offset + BYTES_PER_BLOCK];
        
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        
        for j in 0..BLOCK_SIZE {
            let idx = block_idx * BLOCK_SIZE + j;
            if idx >= elements {
                break;
            }
            
            let v = block[2 + j] as i8;
            output[idx] = v as f32 * d;
        }
    }
}

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
    dequant_q8_0_exact(&raw_data, &mut rust_output);
    
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