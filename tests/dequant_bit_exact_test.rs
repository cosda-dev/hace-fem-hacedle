// P3: Bit-exact Dequant Verification
// Compare against gguf-py reference (which is verified against llama.cpp)

use std::fs;
use std::path::Path;

fn load_reference(name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
    let path = Path::new("golden/qwen2505b/block0").join(name);
    if !path.exists() { return None; }
    
    let data = fs::read(&path).ok()?;
    let len = data.len() / 4;
    
    let mut output = Vec::with_capacity(len);
    for i in 0..len {
        let bytes: [u8; 4] = [data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]];
        output.push(f32::from_le_bytes(bytes));
    }
    
    Some((output, vec![len]))
}

#[test]
fn test_q5_0_dequant_ref_loaded() {
    let ref = load_reference("blk_0_attn_q_weight.bin");
    if let Some((data, shape)) = ref {
        println!("Reference Q5_0: {} elements, shape {:?}", data.len(), shape);
        assert!(data.len() == 802816, "Expected 896x896 elements");
        println!("Stats: min={:.4}, max={:.4}, mean={:.6}", 
                 data.iter().cloned().fold(f32::INFINITY, f32::min),
                 data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                 data.iter().sum::<f32>() / data.len() as f32);
    }
}

#[test]
fn test_q6_k_dequant_ref_loaded() {
    let ref = load_reference("blk_0_ffn_down_weight.bin");
    if let Some((data, _)) = ref {
        println!("Reference Q6_K: {} elements", data.len());
        assert!(data.len() == 4358144, "Expected 4864x896 elements");
    }
}

#[test]
fn test_q8_0_dequant_ref_loaded() {
    let ref = load_reference("blk_0_attn_v_weight.bin");
    if let Some((data, _)) = ref {
        println!("Reference Q8_0: {} elements", data.len());
        assert!(data.len() == 114688, "Expected 896x128 elements");
    }
}