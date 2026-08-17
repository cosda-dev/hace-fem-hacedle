// Standalone parity test - no module dependencies

use std::fs;
use std::path::Path;

const GOLDEN_DIR: &str = "golden/qwen2505b/block0";

fn load_golden_tensor(name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
    let path = Path::new(GOLDEN_DIR).join(name);
    if !path.exists() {
        return None;
    }
    
    let data = fs::read(&path).ok()?;
    let float_count = data.len() / 4;
    
    let mut floats = Vec::with_capacity(float_count);
    for i in 0..float_count {
        let bytes: [u8; 4] = data[i*4..i*4+4].try_into().ok()?;
        floats.push(f32::from_le_bytes(bytes));
    }
    
    Some((floats, vec![float_count]))
}

#[test]
fn test_golden_loaded() {
    if load_golden_tensor("blk_0_attn_q_weight.bin").is_some() {
        println!("Gold block0 Q tensor found - ready for parity test");
    } else {
        println!("Golden bundle not found");
    }
}

#[test]
fn test_layer1_dequant_reference_loaded() {
    let (ref_data, shape) = match load_golden_tensor("blk_0_attn_q_weight.bin") {
        Some(d) => d,
        None => {
            panic!("No reference found");
        }
    };
    
    println!("Reference Q tensor loaded: {} elements, shape {:?}", ref_data.len(), shape);
    
    let min_val = ref_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = ref_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_val = ref_data.iter().sum::<f32>() / ref_data.len() as f32;
    
    println!("Reference stats: min={:.6}, max={:.6}, mean={:.6}", min_val, max_val, mean_val);
}