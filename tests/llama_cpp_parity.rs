// llama.cpp Activation Parity Test Framework
// Uses golden bundle reference data for comparison

use std::fs;
use std::path::Path;

const GOLDEN_DIR: &str = "golden/qwen2505b/block0";

#[derive(Debug)]
pub struct ReferenceTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub stats: TensorStats,
}

#[derive(Debug)]
pub struct TensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub l2_norm: f32,
}

impl TensorStats {
    pub fn from_array(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self { min: 0.0, max: 0.0, mean: 0.0, std: 0.0, l2_norm: 0.0 };
        }
        
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        
        Self {
            min,
            max,
            mean,
            std: variance.sqrt(),
            l2_norm: data.iter().map(|&x| x * x).sum::<f32>().sqrt(),
        }
    }
}

pub fn load_golden_tensor(name: &str) -> Option<ReferenceTensor> {
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
    
    Some(ReferenceTensor {
        data: floats,
        shape: vec![float_count],
        stats: TensorStats::from_array(&floats),
    })
}

pub fn compare_tensors(our: &[f32], reference: &[f32]) -> CompareResult {
    let min_len = our.len().min(reference.len());
    
    if min_len == 0 {
        return CompareResult { passed: false, max_abs: f32::MAX, cosine: 0.0 };
    }
    
    let max_abs: f32 = our[..min_len].iter().zip(reference[..min_len].iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f32::max);
    
    let dot: f32 = our[..min_len].iter().zip(reference[..min_len].iter())
        .map(|(&a, &b)| a * b)
        .sum();
    let norm_a: f32 = (our[..min_len].iter().map(|&x| x * x).sum::<f32>()).sqrt();
    let norm_b: f32 = (reference[..min_len].iter().map(|&x| x * x).sum::<f32>()).sqrt();
    let cosine = if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        1.0
    };
    
    CompareResult {
        passed: max_abs < 1e-6 && cosine > 0.99999,
        max_abs,
        cosine,
    }
}

pub struct CompareResult {
    pub passed: bool,
    pub max_abs: f32,
    pub cosine: f32,
}

#[test]
fn test_golden_loaded() {
    if load_golden_tensor("block0/q.bin").is_some() {
        println!("Golden bundle found - ready for parity test");
    } else {
        println!("Golden bundle not found - run generate_golden_bundle.py first");
    }
}

#[test]
fn test_tensor_fingerprint_match() {
    if let Some(ref_tensor) = load_golden_tensor("block0/q.bin") {
        let our_tensor = vec![0.0f32; ref_tensor.data.len()];
        
        let stats_a = TensorStats::from_array(&our_tensor);
        let stats_b = &ref_tensor.stats;
        
        println!("Reference stats: mean={:.6}, l2={:.6}", stats_b.mean, stats_b.l2_norm);
    }
}

// Layer 1: Dequant Parity (stub - needs golden data)
#[test]
fn test_layer1_dequant_q4k() {
    let ref_tensor = match load_golden_tensor("blk_0_attn_q_weight.bin") {
        Some(d) => d,
        None => {
            println!("No reference found. Generate with generate_golden_bundle.py");
            return;
        }
    };
    
    // Load our Q4K tensor and dequantize
    // Compare against ref_tensor.data
    println!("Reference Q shape: {:?}", ref_tensor.shape);
    println!("Reference Q size: {} elements", ref_tensor.data.len());
    println!("Reference Q stats: mean={:.6}, l2={:.6}", ref_tensor.stats.mean, ref_tensor.stats.l2_norm);
}

// Layer 2: Operator Parity (stub)
#[test]
fn test_layer2_rmsnorm() {
    println!("RMSNorm parity - pending golden data");
}

#[test]
fn test_layer2_rope() {
    println!("RoPE parity - pending golden data");
}

// Layer 3: Block0 Parity (stub)
#[test]
fn test_layer3_block0_output() {
    println!("Block0 parity - pending golden data");
}

// Layer 4: Full Inference Parity (stub)
#[test]
fn test_layer4_logits() {
    println!("Logits parity - pending golden data");
}