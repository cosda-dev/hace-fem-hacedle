// Directive A.1: Operator Replay Engine
// Each operator xuất report chi tiết, không chỉ PASS/FAIL

use std::fs;
use std::path::Path;

#[derive(Debug)]
pub struct ReplayReport {
    pub operator_id: &'static str,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub cosine_similarity: f64,
    pub sha256_golden: String,
    pub sha256_runtime: String,
    pub shape_match: bool,
}

fn load_f32(path: &Path) -> Vec<f32> {
    if !path.exists() { return vec![]; }
    let data = fs::read(&path).unwrap();
    data.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn compare_tensors(runtime: &[f32], golden: &[f32]) -> ReplayReport {
    let min_len = runtime.len().min(golden.len()).max(1) as f64;
    
    let max_err: f32 = runtime[..min_len as usize]
        .iter()
        .zip(golden[..min_len as usize].iter())
        .map(|(a, g)| (a - g).abs())
        .fold(0.0, f32::max);
    
    let mean_err: f32 = runtime[..min_len as usize]
        .iter()
        .zip(golden[..min_len as usize].iter())
        .map(|(a, g)| (a - g).abs())
        .sum::<f32>() / min_len as f32;
    
    let dot: f32 = runtime[..min_len as usize]
        .iter()
        .zip(golden[..min_len as usize].iter())
        .map(|(a, g)| a * g)
        .sum();
    
    let norm_r: f32 = runtime[..min_len as usize].iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_g: f32 = golden[..min_len as usize].iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = if norm_r > 0.0 && norm_g > 0.0 {
        dot / (norm_r * norm_g)
    } else { 1.0 };
    
    use std::sha2::{Sha256, Digest};
    
    let sha256_golden = {
        let mut hasher = Sha256::new();
        hasher.update(bytemuck::cast_slice(&golden[..min_len as usize]));
        format!("{:064x}", hasher.finalize())
    };
    
    let sha256_runtime = {
        let mut hasher = Sha256::new();
        hasher.update(bytemuck::cast_slice(&runtime[..min_len as usize]));
        format!("{:064x}", hasher.finalize())
    };
    
    ReplayReport {
        operator_id: "unknown",
        max_abs_error: max_err as f64,
        mean_abs_error: mean_err as f64,
        cosine_similarity: cos as f64,
        sha256_golden,
        sha256_runtime,
        shape_match: runtime.len() == golden.len(),
    }
}

fn save_report(report: &ReplayReport, path: &Path) {
    let yaml = format!(
        "operator_id: {}\nshape_match: {}\nmax_abs_error: {:.10}\nmean_abs_error: {:.10}\ncosine_similarity: {:.10}\nsha256_golden: {}\nsha256_runtime: {}",
        report.operator_id,
        report.shape_match,
        report.max_abs_error,
        report.mean_abs_error,
        report.cosine_similarity,
        report.sha256_golden,
        report.sha256_runtime
    );
    let _ = fs::write(path, yaml);
}