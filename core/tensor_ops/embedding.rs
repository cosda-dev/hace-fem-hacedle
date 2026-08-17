use crate::alloc_exports::*;

/// Embedding lookup for token ID
pub fn embedding_lookup(token_id: u32, tensor_data: &[f32], vocab_size: usize, hidden_size: usize) -> Vec<f32> {
    if token_id as usize >= vocab_size {
        return vec![0.0; hidden_size];
    }
    
    let start = token_id as usize * hidden_size;
    let end = start + hidden_size;
    
    if end > tensor_data.len() {
        return vec![0.0; hidden_size];
    }
    
    tensor_data[start..end].to_vec()
}

/// Compute embedding statistics (for golden test)
pub fn embedding_stats(embedding: &[f32]) -> (f32, f32, f64) {
    let sum: f32 = embedding.iter().sum();
    let mean = sum / embedding.len() as f32;
    let variance: f32 = embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / embedding.len() as f32;
    let checksum: f64 = embedding.iter().map(|&x| x as f64).sum();
    
    (mean, variance.sqrt(), checksum)
}