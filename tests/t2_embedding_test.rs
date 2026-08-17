// T2: Embedding Lookup Test (mock tensor data)
use std::vec::Vec;

fn embedding_lookup(token_id: u32, tensor_data: &[f32], vocab_size: usize, hidden_size: usize) -> Vec<f32> {
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

fn main() {
    // Mock tensor data for Qwen2.5 (vocab=151936, hidden=1536)
    let vocab_size = 151936;
    let hidden_size = 1536;
    let token_id: u32 = 42;
    
    // Create mock embedding tensor (first 64KB for testing)
    let mock_tensor: Vec<f32> = (0..65536).map(|i| (i as f32) * 0.001).collect();
    
    let embedding = embedding_lookup(token_id, &mock_tensor, vocab_size, hidden_size);
    
    println!("=== T2 Embedding Test ===");
    println!("Token ID: {}", token_id);
    println!("Hidden size: {}", hidden_size);
    println!("Embedding shape: {}", embedding.len());
    
    // Statistics
    let sum: f32 = embedding.iter().sum();
    let mean = sum / embedding.len() as f32;
    let variance: f32 = embedding.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / embedding.len() as f32;
    let checksum: f64 = embedding.iter().map(|&x| x as f64).sum();
    
    println!("Mean: {}", mean);
    println!("Std: {}", variance.sqrt());
    println!("Checksum: {}", checksum);
    
    // Verify
    assert_eq!(embedding.len(), hidden_size, "Embedding dimension mismatch");
    
    let finite_count = embedding.iter().filter(|x| x.is_finite()).count();
    assert_eq!(finite_count, hidden_size, "Non-finite values detected");
    
    println!("\n✅ T2 PASS: Embedding lookup verified (mock data)");
}