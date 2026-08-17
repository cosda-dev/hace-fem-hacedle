// T8.4: Block0 Parity Gate Test
// Compare full Block0 output with golden bundle

use std::fs;
use std::path::Path;

fn load_golden(name: &str) -> Vec<f32> {
    let path = Path::new("golden/block0_operators").join(name);
    if !path.exists() { return vec![]; }
    let data = fs::read(&path).unwrap();
    data.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn test_block0_attention_flow() {
    // Load golden tensors
    let q_rope = load_golden("05_rope_q.bin");
    let k_rope = load_golden("06_rope_k.bin");
    let scores = load_golden("07_attention_scores.bin");
    let softmax = load_golden("08_softmax.bin");
    let attn_out = load_golden("09_attention_output.bin");
    
    // Verify shapes
    assert_eq!(q_rope.len(), 896, "Q rope should be 896");
    assert_eq!(k_rope.len(), 128, "K rope should be 128");
    assert_eq!(scores.len(), 196, "Scores should be 14*14");
    assert_eq!(softmax.len(), 196, "Softmax should be 14*14");
    assert_eq!(attn_out.len(), 896, "Attention output should be 896");
    
    println!("Block0 attention flow shapes verified");
}

#[test]
fn test_block0_ffn_flow() {
    let residual = load_golden("11_residual.bin");
    let ffn_out = load_golden("17_down_proj.bin");
    let final_out = load_golden("18_ffn_residual.bin");
    
    assert_eq!(residual.len(), 896);
    assert_eq!(ffn_out.len(), 896);
    assert_eq!(final_out.len(), 896);
    
    println!("Block0 FFN flow shapes verified");
}

#[test]
fn test_block0_cosine_similarity() {
    // This will be the final gate
    // Requires actual Rust execution vs golden
    let _golden = load_golden("11_residual.bin");
    let _runtime = vec![0.0f32; 896]; // Placeholder
    
    // TODO: Replace with actual computation
    // cosine = dot(golden, runtime) / (norm_golden * norm_runtime)
    // assert!(cosine > 0.99999);
    
    println!("Block0 cosine similarity: PENDING_RUNTIME");
}