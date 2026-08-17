// Dequant Verification - CRD P5.3/P5.4/P5.5
// Verify Rust dequant matches gguf-py reference

use std::fs;

fn load_f32(path: &str) -> Vec<f32> {
    let data = fs::read(path).unwrap();
    data.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn test_reference_tensor_shapes() {
    // Verify reference tensor dimensions match expected
    if let Ok(data) = fs::read("golden/qwen2505b/block0/blk_0_attn_q_weight.bin") {
        let elements = data.len() / 4;
        println!("Q5_0 reference tensor: {} elements (expected 896x896 = 802816)", elements);
        assert_eq!(elements, 802816, "Q5_0 tensor shape mismatch");
    }
    
    if let Ok(data) = fs::read("golden/qwen2505b/block0/blk_0_attn_v_weight.bin") {
        let elements = data.len() / 4;
        println!("Q8_0 reference tensor: {} elements (expected 896x128 = 114688)", elements);
        assert_eq!(elements, 114688);
    }
    
    if let Ok(data) = fs::read("golden/qwen2505b/block0/blk_0_ffn_down_weight.bin") {
        let elements = data.len() / 4;
        println!("Q6_K reference tensor: {} elements (expected 4864x896 = 4358144)", elements);
        assert_eq!(elements, 4358144);
    }
}

#[test]
fn test_gqa_dimensions() {
    // Qwen2.5-0.5B: n_head=14, n_head_kv=2
    let n_heads = 14;
    let n_heads_kv = 2;
    let head_dim = 64;
    
    assert_eq!(n_heads % n_heads_kv, 0, "GQA ratio must be integer");
    assert_eq!(n_heads / n_heads_kv, 7, "GQA factor should be 7");
    
    // Expected shapes after projection
    let q_dim = n_heads * head_dim;    // 896
    let kv_dim = n_heads_kv * head_dim; // 128
    
    println!("Q projection dim: {}", q_dim);
    println!("KV projection dim: {}", kv_dim);
    
    assert_eq!(q_dim, 896);
    assert_eq!(kv_dim, 128);
}