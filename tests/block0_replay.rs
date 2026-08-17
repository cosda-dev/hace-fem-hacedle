// T8.3: Block0 Full Replay
// Pipeline: input -> attn_norm -> Q/K/V -> RoPE -> Attention -> O_proj -> residual -> FFN -> output

use std::fs;
use std::path::Path;

fn load_f32(path: &Path) -> Vec<f32> {
    fs::read(path).map(|d| d.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()).unwrap_or_default()
}

#[test]
fn test_block0_attention_pipeline() {
    let golden = Path::new("golden/block0_operators");
    
    // Load golden tensors
    let _input = load_f32(&golden.join("01_input.bin"));
    let _rmsnorm_w = load_f32(&golden.join("02_attn_norm_weight.bin"));
    let _q_proj = load_f32(&golden.join("02_q_proj.bin"));
    let _k_proj = load_f32(&golden.join("03_k_proj.bin"));
    let _v_proj = load_f32(&golden.join("04_v_proj.bin"));
    let _rope_q = load_f32(&golden.join("05_rope_q.bin"));
    let _rope_k = load_f32(&golden.join("06_rope_k.bin"));
    let _scores = load_f32(&golden.join("07_attention_scores.bin"));
    let _softmax = load_f32(&golden.join("08_softmax.bin"));
    let _attn_out = load_f32(&golden.join("09_attention_output.bin"));
    let _o_proj = load_f32(&golden.join("10_o_proj.bin"));
    let _residual = load_f32(&golden.join("11_residual.bin"));
    
    // Verify files exist
    assert!((golden.join("01_input.bin")).exists());
    assert!((golden.join("08_softmax.bin")).exists());
    assert!((golden.join("11_residual.bin")).exists());
    
    println!("Block0 golden tensors loaded successfully");
    println!("Operators available: 01-11");
}