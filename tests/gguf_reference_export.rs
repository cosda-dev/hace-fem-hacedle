// GGUF Reference Export Test
// Export tensors from GGUF for parity comparison

use std::fs;
use std::path::Path;

#[test]
fn test_export_block0_tensors() {
    if let Ok(loader) = GgufLoader::load("D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf") {
        let tensor_names = vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight", 
            "blk.0.ffn_down.weight",
        ];
        
        for name in tensor_names {
            if loader.get_tensor(name).is_some() {
                println!("Found: {}", name);
            } else {
                println!("Missing: {}", name);
            }
        }
    }
}

#[test]
fn test_model_spec_extraction() {
    let loader = GgufLoader::load("D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
        .expect("Failed to load model");
    
    let spec = ModelSpec::from_metadata(&[]);
    if let Some(s) = spec {
        println!("Model spec extracted:");
        println!("  arch: {}", s.arch);
        println!("  hidden_size: {}", s.hidden_size);
        println!("  n_head: {}", s.n_head);
        println!("  rope_theta: {}", s.rope_theta);
    }
}