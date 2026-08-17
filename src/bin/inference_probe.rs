// Inference Probe - Test real GGUF inference through hacedle
// Pipeline: GGUF -> Load weights -> Infer -> First token

use std::env;
use std::path::Path;

use hace_fem_hacedle::x::loader::gguf::GgufLoader;
use hace_fem_hacedle::x::provider::candle::{InferenceEngine, TokenizerEngine};

fn main() {
    let args: Vec<String> = env::args().collect();
    let prompt = args.get(1).map(|s| s.as_str()).unwrap_or("hello world, what your model name?");

    // Find model
    let model_paths = [
        "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "D:/host/llama-models/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    ];

    let mut model_path = None;
    for p in &model_paths {
        if Path::new(p).exists() {
            model_path = Some(*p);
            break;
        }
    }

    let path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("ERROR: No GGUF model found");
            for p in &model_paths {
                eprintln!("  Checked: {}", p);
            }
            std::process::exit(1);
        }
    };

    println!("=====================================");
    println!("Inference Probe - Hacedle Pipeline");
    println!("=====================================");
    println!("\nModel: {}", Path::new(path).file_name().unwrap().to_string_lossy());
    println!("Prompt: {}", prompt);

    // Load GGUF
    let loader = match GgufLoader::load(path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ERROR loading GGUF: {}", e);
            std::process::exit(1);
        }
    };

    println!("\nTensors found: {}", loader.tensor_count());

    // List key tensors
    let key_tensors = ["token_embd.weight", "output.weight", "tok-0", "tok-1"];
    for tensor_name in &key_tensors {
        if let Some(t) = loader.get_tensor(tensor_name) {
            println!("  {}: shape={:?}", tensor_name, t.shape);
        }
    }

    // Create inference engine
    let engine = InferenceEngine::default();

    // Get logits
    let logits = engine.infer_logits(prompt);

    // Find top 5
    let mut top5: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top5.truncate(5);

    println!("\nTop 5 logits:");
    for (i, (idx, val)) in top5.iter().enumerate() {
        println!("  {}. index={} score={:.4}", i + 1, idx, val);
    }

    let first_idx = top5.first().map(|(i, _)| *i).unwrap_or(0);
    let decoded = engine.tokenizer.decode(&[first_idx as u32]);

    println!("\nFirst token index: {}", first_idx);
    println!("Decoded: {:?}", decoded);

    // Check if logits are non-zero
    let non_zero: usize = logits.iter().filter(|&&v| v.abs() > 1e-6).count();
    if non_zero > 0 {
        println!("\nSTATUS: REAL_WEIGHTS_ACTIVE ({} non-zero logits)", non_zero);
    } else {
        println!("\nSTATUS: STUB_WEIGHTS (all logits zero)");
    }
}