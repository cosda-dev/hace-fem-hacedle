// Logits Probe - Check if logits are real (not stub zeros)
// Pipeline: GGUF -> Encode -> Embed -> Forward -> Logits

use std::env;
use std::path::Path;

use hace_fem_hacedle::x::loader::gguf::GgufLoader;
use hace_fem_hacedle::x::provider::candle::{InferenceEngine, TokenizerEngine, LogitsProcessor};

fn main() {
    let args: Vec<String> = env::args().collect();
    let prompt = args.get(1).map(|s| s.as_str()).unwrap_or("hello world");

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
            std::process::exit(1);
        }
    };

    println!("=====================================");
    println!("Logits Probe - Real Logits Detection");
    println!("=====================================");
    println!("\nModel: {}", Path::new(path).file_name().unwrap().to_string_lossy());
    println!("Prompt: {}", prompt);

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
        let text = engine.tokenizer.decode(&[*idx as u32]);
        println!("  {}. idx={} val={:.4} text={:?}", i + 1, idx, val, text);
    }

    // Check if logits are differentiated (not all same/zero)
    let max_logit = top5.first().map(|(_, v)| *v).unwrap_or(0.0);
    let min_logit = top5.last().map(|(_, v)| *v).unwrap_or(0.0);
    let diff = max_logit - min_logit;

    println!("\nLogit spread: {:.4}", diff);

    // Load GGUF info
    let loader = GgufLoader::load(path);
    match loader {
        Ok(l) => {
            println!("\nGGUF tensor count: {}", l.tensor_count());
        }
        Err(_) => {
            println!("\nGGUF load failed or stub");
        }
    }

    if diff > 1e-3 {
        println!("\nSTATUS: NON_ZERO_LOGITS (forward pass active)");
    } else {
        println!("\nSTATUS: STUB_LOGITS (weights not loaded)");
    }

    // Test with different prompt to verify differentiation
    let logits2 = engine.infer_logits("different prompt");
    let mut top5_2: Vec<(usize, f32)> = logits2.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    top5_2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top5_2.truncate(5);

    // Compare patterns
    let same_pattern = top5.iter().zip(top5_2.iter())
        .take(3)
        .all(|(&(i1, _), &(i2, _))| i1 == i2);

    if same_pattern && diff <= 1e-3 {
        println!("\nCRITICAL: Both prompts give same top tokens with zero logits - stub detected");
    }
}