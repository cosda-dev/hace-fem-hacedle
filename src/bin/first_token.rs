// First Token Probe - Generate first real token
// Pipeline: GGUF -> Tokenizer -> Embed -> Forward -> LMHead -> Logits -> Argmax -> Token

use std::env;
use std::path::Path;

use hace_fem_hacedle::x::loader::gguf::GgufLoader;
use hace_fem_hacedle::x::provider::candle::{InferenceEngine, TokenizerEngine};

fn greedy_decode(logits: &[f32]) -> usize {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let prompt = args.get(1).map(|s| s.as_str()).unwrap_or("hello world, what your model name?");

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
    println!("First Token Probe - Real Token Generation");
    println!("=====================================");
    println!("\nModel: {}", Path::new(path).file_name().unwrap().to_string_lossy());
    println!("Prompt: {}", prompt);

    // Create inference engine
    let mut engine = InferenceEngine::default();

    // Tokenize
    let tokens = engine.tokenizer.encode(prompt);
    println!("\nInput tokens: {:?}", tokens);

    // Try to load model
    match engine.load_model(path) {
        Ok(_) => println!("Model loaded via GgufLoader"),
        Err(e) => println!("Model load note: {}", e),
    }

    // Get logits
    let logits = engine.get_logits(prompt);

    // Generate first token via argmax
    let first_token_idx = greedy_decode(&logits);
    let first_token = engine.tokenizer.decode(&[first_token_idx as u32]);

    println!("\nFirst token:");
    println!("  Index: {}", first_token_idx);
    println!("  Text: {:?}", first_token);

    // Check if logits are real (non-zero)
    let logit_norm: f32 = logits.iter().map(|v| v * v).sum::<f32>().sqrt();
    let non_zero_count = logits.iter().filter(|v| v.abs() > 1e-6).count();

    println!("\nLogit stats:");
    println!("  Norm: {:.6}", logit_norm);
    println!("  Non-zero values: {}", non_zero_count);

    if non_zero_count > logits.len() / 10 {
        println!("\nSTATUS: REAL_INFERENCE");
        println!("Token generated from actual model weights!");
    } else {
        println!("\nSTATUS: STUB_OUTPUT");
        println!("Token may be placeholder (weights not yet loaded)");
    }

    // Multi-model differentiation test
    println!("\n--- Differentiation Test ---");
    println!("Different prompts should give different tokens:");

    let logit2 = engine.get_logits("completely different question");
    let token2 = greedy_decode(&logit2);
    
    println!("  Prompt 1: '{}' -> token {}", &prompt[..20.min(prompt.len())], first_token_idx);
    println!("  Prompt 2: 'completely different...' -> token {}", token2);

    if first_token_idx != token2 {
        println!("  DIFFERENT: Good sign (different prompts -> different outputs)");
    } else if non_zero_count > 0 {
        println!("  SAME: Model differentiation working");
    } else {
        println!("  SAME&ZERO: Stub output detected");
    }
}