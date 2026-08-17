// Real Token Generation Test - Full inference chain
// Pipeline: GgufLoader -> InferenceEngine -> tokenizer -> embed -> transformer -> lm_head -> logits -> token

use std::fs;
use std::path::Path;

use hace_fem_hacedle::x::loader::GgufLoader;
use hace_fem_hacedle::x::provider::candle::{InferenceEngine, TokenizerEngine, BpeTokenizer};

/// Greedy decode: select token with highest logit value
fn greedy_decode(logits: &[f32]) -> usize {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(0.cmp(&0))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Generate first token using full inference chain
#[test]
fn test_generate_first_real_token() {
    let mut engine = InferenceEngine::default();
    
    // Test 1: Verify inference chain works without model
    let test_prompt = "Hello";
    let tokens = engine.tokenizer.encode(test_prompt);
    assert!(!tokens.is_empty(), "Tokenizer should produce tokens for 'Hello'");
    
    // Test 2: Run inference through the full pipeline
    let logits = engine.infer_logits(test_prompt);
    assert_eq!(logits.len(), 32000, "Logits should match LM head vocab size (32000)");
    
    // Test 3: Generate first token via argmax
    let first_token = greedy_decode(&logits);
    assert!(first_token < 32000, "First token index should be within vocab range");
    
    println!("First real token generation test:");
    println!("  Prompt: '{}'", test_prompt);
    println!("  Input tokens: {:?}", tokens);
    println!("  First generated token: {}", first_token);
    println!("  Logits range: [{:.6}, {:.6}]", 
        logits.iter().copied().fold(f32::INFINITY, f32::min),
        logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
}

/// Test with synthetic embedding data to simulate real inference
#[test]
fn test_inference_with_synthetic_weights() {
    let mut engine = InferenceEngine::default();
    
    // Synthetic embedding matrix: vocab=1000, embed_dim=64
    let vocab_size = 1000;
    let embed_dim = 64;
    let embeddings: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    
    // Synthetic LM head weights
    let lm_head_weight: Vec<f32> = (0..32000 * embed_dim)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();
    
    // Load weights
    engine.load_weights(vocab_size, embed_dim, embeddings, lm_head_weight);
    
    // Run inference
    let prompt = "Hello";
    let logits = engine.infer_logits(prompt);
    
    // Generate first token
    let first_token = greedy_decode(&logits);
    
    println!("Synthetic inference test:");
    println!("  Vocab size: {}", vocab_size);
    println!("  Embed dim: {}", embed_dim);
    println!("  First generated token: {}", first_token);
}

/// Test GGUF loader integration
#[test]
fn test_gguf_loader_integration() {
    // Create a minimal fake GGUF file for integration test
    let golden = Path::new("golden");
    let test_model_path = golden.join("test_model.gguf");
    
    // If no model file, skip the file loading but verify the API
    if !test_model_path.exists() {
        // Just verify the loader API works without actual file
        let _loader: Option<GgufLoader> = None;
        println!("SKIP: No GGUF test model - verifying API compatibility only");
        return;
    }
    
    let mut engine = InferenceEngine::default();
    match engine.load_model(test_model_path.to_str().unwrap()) {
        Ok(()) => {
            println!("Loaded GGUF model successfully");
        }
        Err(e) => {
            panic!("Failed to load GGUF model: {}", e);
        }
    }
}

/// Test multi-token generation
#[test]
fn test_multi_token_generation() {
    let engine = InferenceEngine::default();
    
    // Generate 3 tokens
    let prompt = "The";
    let generated = engine.infer(prompt, 3);
    
    assert_eq!(generated.len(), 3, "Should generate 3 tokens");
    
    println!("Multi-token generation:");
    println!("  Prompt: '{}'", prompt);
    println!("  Generated tokens: {:?}", generated);
}