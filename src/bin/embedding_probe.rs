// Embedding Probe - Tokenize and embed real tokens from GGUF
// Pipeline: GGUF -> Tokenizer -> Embedding Matrix -> Embedding Vector

use std::env;
use std::path::Path;

use hace_fem_hacedle::x::loader::gguf::GgufLoader;
use hace_fem_hacedle::x::provider::candle::{InferenceEngine, TokenizerEngine, EmbedEngine};

fn main() {
    let args: Vec<String> = env::args().collect();
    let prompt = args.get(1).map(|s| s.as_str()).unwrap_or("hello");

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
    println!("Embedding Probe - Real Embedding Values");
    println!("=====================================");
    println!("\nModel: {}", Path::new(path).file_name().unwrap().to_string_lossy());
    println!("Prompt: {}", prompt);

    // Create inference engine
    let engine = InferenceEngine::default();

    // Tokenize prompt
    let tokens = engine.tokenizer.encode(prompt);
    println!("\nTokens: {:?}", tokens);

    // Get embeddings - currently stub (zeros)
    let embeddings = engine.embed.embed_sequence(&tokens);
    
    // Compute embedding norm to detect stub vs real
    let norm: f32 = embeddings.iter().map(|v| v * v).sum::<f32>().sqrt();
    
    println!("\nEmbedding vector length: {}", embeddings.len());
    println!("Embedding norm: {:.6}", norm);

    if norm > 1e-3 {
        println!("\nEmbedding preview (first 16 values):");
        for (i, &v) in embeddings.iter().take(16).enumerate() {
            println!("  [{}] {:.6}", i, v);
        }
        println!("\nSTATUS: REAL_EMBEDDING_ACTIVE");
    } else {
        println!("\nNOTE: Embeddings are zero (stub weights not yet loaded)");
        println!("STATUS: STUB_EMBEDDING");
    }

    // TODO: After dequant implemented:
    // let loader = GgufLoader::load(path)?;
    // let embed_tensor = loader.get_tensor("token_embd.weight")?;
    // let embed_data = loader.tensor_bytes("token_embd.weight")?;
    // dequant_q4_k(embed_data, &mut real_weights);
    // engine.embed.load_weights(real_weights);
}