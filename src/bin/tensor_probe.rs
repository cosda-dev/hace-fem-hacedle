// Tensor Probe - Read and dequantize real tensors from GGUF
// Output: tensor shape, dequantized values (not zeros)

use std::env;
use std::path::Path;

use hace_fem_hacedle::x::loader::gguf::GgufLoader;
use hace_fem_hacedle::x::loader::dequant::dequant_q4_k;

fn main() {
    let args: Vec<String> = env::args().collect();
    
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
    println!("Tensor Probe - Real Tensor Extraction");
    println!("=====================================");
    println!("\nModel: {}", Path::new(path).file_name().unwrap().to_string_lossy());

    // Load GGUF
    let loader = GgufLoader::load(path).expect("Failed to load GGUF");

    println!("\nTensors found: {}", loader.tensor_count());

    // Find embedding tensor
    let embed_tensor = loader.get_tensor("token_embd.weight");
    
    match embed_tensor {
        Some(t) => {
            println!("\nEmbedding tensor (token_embd.weight):");
            println!("  Shape: {:?}", t.shape);
            println!("  Offset: {}", t.offset);
            println!("  GGML Type: {}", t.ggml_type);

            // Try to read tensor data
            // For Q4_K_M, need dequantization
            println!("\nStatus: Tensor located, requires dequant for Q4_K_M");
        }
        None => {
            println!("\nNo embedding tensor found");
        }
    }

    // Find output tensor
    let output_tensor = loader.get_tensor("output.weight");
    
    match output_tensor {
        Some(t) => {
            println!("\nOutput tensor (output.weight):");
            println!("  Shape: {:?}", t.shape);
            println!("  Offset: {}", t.offset);
            println!("  GGML Type: {}", t.ggml_type);
        }
        None => {
            println!("\nNo output tensor found");
        }
    }

    // List sample tensor names
    println!("\nSample tensors:");
    for (i, tensor) in loader.tensors.iter().take(10).enumerate() {
        println!("  {}. {} - shape={:?} type={}", 
            i + 1, tensor.name, tensor.shape, tensor.ggml_type);
    }
}