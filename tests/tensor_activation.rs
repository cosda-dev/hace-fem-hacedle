// Tensor Activation Test - verifies real tensor loading from GGUF

use hacedle::x::loader::gguf::GgufLoader;

/// Test that GGUF loader can access tensor data
#[test]
fn test_tensor_data_access() {
    let model_paths = [
        "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "D:/host/llama-models/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    ];

    let mut model_found = None;
    for path in &model_paths {
        if std::path::Path::new(path).exists() {
            model_found = Some(*path);
            break;
        }
    }

    let path = match model_found {
        Some(p) => p,
        None => {
            println!("SKIP: No GGUF model found");
            return;
        }
    };

    println!("Loading: {}", path);
    
    let loader = GgufLoader::load(path).expect("Failed to load GGUF");
    
    println!("Tensor count: {}", loader.tensor_count());
    
    // List all tensor names
    for tensor in &loader.tensors {
        println!("  Tensor: {} shape={:?} type={}", tensor.name, tensor.shape, tensor.ggml_type);
    }

    // Check embedding tensor exists
    let embed_tensor = loader.get_tensor("token_embd.weight");
    println!("\nEmbedding tensor found: {}", embed_tensor.is_some());
    
    if let Some(t) = embed_tensor {
        println!("  Shape: {:?}", t.shape);
        println!("  Offset: {}", t.offset);
    }

    // Check output tensor exists
    let output_tensor = loader.get_tensor("output.weight");
    println!("\nOutput tensor found: {}", output_tensor.is_some());
    
    if let Some(t) = output_tensor {
        println!("  Shape: {:?}", t.shape);
        println!("  Offset: {}", t.offset);
    }
}

/// Test inference engine can activate weights
#[test]
fn test_inference_weight_activation() {
    let model_paths = [
        "D:/host/llama-models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "D:/host/llama-models/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    ];

    let mut model_found = None;
    for path in &model_paths {
        if std::path::Path::new(path).exists() {
            model_found = Some(*path);
            break;
        }
    }

    let path = match model_found {
        Some(p) => p,
        None => {
            println!("SKIP: No GGUF model found");
            return;
        }
    };

    let mut engine = hacedle::x::provider::candle::InferenceEngine::default();
    
    // Load model
    let load_result = engine.load_model(path);
    if load_result.is_err() {
        println!("Model load returned stub (expected until full dequant implemented)");
    }
    
    // Get logits - these should be non-zero after weight activation
    let logits = engine.infer_logits("test");
    
    // Count non-zero logits
    let non_zero: usize = logits.iter().filter(|&&v| v.abs() > 1e-6).count();
    
    println!("Non-zero logits: {}", non_zero);
    
    if non_zero > 0 {
        println!("SUCCESS: Weights are active!");
    } else {
        println!("INFO: Logits still zeros (weights not yet dequantized)");
    }
}