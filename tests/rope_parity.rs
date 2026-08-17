// RoPE Parity Test - Compare against reference implementation
// Run: cargo test --test rope_parity --features std -- --nocapture

fn rope_reference_old(x: &mut [f32], pos: usize, dim: usize) {
    let base = 10000.0_f32;
    for i in 0..dim / 2 {
        let freq = base.powf(2.0 * (i as f32) / (dim as f32));
        let inv_freq = 1.0 / freq;
        let angle = pos as f32 * inv_freq;
        let cos_val = angle.cos();
        let sin_val = angle.sin();
        
        let x1 = x[i];
        let x2 = x[i + dim / 2];
        
        x[i] = x1 * cos_val - x2 * sin_val;
        x[i + dim / 2] = x1 * sin_val + x2 * cos_val;
    }
}

fn rope_reference_pairwise(x: &mut [f32], pos: usize, dim: usize) {
    let base = 10000.0_f32;
    let half = dim / 2;
    
    for i in 0..half {
        let freq = base.powf(2.0 * (i as f32) / (half as f32));
        let inv_freq = 1.0 / freq;
        let angle = pos as f32 * inv_freq;
        let cos_val = angle.cos();
        let sin_val = angle.sin();
        
        for j in 0..(x.len() / dim) {
            let idx = j * dim + i;
            let idx2 = j * dim + (i + half);
            
            if idx2 < x.len() {
                let x1 = x[idx];
                let x2 = x[idx2];
                
                x[idx] = x1 * cos_val - x2 * sin_val;
                x[idx2] = x1 * sin_val + x2 * cos_val;
            }
        }
    }
}

fn load_gguf_metadata(path: &str) -> Option<(f64, f64, f64)> {
    use std::fs::File;
    use std::io::Read;
    
    let mut file = File::open(path).ok()?;
    
    let mut header = [0u8; 4];
    file.read_exact(&mut header).ok()?;
    if &header != b"GGUF" { return None; }
    
    let mut kv_count = [0u8; 8];
    file.read_exact(&mut kv_count).ok()?;
    let kv_count = u64::from_le_bytes(kv_count) as usize;
    
    let mut rope_theta: f64 = 10000.0;
    let mut rope_scaling: f64 = 1.0;
    
    for _ in 0..kv_count {
        let mut key_len = [0u8; 8];
        file.read_exact(&mut key_len).ok()?;
        let key_len = u64::from_le_bytes(key_len) as usize;
        
        let mut key = vec![0u8; key_len];
        file.read_exact(&mut key).ok()?;
        let key_str = String::from_utf8_lossy(&key);
        
        let mut type_val = [0u8; 1];
        file.read_exact(&mut type_val).ok()?;
        
        match type_val[0] {
            1 => {
                let mut padding = [0u8; 7];
                file.read_exact(&mut padding).ok()?;
                let mut f64_val = [0u8; 8];
                file.read_exact(&mut f64_val).ok()?;
                let val = f64::from_le_bytes(f64_val);
                
                if key_str.contains("rope_theta") {
                    rope_theta = val;
                } else if key_str.contains("rope_scaling") {
                    rope_scaling = val;
                }
            }
            _ => {}
        }
    }
    
    Some((rope_theta, rope_scaling, 10000.0_f64))
}

#[test]
fn test_rope_basic() {
    use hace_fem_hacedle::quant_view::NativeBackend;
    
    let dim = 128;
    let mut our_output = vec![1.0f32; dim];
    let mut ref_output = vec![1.0f32; dim];
    
    let backend = NativeBackend::new();
    backend.rope(&mut our_output, 0, dim);
    
    rope_reference_old(&mut ref_output, 0, dim);
    
    for i in 0..dim {
        let diff = (our_output[i] - ref_output[i]).abs();
        assert!(diff < 1e-5, "RoPE mismatch at {}: our={}, ref={}", i, our_output[i], ref_output[i]);
    }
}

#[test]
fn test_rope_with_position() {
    use hace_fem_hacedle::quant_view::NativeBackend;
    
    let dim = 128;
    
    for pos in [0, 1, 100, 1000] {
        let mut our_output = vec![1.0f32; dim];
        let mut ref_output = vec![1.0f32; dim];
        
        let backend = NativeBackend::new();
        backend.rope(&mut our_output, pos, dim);
        
        rope_reference_old(&mut ref_output, pos, dim);
        
        for i in 0..dim {
            let diff = (our_output[i] - ref_output[i]).abs();
            if diff > 1e-4 {
                println!("RoPE diff at pos={}, idx={}: our={}, ref={}, diff={}", 
                         pos, i, our_output[i], ref_output[i], diff);
            }
        }
    }
}

#[test]
fn test_rope_metadata_extraction() {
    let model_path = "D:/host/llama-models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf";
    
    if let Some((theta, scaling, _)) = load_gguf_metadata(model_path) {
        println!("Rope theta: {}", theta);
        println!("Rope scaling: {}", scaling);
        assert!(theta > 0.0, "Rope theta should be positive");
    } else {
        println!("Model not found, skipping metadata test");
    }
}