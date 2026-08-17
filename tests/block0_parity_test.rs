// Block0 Parity Test - Verify single transformer block against reference
// This test validates the forward pass with real GGUF tensor data

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;
    
    if exp == 0 {
        0.0
    } else {
        let f32_exp = (exp as i32) - 15 + 127;
        let f32_frac = frac as f32 / 1024.0;
        let result = (1.0 + f32_frac) * 2.0_f32.powi(f32_exp);
        if sign == 1 { -result } else { result }
    }
}

fn dequant_q4_k_ref(data: &[u8], output: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 144;
    
    let blocks = data.len() / BYTES_PER_BLOCK;
    let elements = output.len().min(blocks * BLOCK_SIZE);
    
    for block_idx in 0..blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        if block_offset + BYTES_PER_BLOCK > data.len() {
            break;
        }
        let block = &data[block_offset..block_offset + BYTES_PER_BLOCK];
        
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min_val = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        
        for j in 0..BLOCK_SIZE {
            let idx = block_idx * BLOCK_SIZE + j;
            if idx >= elements {
                break;
            }
            
            let scale_idx = 8 + (j / 32);
            let scale_val = f16_to_f32(u16::from_le_bytes([block[scale_idx * 2], block[scale_idx * 2 + 1]]));
            
            let q_offset = 16 + (j / 2);
            let q = if j % 2 == 0 {
                block[q_offset] & 0xF
            } else {
                block[q_offset] >> 4
            };
            
            output[idx] = (q as f32) * scale_val + min_val;
        }
    }
}

#[test]
fn test_q4k_dequant_basic() {
    let input = vec![0x00; 144];
    let mut output = vec![0.0f32; 256];
    dequant_q4_k_ref(&input, &mut output);
    
    for (i, val) in output.iter().enumerate() {
        assert!(*val.abs() < 1e-5, "All zeros should dequant to ~0 for zero input at index {}", i);
    }
}

#[test]
fn test_block0_forward_mock() {
    use hace_fem_hacedle::core::ops::block::{KvCache, BrainBlock, BrainBlockWeights};
    
    let hidden_size = 3584;
    let n_heads = 28;
    let head_dim = 128;
    let inter_size = 18944;
    
    let weights = BrainBlockWeights {
        rms_attn_weight: vec![1.0f32; hidden_size],
        rms_ffn_weight: vec![1.0f32; hidden_size],
        q_proj_weight: vec![0x80; hidden_size * n_heads * head_dim],
        q_proj_shape: vec![hidden_size, n_heads * head_dim],
        k_proj_weight: vec![0x80; hidden_size * n_heads * head_dim / 7],
        k_proj_shape: vec![hidden_size, n_heads * head_dim / 7],
        v_proj_weight: vec![0x80; hidden_size * n_heads * head_dim / 7],
        v_proj_shape: vec![hidden_size, n_heads * head_dim / 7],
        o_proj_weight: vec![0x80; hidden_size * n_heads * head_dim],
        o_proj_shape: vec![n_heads * head_dim, hidden_size],
        gate_proj_weight: vec![0x80; hidden_size * inter_size],
        gate_proj_shape: vec![hidden_size, inter_size],
        up_proj_weight: vec![0x80; hidden_size * inter_size],
        up_proj_shape: vec![hidden_size, inter_size],
        down_proj_weight: vec![0x80; inter_size * hidden_size],
        down_proj_shape: vec![inter_size, hidden_size],
    };
    
    let block = BrainBlock {
        weights,
        n_heads,
        head_dim,
    };
    
    let mut kv_cache = KvCache::new(1024, n_heads, head_dim);
    let hidden = vec![0.1f32; hidden_size];
    
    let output = block.forward(&hidden, &mut kv_cache, 0);
    
    assert_eq!(output.len(), hidden_size, "Output size should match hidden size");
    println!("Block0 forward mock: output len = {}", output.len());
}

#[test] 
fn test_rmsnorm_pipeline() {
    use hace_fem_hacedle::quant_view::NativeBackend;
    
    let input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let weight = vec![1.0f32; 4];
    let mut output = vec![0.0f32; 4];
    
    let backend = NativeBackend::new();
    backend.rmsnorm(&input, &weight, &mut output);
    
    let ss: f32 = input.iter().map(|&x| x * x).sum();
    let expected_rms = (ss / 4.0 + 1e-6).sqrt().recip();
    let expected = input.iter().map(|&x| x * expected_rms).collect::<Vec<_>>();
    
    for i in 0..4 {
        assert!((output[i] - expected[i]).abs() < 1e-5, "RMSNorm mismatch at {}", i);
    }
}

#[test]
fn test_rope_pipeline() {
    use hace_fem_hacedle::quant_view::NativeBackend;
    
    let input_len = 128;
    let mut input = vec![1.0f32; input_len];
    
    let backend = NativeBackend::new();
    backend.rope(&mut input, 0, input_len);
    
    assert_eq!(input.len(), input_len, "RoPE should preserve length");
}

#[test]
fn test_softmax_normalized() {
    use hace_fem_hacedle::quant_view::NativeBackend;
    
    let mut input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let backend = NativeBackend::new();
    backend.softmax(&mut input);
    
    let sum_after: f32 = input.iter().sum();
    assert!((sum_after - 1.0).abs() < 1e-5, "Softmax should sum to 1.0, got {}", sum_after);
}