use alloc::vec::Vec;

use crate::ops::rope::apply_rope_pairwise;

/// Single Transformer layer with attention and FFN
pub struct TransformerLayer {
    pub layer_idx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    
    // Layer weights (loaded from GGUF)
    pub attn_q_weight: Vec<f32>,
    pub attn_k_weight: Vec<f32>,
    pub attn_v_weight: Vec<f32>,
    pub attn_o_weight: Vec<f32>,
    pub ffn_gate_weight: Vec<f32>,
    pub ffn_up_weight: Vec<f32>,
    pub ffn_down_weight: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
}

impl TransformerLayer {
    pub fn new(layer_idx: usize, n_embd: usize, n_head: usize, n_kv_head: usize) -> Self {
        Self {
            layer_idx,
            n_embd,
            n_head,
            n_kv_head,
            head_dim: n_embd / n_head,
            attn_q_weight: Vec::new(),
            attn_k_weight: Vec::new(),
            attn_v_weight: Vec::new(),
            attn_o_weight: Vec::new(),
            ffn_gate_weight: Vec::new(),
            ffn_up_weight: Vec::new(),
            ffn_down_weight: Vec::new(),
            attn_norm_weight: Vec::new(),
            ffn_norm_weight: Vec::new(),
        }
    }
    
    /// Load weights from GGUF tensor data
    pub fn load_weights(&mut self, loader: &crate::x::loader::gguf::GgufLoader) {
        let prefix = format!("blk.{}.", self.layer_idx);
        
        if let Some(tensor) = loader.get_tensor(&format!("{}attn_q.weight", prefix)) {
            if let Some(bytes) = loader.tensor_bytes(&format!("{}attn_q.weight", prefix)) {
                crate::x::loader::dequant::dequant_q4_k(bytes, &mut self.attn_q_weight);
            }
        }
        if let Some(tensor) = loader.get_tensor(&format!("{}attn_k.weight", prefix)) {
            if let Some(bytes) = loader.tensor_bytes(&format!("{}attn_k.weight", prefix)) {
                crate::x::loader::dequant::dequant_q4_k(bytes, &mut self.attn_k_weight);
            }
        }
        if let Some(tensor) = loader.get_tensor(&format!("{}attn_v.weight", prefix)) {
            if let Some(bytes) = loader.tensor_bytes(&format!("{}attn_v.weight", prefix)) {
                crate::x::loader::dequant::dequant_q4_k(bytes, &mut self.attn_v_weight);
            }
        }
        if let Some(tensor) = loader.get_tensor(&format!("{}attn_o.weight", prefix)) {
            if let Some(bytes) = loader.tensor_bytes(&format!("{}attn_o.weight", prefix)) {
                crate::x::loader::dequant::dequant_q4_k(bytes, &mut self.attn_o_weight);
            }
        }
        if let Some(tensor) = loader.get_tensor(&format!("{}ffn_gate.weight", prefix)) {
            if let Some(bytes) = loader.tensor_bytes(&format!("{}ffn_gate.weight", prefix)) {
                crate::x::loader::dequant::dequant_q4_k(bytes, &mut self.ffn_gate_weight);
            }
        }
        if let Some(tensor) = loader.get_tensor(&format!("{}ffn_up.weight", prefix)) {
            if let Some(bytes) = loader.tensor_bytes(&format!("{}ffn_up.weight", prefix)) {
                crate::x::loader::dequant::dequant_q4_k(bytes, &mut self.ffn_up_weight);
            }
        }
        if let Some(tensor) = loader.get_tensor(&format!("{}ffn_down.weight", prefix)) {
            if let Some(bytes) = loader.tensor_bytes(&format!("{}ffn_down.weight", prefix)) {
                crate::x::loader::dequant::dequant_q4_k(bytes, &mut self.ffn_down_weight);
            }
        }
        if let Some(tensor) = loader.get_tensor(&format!("{}attn_norm.weight", prefix)) {
            if let Some(bytes) = loader.tensor_bytes(&format!("{}attn_norm.weight", prefix)) {
                crate::x::loader::dequant::dequant_q4_k(bytes, &mut self.attn_norm_weight);
            }
        }
        if let Some(tensor) = loader.get_tensor(&format!("{}ffn_norm.weight", prefix)) {
            if let Some(bytes) = loader.tensor_bytes(&format!("{}ffn_norm.weight", prefix)) {
                crate::x::loader::dequant::dequant_q4_k(bytes, &mut self.ffn_norm_weight);
            }
        }
    }
    
    /// RMS Norm
    pub fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        if weight.is_empty() {
            return x.to_vec();
        }
        let len = x.len().min(weight.len());
        let ss: f32 = x[..len].iter().map(|&v| v * v).sum();
        let rms = (ss / len as f32 + eps).sqrt().recip();
        x[..len].iter().zip(weight[..len].iter()).map(|(&xi, &wi)| xi * wi * rms).collect()
    }
    
    /// Matrix multiplication: y = x @ w.T
    fn matmul(&self, x: &[f32], w: &[f32], out_dims: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; out_dims];
        let in_dims = x.len();
        for i in 0..out_dims {
            let mut sum = 0.0f32;
            for j in 0..in_dims {
                sum += x[j] * w[i * in_dims + j];
            }
            y[i] = sum;
        }
        y
    }
    
    /// Forward pass through layer
    pub fn forward(&self, hidden_states: &[f32]) -> Vec<f32> {
        if self.attn_q_weight.is_empty() {
            return hidden_states.to_vec();
        }
        
        let mut residual = hidden_states.to_vec();
        
        // Attn norm
        let h = self.rms_norm(hidden_states, &self.attn_norm_weight, 1e-6);
        
        // QKV projections
        let q = self.matmul(&h, &self.attn_q_weight, self.n_embd);
        let k = self.matmul(&h, &self.attn_k_weight, self.n_embd);
        let v = self.matmul(&h, &self.attn_v_weight, self.n_embd);
        
        // Simple attention (no causal mask for now, just take last token)
        let qlast = &q[h.len() - self.n_embd..];
        let vlast = &v[h.len() - self.n_embd..];
        
        // Output projection
        let attn_out = self.matmul(qlast, &self.attn_o_weight, self.n_embd);
        
        // Add residual
        let mut h = attn_out.iter().zip(residual.iter()).map(|(&a, &r)| a + r).collect::<Vec<_>>();
        
        // FFN norm
        h = self.rms_norm(&h, &self.ffn_norm_weight, 1e-6);
        
        // FFN: gate and up projections, then down
        let gate = self.matmul(&h, &self.ffn_gate_weight, self.n_embd);
        let up = self.matmul(&h, &self.ffn_up_weight, self.n_embd);
        
        // SiLU activation: x * sigmoid(x)
        let gate_act: Vec<f32> = gate.iter().map(|&g| g * (1.0f32 / (1.0f32 + (-g).exp())).min(1.0)).collect();
        
        // Multiply gate * up
        let fused: Vec<f32> = gate_act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();
        
        // Down projection
        let ffn_out = self.matmul(&fused, &self.ffn_down_weight, self.n_embd);
        
        // Add residual
        h = ffn_out.iter().zip(residual.iter()).map(|(&a, &r)| a + r).collect();
        
        h
    }
}

/// 24-layer Transformer model runtime
pub struct Transformer24 {
    pub layers: Vec<TransformerLayer>,
    pub n_embd: usize,
    pub vocab_size: usize,
}

impl Transformer24 {
    pub const NUM_LAYERS: usize = 24;

    pub fn new(n_embd: usize, n_head: usize, n_kv_head: usize, vocab_size: usize) -> Self {
        let layers = (0..Self::NUM_LAYERS)
            .map(|i| TransformerLayer::new(i, n_embd, n_head, n_kv_head))
            .collect();
        Self { layers, n_embd, vocab_size }
    }
    
    /// Load all layer weights from GGUF
    pub fn load_weights(&mut self, loader: &crate::x::loader::gguf::GgufLoader) {
        for layer in &mut self.layers {
            layer.load_weights(loader);
        }
    }

    /// Full forward pass through all 24 layers
    pub fn forward(&self, embeddings: &[f32]) -> Vec<f32> {
        let mut hidden = embeddings.to_vec();
        for layer in &self.layers {
            hidden = layer.forward(&hidden);
        }
        // Final layer norm would go here
        hidden
    }
}

impl Default for Transformer24 {
    fn default() -> Self {
        Self::new(4096, 32, 8, 32000)
    }
}