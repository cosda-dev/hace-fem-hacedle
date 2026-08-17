use crate::alloc_exports::*;
use crate::quant_view::{QuantTensorView, NativeBackend, QuantType};

pub struct TransformerBlock {
    pub hidden_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub n_layers: usize,
}

impl TransformerBlock {
    pub fn new(hidden_size: usize, n_heads: usize, head_dim: usize, n_layers: usize) -> Self {
        Self {
            hidden_size,
            n_heads,
            head_dim,
            n_layers,
        }
    }
}

pub struct BrainBlockWeights {
    pub rms_attn_weight: Vec<f32>,
    pub rms_ffn_weight: Vec<f32>,
    
    pub q_proj_weight: Vec<u8>,
    pub q_proj_shape: Vec<usize>,
    pub k_proj_weight: Vec<u8>,
    pub k_proj_shape: Vec<usize>,
    pub v_proj_weight: Vec<u8>,
    pub v_proj_shape: Vec<usize>,
    pub o_proj_weight: Vec<u8>,
    pub o_proj_shape: Vec<usize>,
    
    pub gate_proj_weight: Vec<u8>,
    pub gate_proj_shape: Vec<usize>,
    pub up_proj_weight: Vec<u8>,
    pub up_proj_shape: Vec<usize>,
    pub down_proj_weight: Vec<u8>,
    pub down_proj_shape: Vec<usize>,
}

pub struct BrainBlock {
    pub weights: BrainBlockWeights,
    pub n_heads: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
}

impl BrainBlock {
    pub fn forward(
        &self,
        hidden: &[f32],
        kv_cache: &mut KvCache,
        pos: usize,
    ) -> Vec<f32> {
        let backend = NativeBackend::new();
        let hidden_size = hidden.len();
        let n_heads = self.n_heads;
        let n_heads_kv = self.n_heads_kv;
        let head_dim = self.head_dim;
        let repeat_kv = n_heads / n_heads_kv;
        
        let mut output = hidden.to_vec();
        
        // 1. Attention RMSNorm
        let mut attn_norm = vec![0.0f32; hidden_size];
        backend.rmsnorm(hidden, &self.weights.rms_attn_weight, &mut attn_norm);
        
        // 2. QKV Projections
        let q_shape = self.weights.q_proj_shape.clone();
        let k_shape = self.weights.k_proj_shape.clone();
        let v_shape = self.weights.v_proj_shape.clone();
        
        let q_tensor = QuantTensorView::new(
            self.weights.q_proj_weight.clone(),
            q_shape,
            QuantType::Q5_0,
        );
        let k_tensor = QuantTensorView::new(
            self.weights.k_proj_weight.clone(),
            k_shape,
            QuantType::Q5_0,
        );
        let v_tensor = QuantTensorView::new(
            self.weights.v_proj_weight.clone(),
            v_shape,
            QuantType::Q8_0,
        );
        
        let q_proj = q_tensor.dequantize();
        let k_proj = k_tensor.dequantize();
        let v_proj = v_tensor.dequantize();
        
        // Q: n_heads * head_dim
        // K, V: n_heads_kv * head_dim
        let mut q = vec![0.0f32; n_heads * head_dim];
        let mut k = vec![0.0f32; n_heads_kv * head_dim];
        let mut v = vec![0.0f32; n_heads_kv * head_dim];
        
        let in_features = hidden_size;
        
        // Q projection: [hidden_size] -> [n_heads * head_dim]
        for i in 0..n_heads * head_dim {
            for j in 0..in_features {
                if i * in_features + j < q_proj.len() {
                    q[i] += q_proj[i * in_features + j] * attn_norm[j];
                }
            }
        }
        
        // K projection: [hidden_size] -> [n_heads_kv * head_dim] (GQA)
        for i in 0..n_heads_kv * head_dim {
            for j in 0..in_features {
                if i * in_features + j < k_proj.len() {
                    k[i] += k_proj[i * in_features + j] * attn_norm[j];
                }
            }
        }
        
        // V projection: [hidden_size] -> [n_heads_kv * head_dim] (GQA)
        for i in 0..n_heads_kv * head_dim {
            for j in 0..in_features {
                if i * in_features + j < v_proj.len() {
                    v[i] += v_proj[i * in_features + j] * attn_norm[j];
                }
            }
        }
        
        // 3. RoPE
        backend.rope(&mut q, pos, head_dim);
        backend.rope(&mut k, pos, head_dim);
        
        // 4. KV Cache Update
        kv_cache.append(&k, &v, pos);
        
        // 5. Attention Computation (simplified)
        let kv_len = kv_cache.current_len();
        let mut attn_scores = vec![0.0f32; n_heads * kv_len];
        
        for h in 0..n_heads {
            for i in 0..kv_len {
                let mut sum = 0.0f32;
                for d in 0..head_dim {
                    sum += q[h * head_dim + d] * kv_cache.get_cached_k(h, i, head_dim);
                }
                attn_scores[h * kv_len + i] = sum / (head_dim as f32).sqrt();
            }
            let scores = &mut attn_scores[h * kv_len..(h + 1) * kv_len];
            backend.softmax(scores);
        }
        
        // 6. Attention Output
        let o_tensor = QuantTensorView::new(
            self.weights.o_proj_weight.clone(),
            self.weights.o_proj_shape.clone(),
            QuantType::Q4K,
        );
        let o_proj = o_tensor.dequantize();
        
        let mut attn_out = vec![0.0f32; hidden_size];
        for h in 0..n_heads {
            for d in 0..head_dim {
                let idx = h * head_dim + d;
                if idx < attn_out.len() {
                    let mut sum = 0.0f32;
                    for i in 0..kv_len {
                        sum += attn_scores[h * kv_len + i] * kv_cache.get_cached_v(h, i, head_dim);
                    }
                    attn_out[idx] = sum;
                }
            }
        }
        
        // 7. Residual + FFN RMSNorm
        for i in 0..hidden_size {
            if i < attn_out.len() {
                attn_out[i] += hidden[i];
            }
        }
        
        let mut ffn_norm = vec![0.0f32; hidden_size];
        backend.rmsnorm(&attn_out, &self.weights.rms_ffn_weight, &mut ffn_norm);
        
        // 8. FFN (Gate, Up, SiLU, Down)
        let gate_tensor = QuantTensorView::new(
            self.weights.gate_proj_weight.clone(),
            self.weights.gate_proj_shape.clone(),
            QuantType::Q4K,
        );
        let up_tensor = QuantTensorView::new(
            self.weights.up_proj_weight.clone(),
            self.weights.up_proj_shape.clone(),
            QuantType::Q4K,
        );
        let down_tensor = QuantTensorView::new(
            self.weights.down_proj_weight.clone(),
            self.weights.down_proj_shape.clone(),
            QuantType::Q4K,
        );
        
        let gate_proj = gate_tensor.dequantize();
        let up_proj = up_tensor.dequantize();
        let down_proj = down_tensor.dequantize();
        
        let inter_size = if gate_proj.len() > hidden_size { gate_proj.len() / hidden_size } else { hidden_size };
        let mut gate_out = vec![0.0f32; inter_size];
        let mut up_out = vec![0.0f32; inter_size];
        
        for i in 0..inter_size {
            for j in 0..hidden_size {
                if i * hidden_size + j < gate_proj.len() {
                    gate_out[i] += gate_proj[i * hidden_size + j] * ffn_norm[j];
                    up_out[i] += up_proj[i * hidden_size + j] * ffn_norm[j];
                }
            }
        }
        
        let mut silu_out = vec![0.0f32; inter_size];
        for i in 0..inter_size {
            silu_out[i] = gate_out[i] * up_out[i] / (1.0 + (-up_out[i].abs()).exp());
        }
        
        let mut ffn_out = vec![0.0f32; hidden_size];
        for i in 0..hidden_size {
            for j in 0..inter_size {
                if i * inter_size + j < down_proj.len() {
                    ffn_out[i] += down_proj[i * inter_size + j] * silu_out[j];
                }
            }
        }
        
        for i in 0..hidden_size {
            output[i] = ffn_out[i] + attn_out[i];
        }
        
        output
    }
}

pub struct KvCache {
    pub max_seq_len: usize,
    pub n_heads: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub current_len: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize, n_heads: usize, n_heads_kv: usize, head_dim: usize) -> Self {
        Self {
            max_seq_len,
            n_heads,
            n_heads_kv,
            head_dim,
            k_cache: vec![0.0f32; max_seq_len * n_heads_kv * head_dim],
            v_cache: vec![0.0f32; max_seq_len * n_heads_kv * head_dim],
            current_len: 0,
        }
    }
    
    pub fn current_len(&self) -> usize {
        self.current_len
    }
    
    pub fn append(&mut self, k: &[f32], v: &[f32], pos: usize) {
        if pos >= self.max_seq_len {
            return;
        }
        let offset = pos * self.n_heads_kv * self.head_dim;
        let count = k.len().min(v.len()).min(self.n_heads_kv * self.head_dim);
        
        for i in 0..count {
            if offset + i < self.k_cache.len() {
                self.k_cache[offset + i] = k[i];
                self.v_cache[offset + i] = v[i];
            }
        }
        self.current_len = (pos + 1).max(self.current_len);
    }
    
    pub fn get_cached_k(&self, head: usize, token_idx: usize, head_dim: usize) -> f32 {
        // With GQA: repeat head index across KV heads
        let kv_head = head % self.n_heads_kv;
        let offset = token_idx * self.n_heads_kv * head_dim + kv_head * head_dim;
        if offset < self.k_cache.len() {
            self.k_cache[offset]
        } else {
            0.0
        }
    }
    
    pub fn get_cached_v(&self, head: usize, token_idx: usize, head_dim: usize) -> f32 {
        let kv_head = head % self.n_heads_kv;
        let offset = token_idx * self.n_heads_kv * head_dim + kv_head * head_dim;
        if offset < self.v_cache.len() {
            self.v_cache[offset]
        } else {
            0.0
        }
    }
}