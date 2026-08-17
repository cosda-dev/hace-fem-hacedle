use alloc::vec::Vec;
use crate::x::loader::dequant;

/// Embedding engine for GGUF runtime
/// Loads embedding matrix from token embeddings tensor (Q4_K_M dequantized)
pub struct EmbedEngine {
    pub embedding_dim: usize,
    pub vocab_size: usize,
    pub embeddings: Vec<f32>,
}

impl EmbedEngine {
    pub fn new() -> Self {
        Self {
            embedding_dim: 0,
            vocab_size: 0,
            embeddings: Vec::new(),
        }
    }

    /// Load embedding tensor from raw bytes (Q4_K_M format)
    /// Expects quant data from GGUF tensor
    pub fn load_quantized(&mut self, quant_data: &[u8], emb_dim: usize, vocab_size: usize) {
        let total_elements = vocab_size * emb_dim;
        self.embeddings.resize(total_elements, 0.0);
        
        #[cfg(feature = "std")]
        {
            // Dequantize Q4_K_M to f32
            crate::x::loader::dequant::dequant_q4_k(quant_data, &mut self.embeddings);
        }
        
        self.embedding_dim = emb_dim;
        self.vocab_size = vocab_size;
    }

    /// Load embedding tensor from GGUF model (pre-dequantized)
    pub fn load_from_gguf(&mut self, tensor_data: &[f32], emb_dim: usize, vocab_size: usize) {
        self.embeddings = tensor_data.to_vec();
        self.embedding_dim = emb_dim;
        self.vocab_size = vocab_size;
    }

    /// Get embedding vector for a single token
    pub fn embed_token(&self, token_id: u32) -> Vec<f32> {
        if self.embeddings.is_empty() {
            return vec![0.0; self.embedding_dim.max(1)];
        }
        if token_id as usize >= self.vocab_size {
            return vec![0.0; self.embedding_dim];
        }
        let start = token_id as usize * self.embedding_dim;
        let end = (start + self.embedding_dim).min(self.embeddings.len());
        self.embeddings[start..end].to_vec()
    }

    /// Embed a sequence of tokens
    pub fn embed_sequence(&self, tokens: &[u32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(tokens.len() * self.embedding_dim.max(1));
        for token in tokens {
            result.extend(self.embed_token(*token));
        }
        result
    }
}

impl Default for EmbedEngine {
    fn default() -> Self {
        Self::new()
    }
}