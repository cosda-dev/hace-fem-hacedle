use alloc::vec::Vec;

use super::lmhead::{LMHead, LogitsProcessor};
use super::embed::EmbedEngine;
use super::tokenizer::{BpeTokenizer, TokenizerEngine};
use super::layer::Transformer24;
use crate::x::loader::GgufLoader;

/// Inference engine with full GGUF runtime
pub struct InferenceEngine {
    pub context_size: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub embed: EmbedEngine,
    pub transformer: Transformer24,
    pub lm_head: LMHead,
    pub tokenizer: BpeTokenizer,
    pub logits_processor: LogitsProcessor,
    pub loader: Option<GgufLoader>,
}

impl InferenceEngine {
    pub fn new(context_size: u32, top_p: f32, temperature: f32) -> Self {
        Self {
            context_size,
            top_p,
            temperature,
            embed: EmbedEngine::new(),
            transformer: Transformer24::default(),
            lm_head: LMHead::new(32000, 4096),
            tokenizer: BpeTokenizer::new(),
            logits_processor: LogitsProcessor::new(temperature, top_p, 50),
            loader: None,
        }
    }

    /// Load model from GGUF file - must be called before inference
    pub fn load_model(&mut self, path: &str) -> Result<(), &'static str> {
        let mut loader = GgufLoader::load(path).map_err(|_| "gguf_load_failed")?;
        
        // Load embedding weights and model spec
        #[cfg(feature = "std")]
        {
            let mut vocab_size = 32000;
            let mut embed_dim = 4096;
            
            if let Some(embed_tensor) = loader.get_tensor("token_embd.weight") {
                let shape = &embed_tensor.shape;
                vocab_size = shape.get(0).copied().unwrap_or(0);
                embed_dim = shape.get(1).copied().unwrap_or(0);
                
                if let Some(tensor_bytes) = loader.tensor_bytes("token_embd.weight") {
                    // Allocate output buffer and dequant
                    let mut embeddings = vec![0.0f32; vocab_size * embed_dim];
                    crate::x::loader::dequant::dequant_q4_k(tensor_bytes, &mut embeddings);
                    self.embed.load_from_gguf(&embeddings, embed_dim, vocab_size);
                }
                
                self.transformer.n_embd = embed_dim;
                self.transformer.vocab_size = vocab_size;
            }
            
            // Load output weights
            if let Some(output_tensor) = loader.get_tensor("output.weight") {
                let shape = &output_tensor.shape;
                let out_vocab = shape.get(0).copied().unwrap_or(vocab_size);
                let out_dim = shape.get(1).copied().unwrap_or(embed_dim);
                
                if let Some(tensor_bytes) = loader.tensor_bytes("output.weight") {
                    let mut weight_data = vec![0.0f32; out_vocab * out_dim];
                    crate::x::loader::dequant::dequant_q4_k(tensor_bytes, &mut weight_data);
                    self.lm_head.load_weight(weight_data);
                    self.lm_head.vocab_size = out_vocab;
                    self.lm_head.embed_dim = out_dim;
                }
            }
            
            // Load transformer layer weights
            self.transformer.load_weights(&loader);
            
            // TODO: Load tokenizer vocab from GGUF metadata
        }
        
        self.loader = Some(loader);
        Ok(())
    }
    
    /// Load model weights from pre-loaded GGUF tensor data (for testing)
    pub fn load_weights(&mut self, vocab_size: usize, embed_dim: usize, embeddings: Vec<f32>, lm_head_weight: Vec<f32>) {
        self.embed.load_from_gguf(&embeddings, embed_dim, vocab_size);
        self.lm_head.load_weight(lm_head_weight);
        self.transformer.n_embd = embed_dim;
        self.transformer.vocab_size = vocab_size;
    }
    
    /// Initialize from GGUF file - convenience method that loads model and returns success
    pub fn from_gguf(path: &str) -> Result<Self, &'static str> {
        let mut engine = Self::default();
        engine.load_model(path)?;
        Ok(engine)
    }

    /// Run inference and return logits
    pub fn infer_logits(&self, prompt: &str) -> Vec<f32> {
        // Tokenize
        let tokens = self.tokenizer.encode(prompt);
        
        // Embed
        let embedded = self.embed.embed_sequence(&tokens);
        
        // Forward through transformer
        let hidden = self.transformer.forward(&embedded);
        
        // LM Head - output logits
        let logits = self.lm_head.forward(&hidden);
        
        logits
    }

    /// Run inference with sampling
    pub fn infer(&self, prompt: &str, max_tokens: u32) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(prompt);
        let mut generated = Vec::new();
        
        for _ in 0..max_tokens {
            let logits = self.infer_logits(&self.tokenizer.decode(&tokens));
            let processed = self.logits_processor.process(logits);
            
            // Simple argmax sampling
            let next_token = processed.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            
            tokens.push(next_token);
            generated.push(next_token);
        }
        
        generated
    }

    /// Get logits for next token (for external processing)
    pub fn get_logits(&self, prompt: &str) -> Vec<f32> {
        self.infer_logits(prompt)
    }
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self::new(8192, 0.9, 0.7)
    }
}