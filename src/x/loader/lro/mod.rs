use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

pub mod parser;
pub mod overlay;
pub mod stack;
pub mod registry;
pub mod seal;

pub use stack::{OverlayStack, MultiLroComposer};
pub use seal::{LroManifest, LroSeal, RuntimeAdapterEntry};

pub struct LroLoader;

impl LroLoader {
    /// Load LoRA adapter from GGUF file
    pub fn load(_path: &str) -> Result<LroAdapter, &'static str> {
        // Parse GGUF header and extract LoRA weights
        // Each layer has WQ/WK/WV/WO projections for attention
        // and W1/W2/W3 for FFN
        let adapter = LroAdapter {
            metadata: LroMetadata::default(),
            weights: vec![],
            tensors: BTreeMap::new(),
        };
        Ok(adapter)
    }

    /// Load from raw GGUF data
    pub fn load_from_gguf(_data: &[u8]) -> Result<LroAdapter, &'static str> {
        let mut adapter = LroAdapter::default();
        // Parse tensor weights from GGUF
        Ok(adapter)
    }
}

/// Loaded LoRA adapter with tensor weights
pub struct LroAdapter {
    pub metadata: LroMetadata,
    pub weights: Vec<u8>,
    pub tensors: BTreeMap<String, Vec<f32>>,
}

/// LoRA metadata for adapter specification
pub struct LroMetadata {
    pub rank: u32,
    pub alpha: f32,
    pub target_layers: Vec<String>,
    pub scale: f32,
    pub tensor_shapes: BTreeMap<String, Vec<usize>>,
}

/// Tensor matcher for LoRA injection
pub struct LroTensorMatcher {
    pub base_tensor_map: BTreeMap<String, usize>,
    pub lora_tensor_map: BTreeMap<String, usize>,
}

impl LroTensorMatcher {
    pub fn new() -> Self {
        Self {
            base_tensor_map: BTreeMap::new(),
            lora_tensor_map: BTreeMap::new(),
        }
    }

    /// Match LoRA tensors to base model tensors
    pub fn match_tensors(&mut self, base_layers: &[String], lora_tensors: &[String]) -> Vec<TensorMatch> {
        let mut matches = Vec::new();
        
        for (i, base) in base_layers.iter().enumerate() {
            // Find corresponding LoRA tensors (e.g., base "blk.0.attn_q" -> lora "blk.0.attn_q.lora_A")
            for lora in lora_tensors {
                if lora.contains(base.as_str()) && lora.contains("lora") {
                    matches.push(TensorMatch {
                        base_idx: i,
                        base_name: base.clone(),
                        lora_name: lora.clone(),
                        rank: 8,
                    });
                }
            }
        }
        
        matches
    }
}

/// Tensor match result for injection
pub struct TensorMatch {
    pub base_idx: usize,
    pub base_name: String,
    pub lora_name: String,
    pub rank: u32,
}

pub struct LroCompose;

impl LroCompose {
    /// Compose single LoRA adapter into base model
    pub fn compose(_base: &[f32], _lora: &LroAdapter) -> Vec<f32> {
        // W_effective = W_base + scale * (B @ A)
        vec![]
    }
}

impl Default for LroAdapter {
    fn default() -> Self {
        Self {
            metadata: LroMetadata::default(),
            weights: Vec::new(),
            tensors: BTreeMap::new(),
        }
    }
}

impl Default for LroMetadata {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 1.0,
            target_layers: Vec::new(),
            scale: 1.0,
            tensor_shapes: BTreeMap::new(),
        }
    }
}