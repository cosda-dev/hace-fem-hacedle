// LoRA Overlay Loader - Load LoRA adapters and apply overlay at runtime

use alloc::string::String;
use alloc::vec::Vec;

/// LoRA Adapter Info from GGUF
pub struct LoraAdapter {
    pub id: String,
    pub base_model: String,
    pub rank: u32,
    pub alpha: f32,
    pub scale: f32,
    pub target_layers: Vec<String>,
}

impl Default for LoraAdapter {
    fn default() -> Self {
        Self {
            id: String::from("default"),
            base_model: String::from("unknown"),
            rank: 8,
            alpha: 1.0,
            scale: 1.0,
            target_layers: Vec::new(),
        }
    }
}

/// Overlay Tensor - combines base weight with LoRA delta
pub struct OverlayTensor {
    pub base_id: String,
    pub lora_id: Option<String>,
    pub scale: f32,
}