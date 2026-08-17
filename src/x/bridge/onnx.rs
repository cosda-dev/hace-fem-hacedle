use super::{Provider, ProviderKind};

pub struct OnnxProvider;

impl OnnxProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OnnxProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for OnnxProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Onnx
    }

    fn name(&self) -> &'static str {
        "onnx-runtime"
    }

    fn capabilities(&self) -> &[&'static str] {
        &["text_generation", "batch_inference"]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load_priority(&self) -> u32 {
        60
    }
}
