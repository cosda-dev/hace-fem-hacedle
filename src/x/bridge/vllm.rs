use super::{Provider, ProviderKind};

pub struct VllmProvider;

impl VllmProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for VllmProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for VllmProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Vllm
    }

    fn name(&self) -> &'static str {
        "vllm"
    }

    fn capabilities(&self) -> &[&'static str] {
        &["text_generation", "batch_inference", "serving"]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load_priority(&self) -> u32 {
        20
    }
}
