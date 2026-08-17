use super::{Provider, ProviderKind};

pub struct LlamaProvider;

impl LlamaProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LlamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for LlamaProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Llama
    }

    fn name(&self) -> &'static str {
        "llama-cpp"
    }

    fn capabilities(&self) -> &[&'static str] {
        &["text_generation", "streaming", "grammar"]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load_priority(&self) -> u32 {
        80
    }
}
