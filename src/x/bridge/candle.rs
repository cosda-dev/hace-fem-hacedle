use super::{Provider, ProviderKind};

pub struct CandleProvider;

impl CandleProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CandleProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for CandleProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Candle
    }

    fn name(&self) -> &'static str {
        "candle-native"
    }

    fn capabilities(&self) -> &[&'static str] {
        &["text_generation", "streaming", "embedding"]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn load_priority(&self) -> u32 {
        100
    }
}
