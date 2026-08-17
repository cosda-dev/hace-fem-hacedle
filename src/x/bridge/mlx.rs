use super::{Provider, ProviderKind};

pub struct MlxProvider;

impl MlxProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MlxProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for MlxProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Mlx
    }

    fn name(&self) -> &'static str {
        "mlx"
    }

    fn capabilities(&self) -> &[&'static str] {
        &["text_generation", "multimodal"]
    }

    fn is_available(&self) -> bool {
        cfg!(target_os = "macos")
    }

    fn load_priority(&self) -> u32 {
        40
    }
}
