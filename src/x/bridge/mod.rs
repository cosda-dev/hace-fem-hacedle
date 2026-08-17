use alloc::string::String;
use alloc::vec::Vec;

pub mod candle;
pub mod llama;
pub mod onnx;
pub mod mlx;
pub mod vllm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Candle,
    Llama,
    Onnx,
    Mlx,
    Vllm,
}

pub trait Provider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> &[&'static str];
    fn is_available(&self) -> bool;
    fn load_priority(&self) -> u32;
}

pub struct ProviderManager {
    providers: Vec<&'static dyn Provider>,
}

impl ProviderManager {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    pub fn register(&mut self, provider: &'static dyn Provider) {
        self.providers.push(provider);
    }

    pub fn best_provider(&self, required_caps: &[&'static str]) -> Option<&'static dyn Provider> {
        self.providers
            .iter()
            .filter(|p| p.is_available())
            .filter(|p| required_caps.iter().all(|c| p.capabilities().contains(c)))
            .max_by_key(|p| p.load_priority())
            .map(|v| *v)
    }
}

impl Default for ProviderManager {
    fn default() -> Self {
        Self::new()
    }
}
