mod runtime_impl;

pub use runtime_impl::HacedleBrain;

#[cfg(feature = "std")]
use async_trait::async_trait;

#[cfg(feature = "std")]
#[async_trait]
pub trait BrainRuntime: Send + Sync {
    fn mount_projection(&self) -> Result<(), BrainError>;
    fn mount_rules(&self) -> Result<(), BrainError>;
    fn mount_models(&self) -> Result<(), BrainError>;
    fn select_provider(&self, caps: &[&'static str]) -> Option<&'static dyn super::super::x::bridge::Provider>;
    fn reason(&self, ctx: ReasonCtx) -> Result<ReasonResult, BrainError>;
    async fn stream(&self, ctx: ReasonCtx) -> Result<TokenStream, BrainError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Embedded,
    Edge,
    Desktop,
}

#[cfg(feature = "std")]
use alloc::string::String;
#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use serde_json::Value;

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct ReasonCtx {
    pub intent_id: String,
    pub action: String,
    pub payload: Value,
    pub memory: Vec<MemoryItem>,
    pub domain: Option<String>,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct MemoryItem {
    pub key: String,
    pub value: Value,
    pub relevance: f32,
}

#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub aep: u64,
    pub tokens: u32,
    pub estimated_ms: u64,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct ReasonResult {
    pub output: Value,
    pub confidence: f32,
    pub tokens_used: u32,
    pub model_id: String,
    pub plan: Option<Vec<String>>,
}

pub struct TokenStream {
    pub session_id: u64,
    pub finished: bool,
}

#[derive(Debug)]
pub enum BrainError {
    ModelUnavailable(String),
    ContextTooLarge(u32),
    Timeout(u64),
    ProcessFailed(String),
}