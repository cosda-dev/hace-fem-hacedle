use alloc::string::String;

use super::{SkbHeader, AuthorityBlock, SemanticGraph, MemoryProjection, PayloadView};

#[derive(Debug)]
pub enum SecurityError {
    InvalidHeader,
    FailedVerification,
    IoError(String),
}

#[derive(Debug)]
pub struct SealInfo {
    pub is_valid: bool,
    pub author: String,
    pub trust_level: f32,
}

pub trait SkbMountNep {
    fn verify(&self, path: &str) -> Result<SealInfo, SecurityError>;

    fn mmap(&self, path: &str) -> Result<MemoryProjection, SecurityError>;

    fn semantic_graph(&self, proj: &MemoryProjection) -> Result<SemanticGraph, SecurityError>;

    fn payload(&self, proj: &MemoryProjection) -> Result<PayloadView, SecurityError>;
}

pub struct SkbLoader;

impl SkbLoader {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SkbLoader {
    fn default() -> Self {
        Self::new()
    }
}
