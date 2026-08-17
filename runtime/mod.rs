mod execute;
mod inference;
mod provider;
mod context;
mod outcome;
mod kv_cache;
pub mod attention;
pub mod sampler;

pub use execute::HacedleExecutor;
pub use inference::InferenceEngine;
pub use kv_cache::{KvArena, SessionKv, SoulKv, SoulMemoryRegistry, KvCacheManager, KvError, KvSnapshot};
pub use outcome::HacedleOutcome;
pub use attention::AttentionStub;
pub use sampler::ArgMaxSampler;

use alloc::string::String;
use alloc::vec::Vec;

pub struct HacedleRuntime {
    pub model: String,
    pub context_size: u32,
}

impl HacedleRuntime {
    pub fn new() -> Self {
        Self {
            model: String::from("default"),
            context_size: 8192,
        }
    }
}