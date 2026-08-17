use alloc::string::String;

pub struct MemoryProjection {
    pub kv_cache: KvCache,
    pub token_cache: TokenCache,
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub struct TokenCache {
    pub offset: u64,
    pub size: u64,
}

impl MemoryProjection {
    pub fn new() -> Self {
        Self {
            kv_cache: KvCache { offset: 0, size: 0 },
            token_cache: TokenCache { offset: 0, size: 0 },
        }
    }
}

impl Default for MemoryProjection {
    fn default() -> Self {
        Self::new()
    }
}
