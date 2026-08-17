use crate::runtime::kv_cache::KvArena;

pub struct AttentionStub;

impl AttentionStub {
    pub fn new() -> Self {
        Self
    }
}

impl super::AttentionExecutor for AttentionStub {
    fn forward(&self, _token: u32, _kv: &mut KvArena) {
    }
}

impl Default for AttentionStub {
    fn default() -> Self {
        Self::new()
    }
}