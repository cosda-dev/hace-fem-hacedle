mod stub;

pub use stub::AttentionStub;

pub trait AttentionExecutor {
    fn forward(
        &self,
        token: u32,
        kv: &mut crate::runtime::kv_cache::KvArena,
    );
}