#[derive(Debug, Clone, Copy)]
pub struct ResourceLimits {
    pub max_threads: u32,
    pub max_batch: u32,
    pub max_context: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_threads: 8,
            max_batch: 512,
            max_context: 8192,
        }
    }
}