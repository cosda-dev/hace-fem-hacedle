mod budget;
mod limits;
mod policy;

pub use budget::BudgetGuard;
pub use limits::ResourceLimits;
pub use policy::AuthorityPolicy;

#[derive(Debug, Clone, Copy)]
pub enum BudgetAction {
    Continue,
    Throttle,
    Stop,
    Fallback,
}

#[derive(Debug, Clone, Copy)]
pub struct BudgetConfig {
    pub max_tokens: u32,
    pub max_duration_ms: u64,
    pub max_memory_mb: u64,
}