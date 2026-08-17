#[derive(Debug, Clone)]
pub struct AuthorityPolicy {
    pub allow_remote_fallback: bool,
    pub enforce_budget: bool,
    pub collect_evidence: bool,
}

impl Default for AuthorityPolicy {
    fn default() -> Self {
        Self {
            allow_remote_fallback: true,
            enforce_budget: true,
            collect_evidence: true,
        }
    }
}