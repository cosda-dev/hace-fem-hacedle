use alloc::string::String;
use alloc::vec::Vec;

pub struct RulesetLayer {
    pub constraints: Vec<String>,
    pub compliance: Vec<String>,
}

impl RulesetLayer {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            compliance: Vec::new(),
        }
    }
}

impl Default for RulesetLayer {
    fn default() -> Self {
        Self::new()
    }
}
