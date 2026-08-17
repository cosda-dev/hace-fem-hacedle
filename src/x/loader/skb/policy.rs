use alloc::string::String;
use alloc::vec::Vec;
use alloc::vec;

#[derive(Debug, Clone)]
pub struct PolicyBlock {
    pub allow: Vec<String>,
    pub deny: Vec<String>,
    pub audit: bool,
}

impl PolicyBlock {
    pub fn is_allowed(&self, action: &str) -> bool {
        if self.deny.contains(&action.to_string()) {
            return false;
        }
        if self.allow.is_empty() {
            return true;
        }
        self.allow.contains(&action.to_string())
    }
}

impl Default for PolicyBlock {
    fn default() -> Self {
        Self {
            allow: vec![String::from("query"), String::from("summarize")],
            deny: vec![String::from("modify")],
            audit: true,
        }
    }
}