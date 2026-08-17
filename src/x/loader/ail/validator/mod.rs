use alloc::string::String;
use alloc::vec::Vec;

use crate::x::loader::ail::blocks::IntentHeader;

pub struct AilValidator {
    header: IntentHeader,
}

impl AilValidator {
    pub fn new(header: IntentHeader) -> Self {
        Self { header }
    }

    pub fn validate(&self) -> ValidationReport {
        let errors = Vec::new();
        let is_valid = self.header.is_valid();
        ValidationReport { is_valid, errors }
    }
}

pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self { is_valid: true, errors: Vec::new() }
    }
}
