use alloc::string::String;
use alloc::vec::Vec;

use crate::x::loader::ail::{IntentHeader, NarrativeBlock, TechnicalBlock};
use crate::x::loader::ail::validator::ValidationReport;

pub struct AilLoader;

impl AilLoader {
    pub fn load(file_path: &str) -> Result<ExecutionContext, &'static str> {
        Ok(ExecutionContext {
            header: IntentHeader {
                id: String::new(),
                intent: String::new(),
                status: String::from("ACTIVE"),
                locale: String::from("vi-85"),
                authority: String::new(),
                version: String::from("1.0"),
            },
            narrative: NarrativeBlock::new(),
            technical: TechnicalBlock::new(),
            mto: MtoDictionary::new(),
        })
    }
}

pub struct ExecutionContext {
    pub header: IntentHeader,
    pub narrative: NarrativeBlock,
    pub technical: TechnicalBlock,
    pub mto: MtoDictionary,
}

pub struct MtoDictionary {
    pub mappings: Vec<(String, String)>,
}

impl MtoDictionary {
    pub fn new() -> Self {
        Self { mappings: Vec::new() }
    }

    pub fn resolve(&self, term: &str) -> Option<&str> {
        self.mappings.iter().find(|(k, _)| k == term).map(|(_, v)| v.as_str())
    }
}

impl Default for MtoDictionary {
    fn default() -> Self {
        Self::new()
    }
}
