use alloc::string::String;
use alloc::vec::Vec;

use crate::x::loader::ail::blocks::IntentHeader;

pub struct AilParser {
    content: String,
}

impl AilParser {
    pub fn new(content: String) -> Self {
        Self { content }
    }

    pub fn parse(&self) -> ParseResult {
        ParseResult {
            header: IntentHeader {
                id: String::new(),
                intent: String::new(),
                status: String::from("ACTIVE"),
                locale: String::from("vi-85"),
                authority: String::new(),
                version: String::from("1.0"),
            },
        }
    }
}

pub struct ParseResult {
    pub header: IntentHeader,
}

impl Default for ParseResult {
    fn default() -> Self {
        Self {
            header: IntentHeader {
                id: String::new(),
                intent: String::new(),
                status: String::from("ACTIVE"),
                locale: String::from("vi-85"),
                authority: String::new(),
                version: String::from("1.0"),
            },
        }
    }
}
