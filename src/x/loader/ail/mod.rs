mod parser;
mod blocks;
mod validator;
mod context_builder;

pub use parser::{AilParser, ParseResult};
pub use blocks::{IntentHeader, NarrativeBlock, TechnicalBlock};
pub use validator::{AilValidator, ValidationReport};
pub use context_builder::{AilLoader, ExecutionContext, MtoDictionary};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationError {
    MissingHeader,
    InvalidHeader,
    MissingIntent,
    InvalidFormat,
}

pub trait MtoResolver {
    fn resolve(&self, term: &str) -> Option<&'static str>;
}
