mod intent_header;

pub use intent_header::IntentHeader;

pub struct NarrativeBlock;
pub struct TechnicalBlock;

impl NarrativeBlock {
    pub fn new() -> Self { Self {} }
}
impl TechnicalBlock {
    pub fn new() -> Self { Self {} }
}
