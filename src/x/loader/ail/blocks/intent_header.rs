use alloc::string::String;

#[derive(Debug, Clone)]
pub struct IntentHeader {
    pub id: String,
    pub intent: String,
    pub status: String,
    pub locale: String,
    pub authority: String,
    pub version: String,
}

impl IntentHeader {
    pub fn is_valid(&self) -> bool {
        !self.id.is_empty() && !self.intent.is_empty()
    }
}
