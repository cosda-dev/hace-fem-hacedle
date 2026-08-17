use alloc::string::String;

#[derive(Debug, Clone)]
pub struct SkbHeader {
    pub magic: [u8; 4],
    pub version: [u8; 3],
    pub skb_id: Option<String>,
    pub author_id: Option<String>,
    pub owner_id: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub checksum: [u8; 32],
}

impl SkbHeader {
    pub fn magic(&self) -> &[u8; 4] {
        &self.magic
    }

    pub fn is_valid(&self) -> bool {
        self.magic == *b"SKB1"
    }
}

pub const SKB_MAGIC: [u8; 4] = *b"SKB1";
