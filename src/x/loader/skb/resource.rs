use alloc::string::String;
use alloc::vec::Vec;

pub struct ResourceLayer {
    pub urls: Vec<String>,
    pub artifacts: Vec<String>,
    pub media: Vec<String>,
}

impl ResourceLayer {
    pub fn new() -> Self {
        Self {
            urls: Vec::new(),
            artifacts: Vec::new(),
            media: Vec::new(),
        }
    }
}

impl Default for ResourceLayer {
    fn default() -> Self {
        Self::new()
    }
}
