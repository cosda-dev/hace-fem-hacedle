use alloc::string::String;

pub struct ProtocolLayer {
    pub workflow: String,
    pub execution_flow: String,
}

impl ProtocolLayer {
    pub fn new() -> Self {
        Self {
            workflow: String::new(),
            execution_flow: String::new(),
        }
    }
}

impl Default for ProtocolLayer {
    fn default() -> Self {
        Self::new()
    }
}
