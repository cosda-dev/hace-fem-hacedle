use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct Port {
    pub id: String,
    pub input: Vec<String>,
    pub output: Vec<String>,
}

pub struct PortTable {
    pub ports: Vec<Port>,
}

impl PortTable {
    pub fn new() -> Self {
        Self { ports: Vec::new() }
    }

    pub fn add(&mut self, port: Port) {
        self.ports.push(port);
    }

    pub fn find(&self, id: &str) -> Option<&Port> {
        self.ports.iter().find(|p| p.id == id)
    }
}

impl Default for PortTable {
    fn default() -> Self {
        Self::new()
    }
}
