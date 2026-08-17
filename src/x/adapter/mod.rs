use alloc::string::String;
use alloc::vec::Vec;

pub struct AdapterDescriptor {
    pub id: String,
}

pub struct GGUFAdapter;
pub struct SafeTensorAdapter;
pub struct ONNXAdapter;
pub struct MMBAdapter;

pub trait ModelAdapter {
    fn format_name(&self) -> &'static str;
    fn load_model(&self, path: &str) -> Result<ModelHandle, &'static str>;
}

#[derive(Debug, Clone, Copy)]
pub struct ModelHandle {
    pub id: u64,
}

pub struct AdapterRegistry {
    adapters: Vec<&'static dyn ModelAdapter>,
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    pub fn register(&mut self, _adapter: &'static dyn ModelAdapter) {
        // TODO
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}
