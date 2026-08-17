// Adapter Registry - Manage multiple LoRA adapters

#[cfg(feature = "std")]
use std::collections::BTreeMap;
#[cfg(feature = "std")]
use std::sync::Arc;

#[cfg(feature = "std")]
pub struct AdapterRegistry {
    pub adapters: BTreeMap<String, LoraAdapter>,
    pub active_adapters: Vec<String>,
}

#[cfg(feature = "std")]
impl AdapterRegistry {
    pub fn new() -> Self {
        Self {
            adapters: BTreeMap::new(),
            active_adapters: Vec::new(),
        }
    }
    
    pub fn register(&mut self, adapter: LoraAdapter) {
        self.adapters.insert(adapter.id.clone(), adapter);
    }
    
    pub fn activate(&mut self, adapter_id: &str) {
        if !self.active_adapters.contains(&adapter_id.to_string()) {
            self.active_adapters.push(adapter_id.to_string());
        }
    }
}

pub struct LoraAdapter {
    pub id: String,
    pub rank: usize,
    pub alpha: usize,
}