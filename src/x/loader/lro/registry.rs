// LRO Registry - Manage multiple LoRA adapters

use alloc::string::String;
use alloc::vec::Vec;
use super::parser::LoraOverlayCanon;

/// Adapter Registry for Alliag/Bra
pub struct LoraRegistry {
    pub adapters: Vec<RegisteredAdapter>,
}

impl Default for LoraRegistry {
    fn default() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }
}

impl LoraRegistry {
    pub fn register(&mut self, canon: LoraOverlayCanon) {
        self.adapters.push(RegisteredAdapter {
            canon,
            active: false,
        });
    }
    
    pub fn activate(&mut self, adapter_id: &str) {
        for adapter in &mut self.adapters {
            if adapter.canon.adapter_id == adapter_id {
                adapter.active = true;
            }
        }
    }
    
    pub fn list(&self) -> Vec<&LoraOverlayCanon> {
        self.adapters.iter().map(|a| &a.canon).collect()
    }
}

pub struct RegisteredAdapter {
    pub canon: LoraOverlayCanon,
    pub active: bool,
}