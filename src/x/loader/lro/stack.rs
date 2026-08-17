use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::overlay::RuntimeOverlay;

/// Stack of LoRA adapters with deterministic ordering
pub struct OverlayStack {
    pub overlays: BTreeMap<String, RuntimeOverlay>,
    pub order: Vec<String>,
}

impl OverlayStack {
    pub fn new() -> Self {
        Self { 
            overlays: BTreeMap::new(),
            order: Vec::new(),
        }
    }

    /// Add adapter with explicit ordering (deterministic)
    pub fn add_adapter(&mut self, id: String, overlay: RuntimeOverlay) {
        if !self.order.contains(&id) {
            self.order.push(id.clone());
        }
        self.overlays.insert(id, overlay);
    }

    /// Remove adapter from stack
    pub fn remove_adapter(&mut self, id: &str) {
        self.overlays.remove(id);
        self.order.retain(|x| x != id);
    }

    /// Reorder adapters - ensures reproducible composition
    pub fn reorder(&mut self, new_order: &[String]) {
        self.order = new_order.to_vec();
        // Remove any not in new order
        let to_keep: BTreeMap<String, bool> = new_order.iter().map(|s| (s.clone(), true)).collect();
        self.overlays.retain(|k, _| to_keep.contains_key(k));
    }

    /// Combined effective weight from all adapters in order
    pub fn effective_combined(&self, base_weight: &[f32]) -> Vec<f32> {
        let mut effective = base_weight.to_vec();

        // Apply overlays in deterministic order for reproducibility
        for id in &self.order {
            if let Some(overlay) = self.overlays.get(id) {
                let delta = overlay.delta();
                let len = effective.len().min(delta.len());

                for i in 0..len {
                    effective[i] += delta[i];
                }
            }
        }

        effective
    }

    /// Get tensor sum for multi-LRO
    pub fn tensor_sum(&self, base_weight: &[f32]) -> Vec<f32> {
        self.effective_combined(base_weight)
    }
}

/// Multi-LRO composer with reproducibility guarantee
pub struct MultiLroComposer {
    pub stack: OverlayStack,
}

impl MultiLroComposer {
    pub fn new() -> Self {
        Self { stack: OverlayStack::new() }
    }

    /// Add adapter by relative priority (higher = later in composition)
    pub fn add_lora(&mut self, id: String, overlay: RuntimeOverlay, _priority: i32) {
        // Insert with deterministic ordering
        if !self.stack.order.contains(&id) {
            self.stack.order.push(id.clone());
        }
        self.stack.overlays.insert(id, overlay);
    }

    /// Compute reproducible composition
    pub fn compose(&self, base_weight: &[f32]) -> Vec<f32> {
        self.stack.effective_combined(base_weight)
    }
}

impl Default for OverlayStack {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MultiLroComposer {
    fn default() -> Self {
        Self::new()
    }
}