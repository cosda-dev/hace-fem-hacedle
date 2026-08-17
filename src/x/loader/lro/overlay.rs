// LRO Overlay - Runtime tensor overlay without mutation
// W_effective = W_base + scale * (B @ A)

use alloc::vec::Vec;

/// Runtime Overlay Tensor
pub struct RuntimeOverlay {
    pub base_weight: Vec<f32>,
    pub lora_a: Vec<f32>,
    pub lora_b: Vec<f32>,
    pub scale: f32,
}

impl RuntimeOverlay {
    pub fn new(base: Vec<f32>, lora_a: Vec<f32>, lora_b: Vec<f32>, scale: f32) -> Self {
        Self {
            base_weight: base,
            lora_a,
            lora_b,
            scale,
        }
    }
    
    /// Compute effective weight at runtime - NO mutation
    pub fn effective(&self) -> Vec<f32> {
        // B @ A = lora_b * lora_a (matrix multiplication)
        // effective = base + scale * (B @ A)
        // For simplicity: element-wise delta application
        let mut effective = self.base_weight.clone();
        
        let len = effective.len().min(self.lora_a.len()).min(self.lora_b.len());
        
        for i in 0..len {
            let delta = self.scale * self.lora_a[i] * self.lora_b[i];
            effective[i] += delta;
        }
        
        effective
    }
    
    /// Apply overlay and return deltas for verification
    pub fn delta(&self) -> Vec<f32> {
        let len = self.base_weight.len().min(self.lora_a.len()).min(self.lora_b.len());
        (0..len)
            .map(|i| self.scale * self.lora_a[i] * self.lora_b[i])
            .collect()
    }
}