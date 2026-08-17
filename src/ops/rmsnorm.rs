// RMSNorm operator for parity testing

#[cfg(feature = "std")]
use std::f32;

use alloc::vec::Vec;

pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let len = input.len().min(weight.len());
    let ss: f32 = input[..len].iter().map(|&x| x * x).sum();
    let rms = (ss / len as f32 + eps).sqrt().recip();
    input[..len].iter().zip(weight[..len].iter()).map(|(&x, &w)| x * w * rms).collect()
}

pub fn rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    rms_norm(input, weight, eps)
}

/// Final RMSNorm for output projection before LM Head
pub struct FinalRMSNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl FinalRMSNorm {
    pub fn new(eps: f32) -> Self {
        Self { weight: Vec::new(), eps }
    }

    pub fn load_weight(&mut self, weight: Vec<f32>) {
        self.weight = weight;
    }

    pub fn forward(&self, hidden: &[f32]) -> Vec<f32> {
        rms_norm(hidden, &self.weight, self.eps)
    }
}