// Dequant Dispatcher - Dispatch quantized tensors to compute backends
// Supports lazy dequantization for zero-copy inference

use super::q4k_tensor_view::{QuantTensorView, QuantType};
use crate::alloc_exports::*;

#[cfg(feature = "std")]
use std::sync::Arc;

pub trait ComputeBackend {
    fn matmul_q4k(&self, a: &QuantTensorView, b: &QuantTensorView, output: &mut [f32]);
    fn matmul_q6k(&self, a: &QuantTensorView, b: &QuantTensorView, output: &mut [f32]);
    fn matmul_f32(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize, output: &mut [f32]);
    fn rmsnorm(&self, input: &[f32], weight: &[f32], output: &mut [f32]);
    fn rope(&self, input: &mut [f32], pos: usize, dim: usize);
    fn softmax(&self, input: &mut [f32]);
}

pub struct NativeBackend;

impl NativeBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for NativeBackend {
    fn matmul_q4k(&self, a: &QuantTensorView, b: &QuantTensorView, output: &mut [f32]) {
        let a_dequant = a.dequantize();
        let b_dequant = b.dequantize();
        
        let m = a.shape[0].min(output.len());
        let n = if b.shape.len() > 1 { b.shape[1] } else { 1 };
        let k = a.shape[1].min(b_dequant.len() / n);
        
        output.fill(0.0);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_dequant[i * k + l] * b_dequant[l * n + j];
                }
                if i * n + j < output.len() {
                    output[i * n + j] = sum;
                }
            }
        }
    }

    fn matmul_q6k(&self, a: &QuantTensorView, b: &QuantTensorView, output: &mut [f32]) {
        let a_dequant = a.dequantize();
        let b_dequant = b.dequantize();
        
        let m = a.shape[0].min(output.len());
        let n = if b.shape.len() > 1 { b.shape[1] } else { 1 };
        let k = a.shape[1].min(b_dequant.len() / n);
        
        output.fill(0.0);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_dequant[i * k + l] * b_dequant[l * n + j];
                }
                if i * n + j < output.len() {
                    output[i * n + j] = sum;
                }
            }
        }
    }

    fn matmul_f32(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize, output: &mut [f32]) {
        output.fill(0.0);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                output[i * n + j] = sum;
            }
        }
    }

    fn rmsnorm(&self, input: &[f32], weight: &[f32], output: &mut [f32]) {
        let len = input.len().min(weight.len()).min(output.len());
        let ss: f32 = input[..len].iter().map(|&x| x * x).sum();
        let rms = (ss / len as f32 + 1e-6).sqrt().recip();
        
        for i in 0..len {
            output[i] = input[i] * weight[i] * rms;
        }
    }

    fn rope(&self, input: &mut [f32], pos: usize, dim: usize) {
        let base: f32 = 1000000.0;  // Qwen2.5 rope_theta
        for i in 0..dim / 2 {
            let freq = base.powf(2.0 * (i as f32) / (dim as f32));
            let inv_freq = 1.0 / freq;
            let angle = pos as f32 * inv_freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            
            let idx0 = i;
            let idx1 = i + dim / 2;
            
            if idx1 < input.len() {
                let x1 = input[idx0];
                let x2 = input[idx1];
                
                input[idx0] = x1 * cos_val - x2 * sin_val;
                input[idx1] = x1 * sin_val + x2 * cos_val;
            }
        }
    }

    fn softmax(&self, input: &mut [f32]) {
        if input.is_empty() {
            return;
        }
        
        let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = input.iter().map(|&x| (x - max_val).exp()).sum();
        
        for x in input.iter_mut() {
            *x = (*x - max_val).exp() / sum;
        }
    }
}

impl Default for NativeBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "std")]
pub struct DequantDispatcher {
    pub backend: Arc<dyn ComputeBackend + Send + Sync>,
}

#[cfg(feature = "std")]
impl DequantDispatcher {
    pub fn new(backend: Arc<dyn ComputeBackend + Send + Sync>) -> Self {
        Self { backend }
    }

    pub fn dispatch_matmul(&self, a: &QuantTensorView, b: &QuantTensorView, output: &mut [f32]) {
        match a.quant_type {
            QuantType::Q4K => self.backend.matmul_q4k(a, b, output),
            QuantType::Q6K => self.backend.matmul_q6k(a, b, output),
            _ => { /* fallback */ }
        }
    }
}