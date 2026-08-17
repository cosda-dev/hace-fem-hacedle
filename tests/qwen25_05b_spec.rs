// Qwen2.5-0.5B Model Specifications
// For reference when running parity tests

/// Qwen2.5-0.5B Architecture:
/// - hidden_size: 896
/// - intermediate_size: 2816 (FFN hidden size)
/// - num_heads: 14
/// - head_dim: 64
/// - num_layers: 24
/// - vocab_size: 151936

pub struct Qwen25_0_5B_Spec;

impl Qwen25_0_5B_Spec {
    pub const fn hidden_size() -> usize { 896 }
    pub const fn intermediate_size() -> usize { 2816 }
    pub const fn num_heads() -> usize { 14 }
    pub const fn head_dim() -> usize { 64 }
    pub const fn num_layers() -> usize { 24 }
    pub const fn rope_theta() -> f32 { 1000000.0 }
}