use alloc::string::String;
use alloc::vec::Vec;

use super::header::QuantizationType;
use super::tensor_index::TensorInfo;

pub struct LoadedModel {
    pub architecture: String,
    pub quantization: QuantizationType,
    pub tensor_count: usize,
    pub context_length: usize,
    pub embedding_length: usize,
    pub tensors: Vec<TensorInfo>,
}

impl LoadedModel {
    pub fn new() -> Self {
        Self {
            architecture: String::from("qwen2"),
            quantization: QuantizationType::Q4K_M,
            tensor_count: 0,
            context_length: 8192,
            embedding_length: 1536,
            tensors: Vec::new(),
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.architecture.is_empty() && self.tensor_count > 0
    }
}

impl Default for LoadedModel {
    fn default() -> Self {
        Self::new()
    }
}
