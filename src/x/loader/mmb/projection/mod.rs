use alloc::vec::Vec;

pub struct ProjectionTensor {
    pub dimensions: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct MmbProjection {
    pub tensors: Vec<ProjectionTensor>,
}

impl MmbProjection {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }
}

impl Default for MmbProjection {
    fn default() -> Self {
        Self::new()
    }
}
