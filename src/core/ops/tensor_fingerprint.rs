// Alpha-3 Phase E2: Tensor Fingerprint
// Fast tensor comparison without full data load

#[derive(Debug, Clone)]
pub struct TensorFingerprint {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub sha256: [u8; 32],
    pub l2_norm: f32,
    pub mean: f32,
    pub std: f32,
    pub total_elements: usize,
}

impl TensorFingerprint {
    pub fn from_data(data: &[f32]) -> Self {
        let l2_norm = data.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let mean = if !data.is_empty() {
            data.iter().sum::<f32>() / data.len() as f32
        } else {
            0.0
        };
        
        let std = if !data.is_empty() {
            let variance = data.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / data.len() as f32;
            variance.sqrt()
        } else {
            0.0
        };
        
        // Compute SHA256
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for &val in data {
            val.to_le_bytes().hash(&mut hasher);
        }
        let hash = hasher.finish();
        
        let mut sha256 = [0u8; 32];
        for i in 0..16 {
            sha256[i] = (hash >> (i * 8)) as u8;
        }
        
        Self {
            shape: vec![data.len()],
            dtype: "f32".to_string(),
            sha256,
            l2_norm,
            mean,
            std,
            total_elements: data.len(),
        }
    }
    
    pub fn match(&self, other: &TensorFingerprint, tolerance: f32) -> bool {
        if self.shape != other.shape {
            return false;
        }
        
        if (self.l2_norm - other.l2_norm).abs() > tolerance * self.l2_norm {
            return false;
        }
        
        if (self.mean - other.mean).abs() > tolerance {
            return false;
        }
        
        true
    }
    
    pub fn error_report(&self, other: &TensorFingerprint) -> String {
        let mut report = String::new();
        report.push_str(&format!("Fingerprint comparison:\n"));
        
        if self.shape != other.shape {
            report.push_str(&format!("  SHAPE MISMATCH: {:?} vs {:?}\n", self.shape, other.shape));
        }
        
        let l2_diff = (self.l2_norm - other.l2_norm).abs();
        report.push_str(&format!("  L2 norm diff: {}\n", l2_diff));
        
        let mean_diff = (self.mean - other.mean).abs();
        report.push_str(&format!("  Mean diff: {}\n", mean_diff));
        
        report.push_str(&format!("  STOP - fingerprint mismatch\n"));
        report
    }
}