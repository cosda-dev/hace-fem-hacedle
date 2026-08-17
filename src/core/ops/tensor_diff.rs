// T6.4: First Divergence Detector
// Compare Rust operator outputs with golden reference

use alloc::vec::Vec;

pub struct TensorDiff {
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub cosine_similarity: f32,
    pub fingerprint_match: bool,
}

impl TensorDiff {
    pub fn compare(actual: &[f32], reference: &[f32]) -> Self {
        let len = actual.len().min(reference.len());
        
        if len == 0 {
            return Self {
                max_abs_error: f32::NAN,
                mean_abs_error: f32::NAN,
                cosine_similarity: f32::NAN,
                fingerprint_match: false,
            };
        }
        
        // Max absolute error
        let max_abs_error = actual[..len]
            .iter()
            .zip(reference[..len].iter())
            .map(|(a, r)| (a - r).abs())
            .fold(0.0f32, f32::max);
        
        // Mean absolute error
        let mean_abs_error = actual[..len]
            .iter()
            .zip(reference[..len].iter())
            .map(|(a, r)| (a - r).abs())
            .sum::<f32>() / len as f32;
        
        // Cosine similarity
        let dot = actual[..len]
            .iter()
            .zip(reference[..len].iter())
            .map(|(a, r)| a * r)
            .sum::<f32>();
        
        let norm_a = actual[..len]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        
        let norm_r = reference[..len]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        
        let cosine_similarity = if norm_a > 0.0 && norm_r > 0.0 {
            dot / (norm_a * norm_r)
        } else {
            1.0
        };
        
        // Fingerprint (simple hash - production should use SHA256)
        let fingerprint_match = max_abs_error < 1e-6;
        
        Self {
            max_abs_error,
            mean_abs_error,
            cosine_similarity,
            fingerprint_match,
        }
    }
}

pub struct OperatorReport {
    pub operator_name: &'static str,
    pub diff: TensorDiff,
    pub status: &'static str,
}