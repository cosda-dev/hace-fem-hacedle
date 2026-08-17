// P5.1: GGA Runtime Shape Validation
// Dump tensor shapes at runtime for verification

#[derive(Debug, Clone)]
pub struct GgaShapeValidation {
    pub q_shape: Vec<usize>,
    pub k_shape: Vec<usize>,
    pub v_shape: Vec<usize>,
    pub n_heads: usize,
    pub n_heads_kv: usize,
    pub head_dim: usize,
    pub repeat_factor: usize,
}

impl GgaShapeValidation {
    pub fn validate(&self) -> Result<(), String> {
        let expected_repeat = self.n_heads / self.n_heads_kv;
        
        // Q shape: [seq_len, n_heads, head_dim]
        let q_last_dim = *self.q_shape.last().unwrap_or(&0);
        let k_last_dim = *self.k_shape.last().unwrap_or(&0);
        let v_last_dim = *self.v_shape.last().unwrap_or(&0);
        
        if q_last_dim != self.n_heads * self.head_dim {
            return Err(format!(
                "Q shape mismatch: expected last_dim={}, got {}", 
                self.n_heads * self.head_dim, q_last_dim
            ));
        }
        
        if k_last_dim != self.n_heads_kv * self.head_dim {
            return Err(format!(
                "K shape mismatch: expected last_dim={}, got {}", 
                self.n_heads_kv * self.head_dim, k_last_dim
            ));
        }
        
        if v_last_dim != self.n_heads_kv * self.head_dim {
            return Err(format!(
                "V shape mismatch: expected last_dim={}, got {}", 
                self.n_heads_kv * self.head_dim, v_last_dim
            ));
        }
        
        if self.repeat_factor != expected_repeat {
            return Err(format!(
                "Repeat factor mismatch: expected={}, got {}", 
                expected_repeat, self.repeat_factor
            ));
        }
        
        Ok(())
    }
}

pub fn repeat_kv(head_idx: usize, n_heads_kv: usize) -> usize {
    head_idx % n_heads_kv
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gga_shape_validation_qwen25() {
        let validation = GgaShapeValidation {
            q_shape: vec![1, 14, 64],
            k_shape: vec![1, 2, 64],
            v_shape: vec![1, 2, 64],
            n_heads: 14,
            n_heads_kv: 2,
            head_dim: 64,
            repeat_factor: 7,
        };
        
        assert!(validation.validate().is_ok(), "GGA shape validation should pass for Qwen2.5-0.5B");
    }
}