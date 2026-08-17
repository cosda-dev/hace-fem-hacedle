pub struct ArgMaxSampler;

impl ArgMaxSampler {
    pub fn new() -> Self {
        Self
    }

    pub fn sample(logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        
        let mut max_idx = 0;
        let mut max_val = logits[0];
        
        for (idx, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }
        
        max_idx as u32
    }
}

impl Default for ArgMaxSampler {
    fn default() -> Self {
        Self::new()
    }
}