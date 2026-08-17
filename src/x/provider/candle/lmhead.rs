use alloc::vec::Vec;
use alloc::vec;

/// LM Head - vocab projection for logits output
pub struct LMHead {
    pub weight: Vec<f32>,
    pub vocab_size: usize,
    pub embed_dim: usize,
}

impl LMHead {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        Self { weight: Vec::new(), vocab_size, embed_dim }
    }

    /// Load weights from GGUF model
    pub fn load_weight(&mut self, weight: Vec<f32>) {
        self.weight = weight;
    }

    /// Compute logits: hidden @ weight.T
    pub fn forward(&self, hidden: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.vocab_size];

        // Matrix multiplication: hidden (embed_dim,) @ weight (vocab_size, embed_dim)
        // logits[v] = sum(hidden[i] * weight[v * embed_dim + i])
        let hidden_len = hidden.len().min(self.embed_dim);

        for v in 0..self.vocab_size.min(self.weight.len() / self.embed_dim) {
            let mut sum = 0.0f32;
            for i in 0..hidden_len {
                sum += hidden[i] * self.weight[v * self.embed_dim + i];
            }
            logits[v] = sum;
        }

        logits
    }
}

/// Logits processor for sampling
pub struct LogitsProcessor {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl LogitsProcessor {
    pub fn new(temperature: f32, top_p: f32, top_k: usize) -> Self {
        Self { temperature, top_p, top_k }
    }

    /// Apply temperature scaling
    pub fn apply_temperature(&self, logits: &mut [f32]) {
        let t = self.temperature.max(0.001);
        for l in logits.iter_mut() {
            *l /= t;
        }
    }

    /// Apply top-p (nucleus) sampling
    pub fn apply_top_p(&self, logits: &mut [f32]) {
        let mut sorted: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(0.cmp(&0)));

        let mut cumulative = 0.0f32;
        let mut mask_threshold = 0.0f32;

        for (_, prob) in &sorted {
            let p = *prob;
            cumulative += p.exp().min(1.0);
            if cumulative >= self.top_p {
                mask_threshold = p;
                break;
            }
        }

        for l in logits.iter_mut() {
            if *l < mask_threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply top-k filtering
    pub fn apply_top_k(&self, logits: &mut [f32]) {
        let mut sorted: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(0.cmp(&0)));

        let threshold = if self.top_k > 0 && self.top_k < sorted.len() {
            sorted[self.top_k].1
        } else {
            f32::NEG_INFINITY
        };

        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// Process logits for sampling
    pub fn process(&self, logits: Vec<f32>) -> Vec<f32> {
        let mut processed = logits;
        self.apply_temperature(&mut processed);
        self.apply_top_p(&mut processed);
        self.apply_top_k(&mut processed);
        processed
    }
}