// Alpha-3 Phase E3: Operator Registry
// Structured operator testing with detailed error reporting

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct OperatorResult {
    pub name: String,
    pub layer: usize,
    pub head: Option<usize>,
    pub position: Option<usize>,
    pub input_hash: String,
    pub expected_hash: String,
    pub actual_hash: String,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub cosine_similarity: f32,
    pub passed: bool,
}

impl OperatorResult {
    pub fn fail(name: &str) -> Self {
        Self {
            name: name.to_string(),
            layer: 0,
            head: None,
            position: None,
            max_abs_error: f32::MAX,
            mean_abs_error: f32::MAX,
            cosine_similarity: 0.0,
            passed: false,
        }
    }
}

pub struct OperatorRegistry {
    pub results: HashMap<String, OperatorResult>,
}

impl OperatorRegistry {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }
    
    pub fn register_result(&mut self, result: OperatorResult) {
        self.results.insert(result.name.clone(), result);
    }
    
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("Operator Registry Report:\n");
        
        for (name, result) in &self.results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            report.push_str(&format!(
                "  {} {}: max_abs={:.2e}, cosine={:.6}\n",
                name, status, result.max_abs_error, result.cosine_similarity
            ));
            
            if !result.passed {
                if let Some(layer) = result.layer {
                    report.push_str(&format!("    layer={}\n", layer));
                }
                if let Some(head) = result.head {
                    report.push_str(&format!("    head={}\n", head));
                }
                if let Some(pos) = result.position {
                    report.push_str(&format!("    position={}\n", pos));
                }
            }
        }
        
        report
    }
    
    pub fn all_passed(&self) -> bool {
        self.results.values().all(|r| r.passed)
    }
}

pub struct OperatorTest {
    pub name: String,
    pub layer: usize,
    pub head: Option<usize>,
    pub position: Option<usize>,
}

impl OperatorTest {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            layer: 0,
            head: None,
            position: None,
        }
    }
    
    pub fn with_dims(mut self, layer: usize, head: usize, position: usize) -> Self {
        self.layer = layer;
        self.head = Some(head);
        self.position = Some(position);
        self
    }
    
    pub fn compare(&self, our: &[f32], reference: &[f32]) -> OperatorResult {
        let min_len = our.len().min(reference.len());
        
        if min_len == 0 {
            return OperatorResult::fail(&self.name);
        }
        
        let max_abs: f32 = our.iter().zip(reference.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        
        let mean_abs: f32 = our.iter().zip(reference.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>() / min_len as f32;
        
        let dot: f32 = our[..min_len].iter().zip(reference[..min_len].iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let norm_a: f32 = (our[..min_len].iter().map(|&x| x * x).sum::<f32>()).sqrt();
        let norm_b: f32 = (reference[..min_len].iter().map(|&x| x * x).sum::<f32>()).sqrt();
        let cosine = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            1.0
        };
        
        OperatorResult {
            name: self.name.clone(),
            layer: self.layer,
            head: self.head,
            position: self.position,
            max_abs_error: max_abs,
            mean_abs_error: mean_abs,
            cosine_similarity: cosine,
            passed: max_abs < 1e-6 && cosine > 0.99999,
        }
    }
}