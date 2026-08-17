// Directive B: First Divergence Detector v2
// Output: layer_id, operator_id, tensor_index, SHA, parent, error

#[derive(Debug)]
pub struct FirstDivergence {
    pub layer_id: usize,
    pub operator_id: String,
    pub input_sha: String,
    pub output_sha: String,
    pub tensor_index: usize,
    pub expected: f32,
    pub actual: f32,
    pub abs_error: f32,
    pub mean_abs_error: f32,
    pub cosine_similarity: f32,
    pub parent_operator: String,
}

pub fn find_first_divergence(runtime: &[f32], golden: &[f32], operator_id: &str, layer_id: usize) -> Option<FirstDivergence> {
    let min_len = runtime.len().min(golden.len());
    
    // Early exit: shape mismatch
    if runtime.len() != golden.len() {
        return Some(FirstDivergence {
            operator_id: operator_id.to_string(),
            layer_id,
            tensor_index: 0,
            expected: 0.0,
            actual: 0.0,
            abs_error: f32::NAN,
        });
    }
    
    // Find first index with significant error
    let threshold = 1e-6;
    for (i, (r, g)) in runtime.iter().zip(golden.iter()).enumerate() {
        let err = (*r - *g).abs();
        if err > threshold {
            return Some(FirstDivergence {
                operator_id: operator_id.to_string(),
                layer_id,
                tensor_index: i,
                expected: *g,
                actual: *r,
                abs_error: err,
            });
        }
    }
    
    None
}

pub fn divergence_report_to_yaml(div: &FirstDivergence) -> String {
    format!(
        "operator: {}\nlayer: {}\ntensor_index: {}\nexpected: {:.10}\nactual: {:.10}\nabs_error: {:.10}",
        div.operator_id,
        div.layer_id,
        div.tensor_index,
        div.expected,
        div.actual,
        div.abs_error
    )
}

#[test]
fn test_divergence_detection() {
    let runtime = vec![1.0, 2.0, 3.0, 4.0];
    let golden = vec![1.0, 2.0, 3.0, 4.1]; // Error at index 3
    
    if let Some(div) = find_first_divergence(&runtime, &golden, "test_operator", 0) {
        assert_eq!(div.tensor_index, 3);
        assert!(div.abs_error > 0.0);
        println!("First divergence: index {}, error {:.6}", div.tensor_index, div.abs_error);
    }
}