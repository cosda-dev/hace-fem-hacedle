use crate::alloc_exports::*;

/// Matrix-vector multiplication (input × weight^T)
pub fn matmul_vec_matrix(vec: &[f32], mat: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut result = vec![0.0_f32; out_dim];
    
    for i in 0..out_dim {
        let row_start = i * in_dim;
        let mut sum = 0.0_f32;
        
        for j in 0..in_dim {
            if row_start + j < mat.len() && j < vec.len() {
                sum += vec[j] * mat[row_start + j];
            }
        }
        
        result[i] = sum;
    }
    
    result
}

/// Q-projection (transpose)
pub fn q_proj(input: &[f32], weight: &[f32], hidden_size: usize) -> Vec<f32> {
    matmul_vec_matrix(input, weight, hidden_size, hidden_size)
}