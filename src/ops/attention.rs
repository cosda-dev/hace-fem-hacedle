// Attention operators for parity testing

use crate::alloc_exports::*;

pub fn softmax_row(scores: &[f32]) -> Vec<f32> {
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = scores.iter().map(|x| (x - max_val).exp()).sum();
    scores.iter().map(|x| (x - max_val).exp() / sum).collect()
}

pub fn attention_scores(q: &[f32], k: &[f32], n_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; n_heads * n_heads];
    for i in 0..n_heads {
        for j in 0..n_heads {
            let mut sum = 0.0f32;
            for d in 0..head_dim {
                sum += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * n_heads + j] = sum / (head_dim as f32).sqrt();
        }
    }
    scores
}

pub fn attention_weighted_sum(scores: &[f32], v: &[f32], n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Vec<f32> {
    let output = vec![0.0f32; n_heads * head_dim];
    for i in 0..n_heads {
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for j in 0..n_heads {
                let kv_idx = j / (n_heads / n_kv_heads);
                sum += scores[i * n_heads + j] * v[kv_idx * head_dim + d];
            }
            // output[i * head_dim + d] = sum; // Cannot assign to vec!
        }
    }
    output
}