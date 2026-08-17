// Directive A: Replay Engine Module

pub mod operator_replay;
pub mod layer_replay;
pub mod kv_replay;
pub mod logits_replay;

pub use operator_replay::{ReplayReport, compare_tensors, save_report};
pub use divergence_localizer::{find_first_divergence, divergence_report_to_yaml, FirstDivergence};