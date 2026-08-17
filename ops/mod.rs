mod int8;
mod int16;
mod int32;
mod simd;
mod batch;
mod q_ops;
mod fused;

pub use int8::{add_i8, dot_i8, dot_i8_simd};
pub use int16::{add_i16, mul_i16, accumulate_i16};
pub use int32::accumulate_i32;
pub use batch::dot_batch;
pub use q_ops::{exec_q_acc, exec_q_add, exec_q_dot, exec_q_mul};
pub use fused::fused_qdot_qadd_qacc_i8;
