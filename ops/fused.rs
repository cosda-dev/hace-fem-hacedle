use crate::ops::dot_batch;

#[cfg(all(feature = "std", feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut sum = _mm256_setzero_si256();
    let chunks = len / 32;

    let mut i = 0usize;
    while i < chunks {
        let va = _mm256_loadu_si256(a.as_ptr().add(i * 32) as *const _);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i * 32) as *const _);
        let madd16 = _mm256_maddubs_epi16(va, vb);
        let madd32 = _mm256_madd_epi16(madd16, _mm256_set1_epi16(1));
        sum = _mm256_add_epi32(sum, madd32);
        i += 1;
    }

    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, sum);
    let mut acc = tmp.iter().copied().sum::<i32>();

    let mut j = chunks * 32;
    while j < len {
        acc = acc.wrapping_add((a[j] as i32).wrapping_mul(b[j] as i32));
        j += 1;
    }

    acc
}

#[inline]
fn dot_runtime(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(all(feature = "std", feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { dot_i8_avx2(a, b) };
        }
    }
    dot_batch(a, b)
}

/// Fused deterministic path for Q_DOT + Q_ADD + Q_ACC in i32 domain.
///
/// - `bias` models Q_ADD operand
/// - `carry` models Q_ACC rolling state
/// - result is bit-stable for same input and feature set
pub fn fused_qdot_qadd_qacc_i8(a: &[i8], b: &[i8], bias: i32, carry: &mut i32) -> i32 {
    let dot = dot_runtime(a, b);
    let out = dot.wrapping_add(bias).wrapping_add(*carry);
    *carry = out;
    out
}
