use crate::core::TensorI8;

pub fn dot_i8(a: &TensorI8, b: &TensorI8) -> i16 {
    assert_eq!(a.len, b.len);
    let mut acc: i32 = 0;
    let mut i = 0usize;
    while i < a.len {
        acc += (a.data[i] as i32) * (b.data[i] as i32);
        i += 1;
    }
    acc as i16
}

pub fn add_i8(a: &TensorI8, b: &TensorI8) -> TensorI8 {
    assert_eq!(a.len, b.len);
    let mut out = a.data.clone();
    let mut i = 0usize;
    while i < a.len {
        out[i] = out[i].wrapping_add(b.data[i]);
        i += 1;
    }
    TensorI8 { data: out, len: a.len }
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub unsafe fn dot_i8_simd(a: &TensorI8, b: &TensorI8) -> i32 {
    use core::arch::x86_64::*;
    assert_eq!(a.len, b.len);

    let mut sum = _mm256_setzero_si256();
    let chunks = a.len / 32;

    let mut i = 0usize;
    while i < chunks {
        let va = _mm256_loadu_si256(a.data.as_ptr().add(i * 32) as *const _);
        let vb = _mm256_loadu_si256(b.data.as_ptr().add(i * 32) as *const _);
        let madd = _mm256_maddubs_epi16(va, vb);
        let madd2 = _mm256_madd_epi16(madd, _mm256_set1_epi16(1));
        sum = _mm256_add_epi32(sum, madd2);
        i += 1;
    }

    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, sum);
    let mut acc: i32 = 0;
    for v in tmp {
        acc += v;
    }

    let mut j = chunks * 32;
    while j < a.len {
        acc += (a.data[j] as i32) * (b.data[j] as i32);
        j += 1;
    }

    acc
}

#[cfg(not(all(feature = "simd", target_feature = "avx2")))]
pub fn dot_i8_simd(a: &TensorI8, b: &TensorI8) -> i32 {
    dot_i8(a, b) as i32
}
