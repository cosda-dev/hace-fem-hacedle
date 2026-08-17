#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub unsafe fn dot_i8_block(a: *const i8, b: *const i8) -> i32 {
    use core::arch::x86_64::*;

    let va = _mm256_loadu_si256(a as *const _);
    let vb = _mm256_loadu_si256(b as *const _);
    let madd = _mm256_maddubs_epi16(va, vb);
    let madd2 = _mm256_madd_epi16(madd, _mm256_set1_epi16(1));

    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, madd2);

    let mut acc = 0i32;
    for v in tmp {
        acc += v;
    }
    acc
}

#[cfg(not(all(feature = "simd", target_feature = "avx2")))]
pub fn dot_i8_block(a: *const i8, b: *const i8) -> i32 {
    let mut acc = 0i32;
    let mut i = 0usize;
    unsafe {
        while i < 32 {
            acc += (*a.add(i) as i32) * (*b.add(i) as i32);
            i += 1;
        }
    }
    acc
}
