use crate::ops::simd::dot_i8_block;

pub fn dot_batch(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len().min(b.len());
    let mut acc = 0i32;
    let step = 32usize;

    let mut i = 0usize;
    while i + step <= len {
        let ap = unsafe { a.as_ptr().add(i) };
        let bp = unsafe { b.as_ptr().add(i) };
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        {
            acc += unsafe { dot_i8_block(ap, bp) };
        }
        #[cfg(not(all(feature = "simd", target_feature = "avx2")))]
        {
            acc += dot_i8_block(ap, bp);
        }
        i += step;
    }

    while i < len {
        acc += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }

    acc
}
