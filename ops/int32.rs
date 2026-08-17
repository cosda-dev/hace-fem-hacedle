pub fn accumulate_i32(acc: &mut [i32], values: &[i32]) {
    let len = acc.len().min(values.len());
    let mut i = 0usize;
    while i < len {
        acc[i] = acc[i].wrapping_add(values[i]);
        i += 1;
    }
}
