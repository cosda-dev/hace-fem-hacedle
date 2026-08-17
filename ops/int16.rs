use crate::core::TensorI16;

pub fn add_i16(a: &TensorI16, b: &TensorI16) -> TensorI16 {
    assert_eq!(a.len, b.len);
    let mut out = a.data.clone();
    let mut i = 0usize;
    while i < a.len {
        out[i] = out[i].wrapping_add(b.data[i]);
        i += 1;
    }
    TensorI16 { data: out, len: a.len }
}

pub fn mul_i16(a: &TensorI16, b: &TensorI16) -> TensorI16 {
    assert_eq!(a.len, b.len);
    let mut out = a.data.clone();
    let mut i = 0usize;
    while i < a.len {
        out[i] = out[i].wrapping_mul(b.data[i]);
        i += 1;
    }
    TensorI16 { data: out, len: a.len }
}

pub fn accumulate_i16(acc: &mut [i16], values: &[i16]) {
    let len = acc.len().min(values.len());
    let mut i = 0usize;
    while i < len {
        acc[i] = acc[i].wrapping_add(values[i]);
        i += 1;
    }
}
