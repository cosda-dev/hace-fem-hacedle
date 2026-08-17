use crate::ops::{accumulate_i16, dot_batch};

pub fn exec_q_add(regs: &mut [i16; 16], a: usize, b: usize, out: usize) {
    regs[out] = regs[a].wrapping_add(regs[b]);
}

pub fn exec_q_mul(regs: &mut [i16; 16], a: usize, b: usize, out: usize) {
    regs[out] = regs[a].wrapping_mul(regs[b]);
}

pub fn exec_q_dot(a: &[i8], b: &[i8]) -> i32 {
    dot_batch(a, b)
}

pub fn exec_q_acc(regs: &mut [i16; 16]) {
    let mut tmp = *regs;
    accumulate_i16(&mut tmp, regs);
    *regs = tmp;
}
