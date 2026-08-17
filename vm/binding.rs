use crate::{Arena, RegFile};
use crate::ops::{exec_q_acc, exec_q_add, exec_q_dot, exec_q_mul};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HacedleOpcode {
    QAdd = 0x10,
    QMul = 0x11,
    QDot = 0x12,
    QAcc = 0x13,
}

pub struct HacedleBinding<'a> {
    pub regs: RegFile<'a>,
    pub arena: &'a mut Arena,
    pub tensor_a: &'a [i8],
    pub tensor_b: &'a [i8],
}

pub fn exec_hacedle(opcode: u8, binding: &mut HacedleBinding<'_>) {
    match opcode {
        x if x == HacedleOpcode::QAdd as u8 => {
            exec_q_add(&mut *binding.regs.regs, 1, 2, 0);
        }
        x if x == HacedleOpcode::QMul as u8 => {
            exec_q_mul(&mut *binding.regs.regs, 1, 2, 0);
        }
        x if x == HacedleOpcode::QDot as u8 => {
            let acc = exec_q_dot(binding.tensor_a, binding.tensor_b);
            binding.regs.regs[0] = acc as i16;
        }
        x if x == HacedleOpcode::QAcc as u8 => {
            exec_q_acc(&mut *binding.regs.regs);
        }
        _ => {}
    }

    let _ = binding.arena.alloc_bytes(0);
}
