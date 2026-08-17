
extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::authority::guard::AuthorityMeta;
use crate::core::ops::{add::add_f32, matmul::matmul_f32, rmsnorm::rmsnorm_f32, rope::rope_f32};
use crate::runtime::executor::{GuardedOp, Op, TensorCtx};
use crate::runtime::graph_format::{GraphError, GraphView, OpCode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadError {
    Graph(GraphError),
    InvalidInputs,
    UnsupportedOp,
    ConstOutOfBounds,
}

impl From<GraphError> for LoadError {
    fn from(err: GraphError) -> Self {
        LoadError::Graph(err)
    }
}

pub struct LoadedGraph {
    pub ops: Vec<Box<dyn Op>>,
}

pub struct GraphLoader;

impl GraphLoader {
    pub fn load<'a>(graph: GraphView<'a>) -> Result<LoadedGraph, LoadError> {
        let mut ops: Vec<Box<dyn Op>> = Vec::new();
        let edges = graph.edges();

        for node in graph.nodes() {
            let start = node.input_start as usize;
            let end = start
                .checked_add(node.input_len as usize)
                .ok_or(LoadError::InvalidInputs)?;
            if end > edges.len() {
                return Err(LoadError::InvalidInputs);
            }

            let inputs = &edges[start..end];

            let op: Box<dyn Op> = match node.op_code {
                x if x == OpCode::MatMul as u16 => Box::new(MatMulOp {
                    a: inputs.get(0).ok_or(LoadError::InvalidInputs)?.tensor_id as usize,
                    b: inputs.get(1).ok_or(LoadError::InvalidInputs)?.tensor_id as usize,
                    out: node.output as usize,
                }),
                x if x == OpCode::Add as u16 => Box::new(AddOp {
                    a: inputs.get(0).ok_or(LoadError::InvalidInputs)?.tensor_id as usize,
                    b: inputs.get(1).ok_or(LoadError::InvalidInputs)?.tensor_id as usize,
                    out: node.output as usize,
                }),
                x if x == OpCode::RmsNorm as u16 => Box::new(RmsNormOp {
                    x: inputs.get(0).ok_or(LoadError::InvalidInputs)?.tensor_id as usize,
                    weight: inputs.get(1).ok_or(LoadError::InvalidInputs)?.tensor_id as usize,
                    out: node.output as usize,
                    eps: 1e-5f32,
                }),
                x if x == OpCode::Rope as u16 => Box::new(RopeOp {
                    x: inputs.get(0).ok_or(LoadError::InvalidInputs)?.tensor_id as usize,
                    cos: inputs.get(1).ok_or(LoadError::InvalidInputs)?.tensor_id as usize,
                    sin: inputs.get(2).ok_or(LoadError::InvalidInputs)?.tensor_id as usize,
                    out: node.output as usize,
                }),
                _ => return Err(LoadError::UnsupportedOp),
            };

            let authority = AuthorityMeta {
                aconx: 0,
                avaty: node.authority_offset,
                atrino: node.authority_offset,
            };

            ops.push(Box::new(GuardedOp { op, authority }));
        }

        Ok(LoadedGraph { ops })
    }
}

struct MatMulOp {
    a: usize,
    b: usize,
    out: usize,
}

impl Op for MatMulOp {
    fn execute(&self, ctx: &mut TensorCtx) {
        let ptr = ctx.tensors.as_mut_ptr();
        let a = unsafe { &*ptr.add(self.a) }.as_view();
        let b = unsafe { &*ptr.add(self.b) }.as_view();
        let out = unsafe { &mut *ptr.add(self.out) };
        let _ = matmul_f32(&a, &b, out);
    }
}

struct AddOp {
    a: usize,
    b: usize,
    out: usize,
}

impl Op for AddOp {
    fn execute(&self, ctx: &mut TensorCtx) {
        let ptr = ctx.tensors.as_mut_ptr();
        let a = unsafe { &*ptr.add(self.a) }.as_view();
        let b = unsafe { &*ptr.add(self.b) }.as_view();
        let out = unsafe { &mut *ptr.add(self.out) };
        let _ = add_f32(&a, &b, out);
    }
}

struct RmsNormOp {
    x: usize,
    weight: usize,
    out: usize,
    eps: f32,
}

impl Op for RmsNormOp {
    fn execute(&self, ctx: &mut TensorCtx) {
        let ptr = ctx.tensors.as_mut_ptr();
        let x = unsafe { &*ptr.add(self.x) }.as_view();
        let w = unsafe { &*ptr.add(self.weight) }.as_view();
        let out = unsafe { &mut *ptr.add(self.out) };
        let _ = rmsnorm_f32(&x, &w, out, self.eps);
    }
}

struct RopeOp {
    x: usize,
    cos: usize,
    sin: usize,
    out: usize,
}

impl Op for RopeOp {
    fn execute(&self, ctx: &mut TensorCtx) {
        let ptr = ctx.tensors.as_mut_ptr();
        let x = unsafe { &*ptr.add(self.x) }.as_view();
        let cos = unsafe { &*ptr.add(self.cos) }.as_view();
        let sin = unsafe { &*ptr.add(self.sin) }.as_view();
        let out = unsafe { &mut *ptr.add(self.out) };
        let _ = rope_f32(&x, &cos, &sin, out);
    }
}
