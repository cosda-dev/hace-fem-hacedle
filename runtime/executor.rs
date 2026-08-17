
extern crate alloc;

use alloc::boxed::Box;
use crate::authority::guard::AuthorityMeta;
use crate::core::tensor::TensorViewMut;

pub struct TensorCtx<'a> {
    pub tensors: &'a mut [TensorViewMut<'a, f32>],
}

pub trait Op {
    fn execute(&self, ctx: &mut TensorCtx);
}

impl<T: Op + ?Sized> Op for Box<T> {
    fn execute(&self, ctx: &mut TensorCtx) {
        (**self).execute(ctx)
    }
}

pub struct GuardedOp<O: Op> {
    pub op: O,
    pub authority: AuthorityMeta,
}

impl<O: Op> Op for GuardedOp<O> {
    fn execute(&self, ctx: &mut TensorCtx) {
        if self.authority.validate() {
            self.op.execute(ctx);
        }
    }
}

pub struct GraphExecutor<'a> {
    pub ops: &'a [Box<dyn Op>],
}

impl<'a> GraphExecutor<'a> {
    pub fn run(&self, ctx: &mut TensorCtx) {
        for op in self.ops {
            op.execute(ctx);
        }
    }
}
