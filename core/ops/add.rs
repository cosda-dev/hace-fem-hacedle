
use crate::core::tensor::{TensorError, TensorView, TensorViewMut};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddError {
    ShapeMismatch,
    Tensor(TensorError),
}

impl From<TensorError> for AddError {
    fn from(err: TensorError) -> Self {
        AddError::Tensor(err)
    }
}

pub fn add_f32(
    a: &TensorView<'_, f32>,
    b: &TensorView<'_, f32>,
    out: &mut TensorViewMut<'_, f32>,
) -> Result<(), AddError> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let o_shape = out.shape();

    if a_shape != b_shape || a_shape != o_shape {
        return Err(AddError::ShapeMismatch);
    }

    if !a.is_contiguous() || !b.is_contiguous() || !out.as_view().is_contiguous() {
        return Err(AddError::ShapeMismatch);
    }

    let len = a.numel();
    for i in 0..len {
        let a_val = unsafe { *a.get_flat_unchecked(i) };
        let b_val = unsafe { *b.get_flat_unchecked(i) };
        unsafe {
            *out.get_flat_unchecked_mut(i) = a_val + b_val;
        }
    }

    Ok(())
}
