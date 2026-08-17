
use core::marker::PhantomData;

pub const MAX_DIMS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    InvalidShape,
    OutOfBounds,
    Overflow,
}

pub struct TensorView<'a, T> {
    pub data: &'a [T],
    pub shape: [usize; MAX_DIMS],
    pub strides: [usize; MAX_DIMS],
    pub rank: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> Copy for TensorView<'a, T> {}
impl<'a, T> Clone for TensorView<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> TensorView<'a, T> {
    pub fn new(data: &'a [T], shape: &[usize], strides: &[usize]) -> Result<Self, TensorError> {
        let rank = shape.len();
        if rank == 0 || rank > MAX_DIMS || strides.len() != rank {
            return Err(TensorError::InvalidShape);
        }

        let mut s = [0usize; MAX_DIMS];
        let mut st = [0usize; MAX_DIMS];
        for i in 0..rank {
            if shape[i] == 0 {
                return Err(TensorError::InvalidShape);
            }
            s[i] = shape[i];
            st[i] = strides[i];
        }

        let required = required_len(rank, &s, &st)?;
        if required > data.len() {
            return Err(TensorError::OutOfBounds);
        }

        Ok(Self {
            data,
            shape: s,
            strides: st,
            rank,
            _marker: PhantomData,
        })
    }

    pub fn from_contiguous(data: &'a [T], shape: &[usize]) -> Result<Self, TensorError> {
        let rank = shape.len();
        if rank == 0 || rank > MAX_DIMS {
            return Err(TensorError::InvalidShape);
        }

        let mut strides = [0usize; MAX_DIMS];
        let mut stride = 1usize;
        for i in (0..rank).rev() {
            strides[i] = stride;
            stride = stride.checked_mul(shape[i]).ok_or(TensorError::Overflow)?;
        }

        Self::new(data, shape, &strides[..rank])
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.rank]
    }

    pub fn numel(&self) -> usize {
        let mut n = 1usize;
        for i in 0..self.rank {
            n = n.saturating_mul(self.shape[i]);
        }
        n
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1usize;
        for i in (0..self.rank).rev() {
            if self.strides[i] != expected {
                return false;
            }
            match expected.checked_mul(self.shape[i]) {
                Some(next) => expected = next,
                None => return false,
            }
        }
        true
    }

    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: &[usize]) -> &T {
        let mut offset = 0usize;
        for i in 0..self.rank {
            offset += idx[i] * self.strides[i];
        }
        self.data.get_unchecked(offset)
    }

    #[inline(always)]
    pub unsafe fn get_flat_unchecked(&self, idx: usize) -> &T {
        self.data.get_unchecked(idx)
    }
}

pub struct TensorViewMut<'a, T> {
    pub data: &'a mut [T],
    pub shape: [usize; MAX_DIMS],
    pub strides: [usize; MAX_DIMS],
    pub rank: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> TensorViewMut<'a, T> {
    pub fn new(data: &'a mut [T], shape: &[usize], strides: &[usize]) -> Result<Self, TensorError> {
        let rank = shape.len();
        if rank == 0 || rank > MAX_DIMS || strides.len() != rank {
            return Err(TensorError::InvalidShape);
        }

        let mut s = [0usize; MAX_DIMS];
        let mut st = [0usize; MAX_DIMS];
        for i in 0..rank {
            if shape[i] == 0 {
                return Err(TensorError::InvalidShape);
            }
            s[i] = shape[i];
            st[i] = strides[i];
        }

        let required = required_len(rank, &s, &st)?;
        if required > data.len() {
            return Err(TensorError::OutOfBounds);
        }

        Ok(Self {
            data,
            shape: s,
            strides: st,
            rank,
            _marker: PhantomData,
        })
    }

    pub fn from_contiguous(data: &'a mut [T], shape: &[usize]) -> Result<Self, TensorError> {
        let rank = shape.len();
        if rank == 0 || rank > MAX_DIMS {
            return Err(TensorError::InvalidShape);
        }

        let mut strides = [0usize; MAX_DIMS];
        let mut stride = 1usize;
        for i in (0..rank).rev() {
            strides[i] = stride;
            stride = stride.checked_mul(shape[i]).ok_or(TensorError::Overflow)?;
        }

        Self::new(data, shape, &strides[..rank])
    }

    pub fn as_view(&self) -> TensorView<'_, T> {
        TensorView {
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            rank: self.rank,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.rank]
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, idx: &[usize]) -> &mut T {
        let mut offset = 0usize;
        for i in 0..self.rank {
            offset += idx[i] * self.strides[i];
        }
        self.data.get_unchecked_mut(offset)
    }

    #[inline(always)]
    pub unsafe fn get_flat_unchecked_mut(&mut self, idx: usize) -> &mut T {
        self.data.get_unchecked_mut(idx)
    }
}

fn required_len(
    rank: usize,
    shape: &[usize; MAX_DIMS],
    strides: &[usize; MAX_DIMS],
) -> Result<usize, TensorError> {
    if rank == 0 {
        return Ok(0);
    }

    let mut last_index = 0usize;
    for i in 0..rank {
        let span = shape[i]
            .checked_sub(1)
            .ok_or(TensorError::Overflow)?
            .checked_mul(strides[i])
            .ok_or(TensorError::Overflow)?;
        last_index = last_index.checked_add(span).ok_or(TensorError::Overflow)?;
    }

    last_index
        .checked_add(1)
        .ok_or(TensorError::Overflow)
}
