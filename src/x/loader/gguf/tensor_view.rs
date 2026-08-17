use crate::alloc_exports::*;
use super::QuantType;

pub struct TensorView<'a> {
    pub data: &'a [u8],
    pub shape: Vec<usize>,
    pub quant_type: QuantType,
}

impl<'a> TensorView<'a> {
    pub fn new(data: &'a [u8], shape: Vec<usize>, quant_type: QuantType) -> Self {
        Self { data, shape, quant_type }
    }
    
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }
    
    pub fn byte_len(&self) -> usize {
        self.elem_count() * self.quant_type.block_size()
    }
}

pub struct TensorReader<'a> {
    mmap: &'a [u8],
}

impl<'a> TensorReader<'a> {
    pub fn new(mmap: &'a [u8]) -> Self {
        Self { mmap }
    }
    
    pub fn view(&self, offset: u64, shape: &[u64], quant_type: QuantType) -> Option<TensorView<'static>> {
        let offset = offset as usize;
        let byte_len = shape.iter().map(|&s| s as usize).product::<usize>() * quant_type.block_size();
        
        if offset + byte_len > self.mmap.len() {
            return None;
        }
        
        let data = &self.mmap[offset..offset + byte_len];
        let shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        
        Some(TensorView::new(data, shape, quant_type))
    }
}