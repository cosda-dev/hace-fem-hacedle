use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
    Q2_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
    F16, F32, F64, BF16,
    Unknown(u32),
}

impl QuantType {
    pub fn from_ggml_type(gt: u32) -> Self {
        match gt {
            0 => QuantType::F32,
            1 => QuantType::F64,
            2 => QuantType::F16,
            3 => QuantType::F32,
            4 => QuantType::F64,
            5 => QuantType::BF16,
            10 => QuantType::Q4_0,
            11 => QuantType::Q4_1,
            12 => QuantType::Q5_0,
            13 => QuantType::Q5_1,
            14 => QuantType::Q8_0,
            15 => QuantType::Q2_0,
            16 => QuantType::Q2_K,
            17 => QuantType::Q3_K,
            18 => QuantType::Q4_K,
            19 => QuantType::Q5_K,
            20 => QuantType::Q6_K,
            _ => QuantType::Unknown(gt),
        }
    }
    
    pub fn dequant_block_size(&self) -> usize {
        match self {
            QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q5_0 | QuantType::Q5_1 | QuantType::Q2_0 | QuantType::Q2_K | QuantType::Q3_K | QuantType::Q4_K | QuantType::Q5_K | QuantType::Q6_K => 128,
            QuantType::Q8_0 => 128,
            _ => 1,
        }
    }
    
    pub fn block_size(&self) -> usize {
        match self {
            QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q5_0 | QuantType::Q5_1 => 32,
            QuantType::Q2_0 | QuantType::Q2_K | QuantType::Q3_K => 32,
            QuantType::Q4_K | QuantType::Q5_K | QuantType::Q6_K => 32,
            QuantType::Q8_0 => 32,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    pub dimensions: Vec<i64>,
    pub offset: u64,
    pub dtype: u32,
    pub ggml_type: u32,
}

pub struct TensorProjection {
    pub tensors: Vec<TensorDescriptor>,
    pub arena_ptr: Option<usize>,
}

impl TensorProjection {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            arena_ptr: None,
        }
    }
}

impl Default for TensorProjection {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TensorHandle {
    pub name: String,
    pub tensor_idx: usize,
    pub dims: Vec<usize>,
    pub quant_type: QuantType,
    pub file_offset: u64,
    pub byte_size: u64,
}

pub struct TensorRegistry {
    name_to_handle: BTreeMap<String, TensorHandle>,
    tensors: Vec<TensorDescriptor>,
}

impl TensorRegistry {
    pub fn new() -> Self {
        Self {
            name_to_handle: BTreeMap::new(),
            tensors: Vec::new(),
        }
    }

    pub fn register(&mut self, tensor: TensorDescriptor) -> &TensorHandle {
        let idx = self.tensors.len();
        let byte_size = self.calculate_byte_size(&tensor);
        
        let handle = TensorHandle {
            name: tensor.name.clone(),
            tensor_idx: idx,
            dims: tensor.dimensions.iter().map(|&d| d as usize).collect(),
            quant_type: QuantType::from_ggml_type(tensor.ggml_type),
            file_offset: tensor.offset,
            byte_size,
        };
        let handle_ref = self.name_to_handle.entry(tensor.name.clone()).or_insert(handle.clone());
        self.tensors.push(tensor);
        handle_ref
    }
    
    fn calculate_byte_size(&self, tensor: &TensorDescriptor) -> u64 {
        let elem_count: u64 = tensor.dimensions.iter().map(|&d| d as u64).product();
        let type_size = match tensor.ggml_type {
            0 | 1 | 3 => 4,
            2 => 2,
            10 | 11 | 16 | 17 | 18 | 19 | 20 | 15 => 2,
            14 => 1,
            _ => 2,
        };
        elem_count * type_size
    }

    pub fn get(&self, name: &str) -> Option<&TensorHandle> {
        self.name_to_handle.get(name)
    }

    pub fn get_by_idx(&self, idx: usize) -> Option<&TensorDescriptor> {
        self.tensors.get(idx)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl Default for TensorRegistry {
    fn default() -> Self {
        Self::new()
    }
}