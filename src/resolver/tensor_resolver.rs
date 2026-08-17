
// Tensor Resolver - Map GGUF tensor names to handles
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::format;

pub struct TensorResolver {
    tensor_map: BTreeMap<String, TensorDescriptor>,
}

#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub quant_type: u32,
    pub offset: u64,
    pub data_len: usize,
}

impl TensorResolver {
    pub fn new() -> Self {
        Self {
            tensor_map: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, desc: TensorDescriptor) {
        self.tensor_map.insert(desc.name.clone(), desc);
    }

    pub fn get(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensor_map.get(name)
    }

    pub fn resolve_attn_q(&self, layer: usize) -> Option<&TensorDescriptor> {
        self.get(&format!("blk.{}.attn_q.weight", layer))
    }

    pub fn resolve_attn_k(&self, layer: usize) -> Option<&TensorDescriptor> {
        self.get(&format!("blk.{}.attn_k.weight", layer))
    }

    pub fn resolve_attn_v(&self, layer: usize) -> Option<&TensorDescriptor> {
        self.get(&format!("blk.{}.attn_v.weight", layer))
    }

    pub fn resolve_attn_o(&self, layer: usize) -> Option<&TensorDescriptor> {
        self.get(&format!("blk.{}.attn_output.weight", layer))
    }

    pub fn resolve_ffn_gate(&self, layer: usize) -> Option<&TensorDescriptor> {
        self.get(&format!("blk.{}.ffn_gate.weight", layer))
    }

    pub fn resolve_ffn_up(&self, layer: usize) -> Option<&TensorDescriptor> {
        self.get(&format!("blk.{}.ffn_up.weight", layer))
    }

    pub fn resolve_ffn_down(&self, layer: usize) -> Option<&TensorDescriptor> {
        self.get(&format!("blk.{}.ffn_down.weight", layer))
    }
}

