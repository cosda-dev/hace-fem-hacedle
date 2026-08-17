use alloc::string::String;
use alloc::vec::Vec;

pub struct HacedleContext {
    pub session_id: String,
    pub kv_cache: Vec<u8>,
}