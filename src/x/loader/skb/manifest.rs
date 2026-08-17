use alloc::string::String;
use alloc::vec::Vec;
use alloc::vec;

#[derive(Debug, Clone)]
pub struct SkbManifest {
    pub title: String,
    pub category: String,
    pub language: String,
    pub domain: String,
    pub dependencies: Vec<Dependency>,
    pub required_runtime: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Dependency {
    pub uri: String,
    pub version: String,
}

impl Default for SkbManifest {
    fn default() -> Self {
        Self {
            title: String::new(),
            category: String::from("knowledge"),
            language: String::from("vi-85"),
            domain: String::new(),
            dependencies: Vec::new(),
            required_runtime: vec![String::from("hacedle>=5.0")],
        }
    }
}
