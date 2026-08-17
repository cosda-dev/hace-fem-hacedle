use alloc::string::String;
use alloc::vec::Vec;

pub struct KnowledgeLayer {
    pub concepts: Vec<Concept>,
    pub facts: Vec<Fact>,
    pub ontology: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Concept {
    pub id: String,
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct Fact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

impl KnowledgeLayer {
    pub fn new() -> Self {
        Self {
            concepts: Vec::new(),
            facts: Vec::new(),
            ontology: Vec::new(),
        }
    }
}

impl Default for KnowledgeLayer {
    fn default() -> Self {
        Self::new()
    }
}
