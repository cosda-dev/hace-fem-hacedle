use alloc::string::String;
use alloc::vec::Vec;

pub struct SemanticGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub properties: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: String,
    pub to: String,
 pub relation: Relation,
}

#[derive(Debug, Clone)]
pub enum Relation {
    DependsOn,
    Extends,
    Constrains,
    DerivesFrom,
}

impl SemanticGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn find_node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn follow_edges(&self, from: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.from == from).collect()
    }
}

impl Default for SemanticGraph {
    fn default() -> Self {
        Self::new()
    }
}
