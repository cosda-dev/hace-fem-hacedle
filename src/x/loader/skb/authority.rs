use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct AuthorityBlock {
    pub producer: ActorIdentity,
    pub operator: Option<ActorIdentity>,
    pub rights: Vec<Right>,
    pub signatures: Vec<Signature>,
    pub trust_level: f32,
}

#[derive(Debug, Clone)]
pub struct ActorIdentity {
    pub actor_id: String,
    pub actor_type: ActorType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorType {
    Human,
    Agent,
    Authority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Right {
    Read,
    Execute,
    Extend,
}

#[derive(Debug, Clone)]
pub enum Signature {
    Ed25519([u8; 64]),
    AES([u8; 16]),
}

impl AuthorityBlock {
    pub fn verify(&self) -> bool {
        self.trust_level > 0.5
    }
}
