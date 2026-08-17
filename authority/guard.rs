
#[derive(Clone, Copy, Debug)]
pub struct AuthorityMeta {
    pub aconx: u64,
    pub avaty: u32,
    pub atrino: u32,
}

impl AuthorityMeta {
    pub fn validate(&self) -> bool {
        // TODO: integrate real authority checks.
        let _ = (self.aconx, self.avaty, self.atrino);
        true
    }
}
