//! Hace binding for BCA (ed25519 + hasi + haceto).

#[allow(unused_imports)]
use haceto::task::spawn;

#[allow(unused_imports)]
use haed25519::Ed25519Provider;

#[allow(unused_imports)]
use hasi::TeeSigner;

/// Binding stub: adapter-only, no rr-core dependency lock.
#[allow(dead_code)]
pub struct HaceBcaBinding<E, T> {
    pub ed25519: E,
    pub tee: T,
}

impl<E, T> HaceBcaBinding<E, T>
where
    E: Ed25519Provider,
    T: TeeSigner,
{
    pub fn attest(&self, snapshot_hash: &str, merkle_root: &str) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend(snapshot_hash.as_bytes());
        payload.extend(merkle_root.as_bytes());
        self.ed25519.sign(&payload)
    }

    pub fn verify(&self, snapshot_hash: &str, merkle_root: &str, sig: &[u8]) -> bool {
        let mut payload = Vec::new();
        payload.extend(snapshot_hash.as_bytes());
        payload.extend(merkle_root.as_bytes());
        self.ed25519.verify(&payload, sig)
    }

    pub fn attest_in_tee(&self, snapshot_hash: &str, merkle_root: &str) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend(snapshot_hash.as_bytes());
        payload.extend(merkle_root.as_bytes());
        self.tee.sign(&payload)
    }
}
