#![cfg_attr(not(any(test, feature = "std")), no_std)]

extern crate alloc;

pub mod authority;
pub mod core;
pub mod rr_bind;
pub mod runtime;
pub mod x;

#[cfg(any(test, feature = "std"))]
extern crate std;
