#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod x;
pub mod resolver;
pub mod quant_view;
pub mod ops;

#[cfg(feature = "alloc")]
pub mod alloc_exports {
    pub use alloc::vec::Vec;
    pub use alloc::string::String;
    pub use alloc::collections::BTreeMap;
    pub use alloc::collections::BTreeSet;
}

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
pub mod brain_kernel;

#[cfg(feature = "std")]
pub use brain_kernel::HacedleBrain;