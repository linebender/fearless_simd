//! Target feature tokens for the x86 and x86-64 CPU families.
//!
//! The general compuotation CPU features associated with each [microarchitecture level] can
//! be found in their corresponding modules:
//!
//! - [`v1`] for x86-64-v1.
//! - [`v2`] for x86-64-v2.
//! - [`v3`] for x86-64-v3.
//! - [`v4`] for x86-64-v4.
//!
//! Tokens for target features which not associated with these levels can be found in this module.
//!
//! [microarchitecture level]: https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels

pub mod adx;
pub mod avx;
pub mod avx512;
pub mod crypto;
pub mod discontinued;
pub mod sse;
pub mod xsave;

pub mod v1;
pub mod v2;
pub mod v3;
pub mod v4;

pub use v1::V1;
pub use v2::V2;
pub use v3::V3;
pub use v4::V4;
