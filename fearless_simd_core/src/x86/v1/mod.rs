//! Target features enabled in the `x86-64-v1` [microarchitecture level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) on x86 and x86-64.
//!
//! This can usually be treated as the baseline for x86-64 support; all of the target features in this module are enabled by
//! default on Rust's x86-64 targets (such as `x86_64-unknown-linux-gnu`).
//!
//! This module also contains [`V1`], which is a token indicating that this level is available.
//! All tokens in this module can be created [`From`] that token.
//! This is re-exported in the parent module, and in most cases that shorter path should be preferred.

pub use crate::x86::sse::Fxsr;
pub use crate::x86::sse::Sse;
pub use crate::x86::sse::Sse2;

mod level;
pub use level::V1;
