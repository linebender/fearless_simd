//! Target features enabled in the `x86-64-v1` [microarchitecture level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) on x86 and x86-64.
//!
//! This is the baseline for x86-64 support.

pub use crate::x86::sse::Fxsr;
pub use crate::x86::sse::Sse;
pub use crate::x86::sse::Sse2;

mod level;
pub use level::V1;
