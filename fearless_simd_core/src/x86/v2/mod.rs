//! Target features enabled in the `x86-64-v2` [microarchitecture level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) on x86 and x86-64.
//!
//! This module also contains [`V2`], which is a token indicating that this level is available.
//! All tokens in this module can be created [`From`] that token.
//! This is re-exported in the parent module, and in most cases that shorter path should be preferred.
//!
//! This level also implies that `x86-64-v1` is available.

pub use crate::x86::sse::Sse3;
pub use crate::x86::sse::Sse4_1;
pub use crate::x86::sse::Sse4_2;
pub use crate::x86::sse::SupplementalSse3;
pub use crate::x86::v1::Fxsr;
pub use crate::x86::v1::Sse;
pub use crate::x86::v1::Sse2;

mod cmpxchg16b;
pub use cmpxchg16b::Cmpxchg16b;

mod popcnt;
pub use popcnt::Popcnt;

mod level;
pub use level::V2;
