// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Target features enabled in the `x86-64-v3` [microarchitecture level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) on x86 and x86-64.
//!
//! This module also contains [`V3`], which is a token indicating that this level is available.
//! All tokens in this module can be created [`From`] that token.
//! This is re-exported in the parent module, and in most cases that shorter path should be preferred.
//!
//! This level also implies that `x86-64-v2` is available.

pub use crate::x86::v1::Fxsr;
pub use crate::x86::v1::Sse;
pub use crate::x86::v1::Sse2;
pub use crate::x86::v2::Cmpxchg16b;
pub use crate::x86::v2::Popcnt;
pub use crate::x86::v2::Sse3;
pub use crate::x86::v2::Sse4_1;
pub use crate::x86::v2::Sse4_2;
pub use crate::x86::v2::SupplementalSse3;

pub use crate::x86::avx::Avx;
pub use crate::x86::avx::Avx2;
pub use crate::x86::xsave::Xsave;

mod bmi1;
pub use bmi1::Bmi1;

mod bmi2;
pub use bmi2::Bmi2;

mod f16c;
pub use f16c::F16c;

mod fma;
pub use fma::Fma;

mod lzcnt;
pub use lzcnt::Lzcnt;

mod movbe;
pub use movbe::Movbe;

mod level;
pub use level::V3;
