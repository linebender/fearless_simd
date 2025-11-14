// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Target features enabled in the `x86-64-v4` [microarchitecture level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) on x86 and x86-64.
//!
//! This module also contains [`V4`], which is a token indicating that this level is available.
//! All tokens in this module can be created [`From`] that token.
//! This is re-exported in the parent module, and in most cases that shorter path should be preferred.
//!
//! This level also implies that `x86-64-v3` is available.

pub use crate::x86::v1::Fxsr;
pub use crate::x86::v1::Sse;
pub use crate::x86::v1::Sse2;
pub use crate::x86::v2::Cmpxchg16b;
pub use crate::x86::v2::Popcnt;
pub use crate::x86::v2::Sse3;
pub use crate::x86::v2::Sse4_1;
pub use crate::x86::v2::Sse4_2;
pub use crate::x86::v2::SupplementalSse3;
pub use crate::x86::v3::Avx;
pub use crate::x86::v3::Avx2;
pub use crate::x86::v3::Bmi1;
pub use crate::x86::v3::Bmi2;
pub use crate::x86::v3::F16c;
pub use crate::x86::v3::Fma;
pub use crate::x86::v3::Lzcnt;
pub use crate::x86::v3::Movbe;
pub use crate::x86::v3::Xsave;

pub use crate::x86::avx512::Avx512bw;
pub use crate::x86::avx512::Avx512cd;
pub use crate::x86::avx512::Avx512dq;
pub use crate::x86::avx512::Avx512f;
pub use crate::x86::avx512::Avx512vl;

mod level;
pub use level::V4;
