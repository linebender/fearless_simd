// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Target feature tokens for the x86 and x86-64 CPU families.
//!
//! The general computation [microarchitecture level]s each have a level in this module.
//! These levels are useful for most users of this crate, as they provide useful categories
//! of supported instructions.
//!
//! - [`V1`] for x86-64-v1.
//! - [`V2`] for x86-64-v2.
//! - [`V3`] for x86-64-v3.
//! - [`V4`] for x86-64-v4.
//!
//! We don't yet provide a way to select the best of these for the current CPU,
//! but that is planned.
//!
//! Tokens for individual target features, including those not associated with these levels,
//! can be found in the modules under this feature.
//! These are less likely to be directly useful for most users, but are provided for use
//! cases which require them (probably especially those under [`crypto`]).
//!
//! Both the x86 and x86-64 CPU families are supported in this module as their code is entirely identical,
//! including using the same [`std::is_x86_feature_detected`] macro.
//! Note that this is not the case for `std::arch`; for example, [`core::arch::x86_64::_mm_crc32_u64`] is
//! only available on x86-64.
//!
//! Documentation for features is adapted from the Rust reference.
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
