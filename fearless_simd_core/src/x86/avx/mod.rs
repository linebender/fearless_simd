// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Target features related to the Advanced Vector Extensions target features (before AVX-512).
//!
//! These are most commonly used through the [x86-64-v3](crate::x86::V3) microarchitecture level.
//!
//! These support SIMD registers of up to 256 bits.
//! For the 512 bit extension, see [`avx512`](crate::x86::avx512).

#[expect(
    clippy::module_inception,
    reason = "The inner module is automatically generated."
)]
mod avx;
pub use avx::Avx;

mod avx2;
pub use avx2::Avx2;

mod avxifma;
pub use avxifma::Avxifma;

mod avxneconvert;
pub use avxneconvert::Avxneconvert;

mod avxvnni;
pub use avxvnni::Avxvnni;

mod avxvnniint8;
pub use avxvnniint8::Avxvnniint8;

mod avxvnniint16;
pub use avxvnniint16::Avxvnniint16;
