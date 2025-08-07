// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A helper library to make SIMD more friendly.
//!
//! # Feature Flags
//!
//! The following crate [feature flags](https://doc.rust-lang.org/cargo/reference/features.html#dependency-features) are available:
//!
//! - `std` (enabled by default): Get floating point functions from the standard library (likely using your targets libc).
//! - `libm`: Use floating point implementations from [libm].
//! - `safe_wrappers`: Include safe wrappers for (some) target feature specific intrinsics,
//!   beyond the basic SIMD operations abstracted on all platforms.
//! - `half`: Use `f16` (16 bit floating point) support from the [half] crate.
//!   If this feature isn't enabled, a minimal subset copied (under license) from that same crate is used.
//!   Only supported on aarch64, as other supported architectures don't have hardware support for these types.
//!   This feature is only useful if the `safe_wrappers` feature is enabled, to use the `core_arch::aarch64::Fp16` type.
//!
//! At least one of `std` and `libm` is required; `std` overrides `libm`.
#![cfg_attr(
    not(all(target_arch = "aarch64", feature = "half")),
    doc = "\n\n[half]: https://docs.rs/half/latest/half/"
)]
// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![allow(non_camel_case_types, reason = "TODO")]
#![expect(clippy::unused_unit, reason = "easier for code generation")]
#![expect(
    clippy::new_without_default,
    missing_docs,
    clippy::use_self,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

#[cfg(all(not(feature = "libm"), not(feature = "std")))]
compile_error!("fearless_simd requires either the `std` or `libm` feature");

// Suppress the unused_crate_dependencies lint when both std and libm are specified.
#[cfg(all(feature = "std", feature = "libm"))]
use libm as _;

pub mod core_arch;
mod impl_macros;

mod generated;
mod macros;
mod traits;

pub use generated::*;
pub use traits::*;

// For now, only bring in f16 on aarch64. We can also bring it in
// on x86_64, but only Sapphire Rapids supports it.

#[cfg(all(target_arch = "aarch64", feature = "half"))]
pub type f16 = half::f16;
#[cfg(all(target_arch = "aarch64", not(feature = "half")))]
mod half_assed;
#[cfg(all(target_arch = "aarch64", not(feature = "half")))]
pub use half_assed::f16;

#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub mod aarch64 {
    pub use crate::generated::Neon;
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm32 {
    pub use crate::generated::WasmSimd128;
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod x86 {
    pub use crate::generated::Sse4_2;
}

/// The level enum with the specific SIMD capabilities available.
#[derive(Clone, Copy, Debug)]
pub enum Level {
    Fallback(Fallback),
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    Neon(Neon),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    WasmSimd128(WasmSimd128),
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    Sse4_2(Sse4_2),
}

impl Level {
    pub fn new() -> Self {
        #[cfg(all(feature = "std", target_arch = "aarch64"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { Level::Neon(Neon::new_unchecked()) };
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return Level::WasmSimd128(WasmSimd128::new_unchecked());
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::arch::is_x86_feature_detected!("sse4.2") {
            return unsafe { Level::Sse4_2(Sse4_2::new_unchecked()) };
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        Self::fallback()
    }

    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    #[inline]
    pub fn as_neon(self) -> Option<Neon> {
        match self {
            Level::Neon(neon) => Some(neon),
            _ => None,
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline]
    pub fn as_wasm_simd128(self) -> Option<WasmSimd128> {
        match self {
            Level::WasmSimd128(simd128) => Some(simd128),
            _ => None,
        }
    }
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline]
    pub fn as_sse4_2(self) -> Option<Sse4_2> {
        match self {
            Level::Sse4_2(sse42) => Some(sse42),
            _ => None,
        }
    }

    #[inline]
    pub fn fallback() -> Self {
        Self::Fallback(Fallback::new())
    }

    #[inline]
    pub fn dispatch<W: WithSimd>(self, f: W) -> W::Output {
        #[cfg(all(feature = "std", target_arch = "aarch64"))]
        #[target_feature(enable = "neon")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn dispatch_neon<W: WithSimd>(f: W, neon: Neon) -> W::Output {
            f.with_simd(neon)
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        #[inline]
        fn dispatch_simd128<W: WithSimd>(f: W, simd128: WasmSimd128) -> W::Output {
            f.with_simd(simd128)
        }

        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        #[target_feature(enable = "sse4.2")]
        #[inline]
        unsafe fn dispatch_sse4_2<W: WithSimd>(f: W, sse4_2: Sse4_2) -> W::Output {
            f.with_simd(sse4_2)
        }

        #[inline]
        fn dispatch_fallback<W: WithSimd>(f: W, fallback: Fallback) -> W::Output {
            f.with_simd(fallback)
        }

        match self {
            #[cfg(all(feature = "std", target_arch = "aarch64"))]
            Level::Neon(neon) => unsafe { dispatch_neon(f, neon) },
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Level::WasmSimd128(simd128) => dispatch_simd128(f, simd128),
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            Level::Sse4_2(sse4_2) => unsafe { dispatch_sse4_2(f, sse4_2) },
            Level::Fallback(fallback) => dispatch_fallback(f, fallback),
        }
    }
}
