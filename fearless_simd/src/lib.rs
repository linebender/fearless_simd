// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A helper library to make SIMD more friendly.
//!
//! Fearless SIMD exposes safe SIMD with ergonomic multi-versioning in Rust.
//!
//! Fearless SIMD uses "marker values" which serve as proofs of which target features are available on the current CPU.
//! These each implement the [`Simd`] trait, which exposes a core set of SIMD operations which are implemented as
//! efficiently as possible on each target platform.
//!
//! Additionally, there are types for packed vectors of a specific width and element type (such as [`f32x4`]).
//! Fearless SIMD does not currently support vectors of less than 128 bits.
//! These vector types implement some standard arithmetic traits (i.e. they can be added together using
//! `+`, multiplied by a scalar using `*`, among others), which are implemented as efficiently
//! as possible using SIMD instructions.
//! These can be created in a SIMD context using the [`SimdFrom`] trait, or the
//! [`from_slice`][SimdBase::from_slice] associated function.
//!
//! To call a function with the best available target features and get the associated `Simd`
//! implementation, use the [`dispatch!()`] macro:
//!
//! ```rust
//! use fearless_simd::{Level, Simd, dispatch};
//!
//! #[inline(always)]
//! fn sigmoid<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) { /* ... */ }
//!
//! // The stored level, which you should only construct once in your application.
//! let level = Level::new();
//!
//! dispatch!(level, simd => sigmoid(simd, &[/*...*/], &mut [/*...*/]));
//! ```
//!
//! A few things to note:
//!
//! 1) `sigmoid` is generic over any `Simd` type.
//! 2) The [`dispatch`] macro is used to invoke the given function with the target features associated with the supplied [`Level`].
//! 3) The function or closure passed to [`dispatch!()`] should be `#[inline(always)]`.
//!    The performance of the SIMD implementation may be poor if that isn't the case. See [the section on inlining for details](#inlining)
//!
//! The first parameter to [`dispatch!()`] is the [`Level`].
//! If you are writing an application, you should create this once (using [`Level::new`]), and pass it to any function which wants to use SIMD.
//! This type stores which instruction sets are available for the current process, which is used
//! in the macro to dispatch to the most optimal variant of the supplied function for this process.
//!
//! # Inlining
//!
//! Fearless SIMD relies heavily on Rust's inlining support to create functions which have the
//! given target features enabled.
//! As such, most functions which you write when using Fearless SIMD should have the `#[inline(always)]` attribute.
//!
//! <!--
//! # Kernels vs not kernels
//!
//! TODO: Talk about writing versions of functions which can be called in other `S: Simd` functions.
//! I think this pattern can also have a macro.
//! -->
//!
//! # Webassembly
//!
//! WASM SIMD doesn't have feature detection, and so you need to compile two versions of your bundle for WASM, one with SIMD and one without,
//! then select the appropriate one for your user's browser.
//! TODO: Expand on this.
//!
//! ## Credits
//!
//! This crate was inspired by [`pulp`], [`std::simd`], among others in the Rust ecosystem, though makes many decisions differently.
//! It benefited from conversations with Luca Versari, though he is not responsible for any of the mistakes or bad decisions.
//!
//! # Feature Flags
//!
//! The following crate [feature flags](https://doc.rust-lang.org/cargo/reference/features.html#dependency-features) are available:
//!
//! - `std` (enabled by default): Get floating point functions from the standard library (likely using your target's libc).
//!   Also allows using [`Level::new`] on all platforms, to detect which target features are enabled.
//! - `libm`: Use floating point implementations from [libm].
//! - `safe_wrappers`: Include safe wrappers for (some) target feature specific intrinsics,
//!   beyond the basic SIMD operations abstracted on all platforms.
//!
//! At least one of `std` and `libm` is required; `std` overrides `libm`.
//!
//! [`pulp`]: https://crates.io/crates/pulp
// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(non_camel_case_types, reason = "TODO")]
#![expect(clippy::unused_unit, reason = "easier for code generation")]
#![expect(
    clippy::new_without_default,
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

/// Implementations of [`Simd`] for 64 bit ARM.
#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    pub use crate::generated::Neon;
}

/// Implementations of [`Simd`] for webassembly.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm32 {
    pub use crate::generated::WasmSimd128;
}

/// Implementations of [`Simd`] on x86 architectures (both 32 and 64 bit).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86 {
    pub use crate::generated::Avx2;
    pub use crate::generated::Sse4_2;
}

/// The level enum with the specific SIMD capabilities available.
///
/// The contained values serve as a proof that the associated target
/// feature is available.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Level {
    /// Scalar fallback level, i.e. no supported SIMD features are to be used.
    ///
    /// This can be created with [`Level::fallback`].
    // We only want to compile the fallback implementation if:
    // - We're on a supported architecture, but don't statically support the lowest alternative level; OR
    // - We're on an unsupported architecture; OR
    // - The fallback is forcibly enabled
    #[cfg(any(
        all(target_arch = "aarch64", not(target_feature = "neon")),
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            not(target_feature = "sse4.2")
        ),
        all(target_arch = "wasm32", not(target_feature = "simd128")),
        not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )),
        feature = "force_support_fallback"
    ))]
    Fallback(Fallback),
    /// The Neon instruction set on 64 bit ARM.
    #[cfg(target_arch = "aarch64")]
    Neon(Neon),
    /// The SIMD 128 instructions on 32-bit WebAssembly.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    WasmSimd128(WasmSimd128),
    /// The SSE4.2 instruction set on (32 and 64 bit) x86.
    // We don't need to support this if the compilation target definitely supports something better.
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(all(target_feature = "avx2", target_feature = "fma"))
    ))]
    Sse4_2(Sse4_2),
    /// The AVX2 and FMA instruction set on (32 and 64 bit) x86.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2(Avx2),
    // If new variants are added, make sure to handle them in `Level::dispatch`
    // and `dispatch!()`
}

impl Level {
    /// Detect the available features on the current CPU, and returns the best level.
    ///
    /// If no SIMD instruction set is available, a scalar fallback will be used instead.
    ///
    /// This function requires the standard library, to use the
    /// [`is_x86_feature_detected`](std::arch::is_x86_feature_detected)
    /// or [`is_aarch64_feature_detected`](std::arch::is_aarch64_feature_detected).
    /// On wasm32, this requirement does not apply, so the standard library isn't required.
    ///
    /// Note that in most cases, this function should only be called by end-user applications.
    /// Libraries should instead accept a `Level` argument, probably as they are
    /// creating their data structures, then storing the level for any computations.
    /// Libraries which wish to abstract away SIMD usage for their common-case clients,
    /// should make their non-`Level` entrypoint match this function's `cfg`; to instead
    /// handle this at runtime, they can use [`try_detect`](Self::try_detect),
    /// handling the `None` case as they deem fit (probably panicking).
    /// This strategy avoids users of the library inadvertently using the fallback level,
    /// even if the requisite target features are available.
    ///
    /// If you are on an embedded device where these macros are not supported,
    /// you should construct the relevant variants yourself, using whatever
    /// way your specific chip supports accessing the current level.
    ///
    /// This value should be passed to [`dispatch!()`].
    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[must_use]
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { Level::Neon(Neon::new_unchecked()) };
        }
        #[cfg(target_arch = "wasm32")]
        {
            // WASM always either has the SIMD feature compiled in or not.
            #[cfg(target_feature = "simd128")]
            return Level::WasmSimd128(WasmSimd128::new_unchecked());
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return unsafe { Level::Avx2(Avx2::new_unchecked()) };
            } else if std::arch::is_x86_feature_detected!("sse4.2") {
                #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
                return unsafe { Level::Sse4_2(Sse4_2::new_unchecked()) };
            }
        }
        #[cfg(any(
            all(target_arch = "aarch64", not(target_feature = "neon")),
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                not(target_feature = "sse4.2")
            ),
            all(target_arch = "wasm32", not(target_feature = "simd128")),
            not(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "wasm32"
            )),
        ))]
        {
            return Level::Fallback(Fallback::new());
        }
        #[allow(
            unreachable_code,
            reason = "`is_x86_feature_detected` or equivalents will have returned `true`, or Fallback was used."
        )]
        {
            unreachable!()
        }
    }

    /// Get the target feature level suitable for this run.
    ///
    /// Should be used in libraries if they wish to handle the case where
    /// target features cannot be detected at runtime.
    /// Most users should prefer [`new`](Self::new).
    /// This is discussed in more detail in `new`'s documentation.
    #[allow(clippy::allow_attributes, reason = "Only needed in some cfgs.")]
    #[allow(unreachable_code, reason = "Fallback unreachable in some cfgs.")]
    pub fn try_detect() -> Option<Self> {
        #[cfg(any(feature = "std", target_arch = "wasm32"))]
        return Some(Self::new());
        None
    }

    /// If this is a proof that Neon (or better) is available, access that instruction set.
    ///
    /// This method should be preferred over matching against the `Neon` variant of self,
    /// because if Fearless SIMD gets support for an instruction set which is a superset of Neon,
    /// this method will return a value even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the `safe_wrappers` feature to gain checked access to
    /// the level-specific SIMD capabilities.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub fn as_neon(self) -> Option<Neon> {
        #[allow(
            unreachable_patterns,
            reason = "On machines which statically support `neon`, there is only one variant."
        )]
        match self {
            Level::Neon(neon) => Some(neon),
            _ => None,
        }
    }

    /// If this is a proof that SIMD 128 (or better) is available, access that instruction set.
    ///
    /// This method should be preferred over matching against the `WasmSimd128` variant of self,
    /// because if Fearless SIMD gets support for an instruction set which is a superset of SIMD 128,
    /// this method will return a value even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the `safe_wrappers` feature to gain checked access to
    /// the level-specific SIMD capabilities.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline]
    pub fn as_wasm_simd128(self) -> Option<WasmSimd128> {
        #[allow(
            unreachable_patterns,
            reason = "On machines which statically support `simd128`, there is only one variant."
        )]
        match self {
            Level::WasmSimd128(simd128) => Some(simd128),
            _ => None,
        }
    }

    /// If this is a proof that SSE4.2 (or better) is available, access that instruction set.
    ///
    /// This method should be preferred over matching against the `Sse4_2` variant of self,
    /// because if Fearless SIMD gets support for an instruction set which is a superset of SSE4.2,
    /// this method will return a value even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the `safe_wrappers` feature to gain checked access to
    /// the level-specific SIMD capabilities.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn as_sse4_2(self) -> Option<Sse4_2> {
        match self {
            // TODO: Level::Avx2(avx2) => Some(...)
            #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
            Level::Sse4_2(sse42) => Some(sse42),
            _ => None,
        }
    }

    /// If this is a proof that AVX2 and FMA (or better) is available, access that instruction set.
    ///
    /// This method should be preferred over matching against the `AVX2` variant of self,
    /// because if Fearless SIMD gets support for an instruction set which is a superset of AVX2,
    /// this method will return a value even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the `safe_wrappers` feature to gain checked access to
    /// the level-specific SIMD capabilities.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn as_avx2(self) -> Option<Avx2> {
        #[allow(
            unreachable_patterns,
            reason = "On machines which statically support `avx2`, there is only one variant."
        )]
        match self {
            Level::Avx2(avx2) => Some(avx2),
            _ => None,
        }
    }

    /// Get the strongest statically supported SIMD level.
    ///
    /// That is, if your compilation run ambiently declares that a target feature is enabled,
    /// this method will take that into account.
    /// In most cases, you should use [`Level::new`] or [`Level::try_detect`].
    /// This method is mainly useful for libraries, where:
    ///
    /// 1) Your crate features request that you not use the standard library, i.e. doesn't enable
    ///    your `"std"` crate feature reason (so you can't use [`Level::new`] and
    ///    [`Level::try_detect`] returns `None`); AND
    /// 2) Your caller does not provide a [`Level`]; AND
    /// 3) The library doesn't want to panic when it can't find a SIMD level.
    ///
    /// Note that in these cases, the library should clearly inform the integrator
    /// that it is using a fallback and so not getting optimal performance (e.g. by panicking if
    /// `debug_assertions` are enabled, and emitting a log with the "error" level otherwise).
    /// The messages given should also provide actionable fixes, such as pointing to the
    /// entry-point which provides a `Level`, or your `"std"` feature.
    ///
    /// Note that this is unaffected by the `force-support-fallback` feature.
    /// Instead, you should use [`Level::fallback`] if you require the fallback level.
    pub const fn baseline() -> Self {
        // TODO: How do we possibly test that this method works in all cases?
        #[cfg(not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )))]
        {
            return Level::Fallback(Fallback::new());
        }
        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(target_feature = "neon")]
            return unsafe { Level::Neon(Neon::new_unchecked()) };
            #[cfg(not(target_feature = "neon"))]
            return Level::Fallback(Fallback::new());
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
            return unsafe { Level::Avx2(Avx2::new_unchecked()) };
            #[cfg(all(
                target_feature = "sse4.2",
                not(all(target_feature = "avx2", target_feature = "fma"))
            ))]
            return unsafe { Level::Sse4_2(Sse4_2::new_unchecked()) };
            #[cfg(not(target_feature = "sse4.2"))]
            return Level::Fallback(Fallback::new());
        }
        #[cfg(target_arch = "wasm32")]
        {
            #[cfg(target_feature = "simd128")]
            return Level::WasmSimd128(WasmSimd128::new_unchecked());
            #[cfg(not(target_feature = "simd128"))]
            return Level::Fallback(Fallback::new());
        }
    }

    /// Create a scalar fallback level, which uses no SIMD instructions.
    ///
    /// This is primarily intended for tests; most users should prefer [`Level::new`] or [`Level::baseline`].
    #[inline]
    #[cfg(feature = "force_support_fallback")]
    pub const fn fallback() -> Self {
        Self::Fallback(Fallback::new())
    }

    /// Dispatch `f` to a context where the target features which this `Level` proves are available are [enabled].
    ///
    /// Most users of Fearless SIMD should prefer to use [`dispatch!()`] to
    /// explicitly vectorize a function. That has a better developer experience
    /// than an implementation of `WithSimd`, and is less likely to miss a vectorization
    /// opportunity.
    ///
    /// This has two use cases:
    /// 1) To call a manually written implementation of [`WithSimd`].
    /// 2) To ask the compiler to auto-vectorize scalar code.
    ///
    /// For the second case to work, the provided function *must* be attributed with `#[inline(always)]`.
    /// Note also that any calls that function makes to other functions will likely not be auto-vectorized,
    /// unless they are also `#[inline(always)]`.
    ///
    /// [enabled]: https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute
    #[inline]
    #[expect(
        unreachable_patterns,
        reason = "Level is `non_exhaustive`, but we are in the crate it's defined."
    )]
    pub fn dispatch<W: WithSimd>(self, f: W) -> W::Output {
        dispatch!(self, simd => f.with_simd(simd))
    }
}

#[cfg(test)]
mod tests {
    use crate::Level;

    const fn assert_is_send_sync<T: Send + Sync>() {}
    /// If this test compiles, we know that [`Level`] is properly `Send` and `Sync`.
    #[test]
    fn level_is_send_sync() {
        assert_is_send_sync::<Level>();
    }
}
