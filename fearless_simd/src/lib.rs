// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// After you edit the crate's doc comment, run this command, then check README.md for any missing links
// cargo rdme --workspace-project=fearless_simd

//! `fearless_simd` takes `unsafe` out of SIMD.
//!
//! No matter what level of abstraction you're after, be it autovectorization and multiversioning, or portable SIMD, or safe access to raw
//! intrinsics and nothing more, `fearless_simd` has you covered!
//!
//! Zero dependencies, from-scratch build time under 1 second, safe public APIs, and [very little](https://gist.github.com/Shnatsel/61fc294987a1e051ce3835c97dc0fc19) `unsafe` under the hood.
//!
//! # Automatic vectorization
//!
//! Put the code to vectorize in an `#[inline(always)]` function generic over [`Simd`].
//!
//! This will generate several implementations for different SIMD levels and select the best one at runtime:
//!
//! ```rust
//! use fearless_simd::{dispatch, Level, Simd};
//!
//! #[inline(always)]
//! fn double_u32s<S: Simd>(_: S, values: &mut [u32]) {
//!     for value in values {
//!         *value = *value * 2;
//!     }
//! }
//!
//! let mut values = [1, 2, 3, 4, 5];
//! let level = Level::new(); // Detect SIMD available on the CPU. Expensive, so do it once.
//! dispatch!(level, simd => double_u32s(simd, &mut values));
//! assert_eq!(values, [2, 4, 6, 8, 10]);
//! ```
//!
//! # Portable SIMD
//!
//! Use the vector types for explicit lane-wise operations while staying generic over the SIMD level:
//!
//! ```rust
//! use fearless_simd::{dispatch, prelude::*, Level};
//!
//! #[inline(always)]
//! fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
//!     let mut chunks = values.chunks_exact_mut(S::u32s::N); // the CPU's native SIMD width
//!     for chunk in &mut chunks {
//!         let v = S::u32s::from_slice(simd, chunk);
//!         (v * 2).store_slice(chunk);
//!     }
//!     for value in chunks.into_remainder() {
//!         *value = *value * 2;
//!     }
//! }
//!
//! let mut values = [1, 2, 3, 4, 5];
//! let level = Level::new(); // Detect SIMD available on the CPU. Expensive, so do it once.
//! dispatch!(level, simd => double_u32s(simd, &mut values));
//! assert_eq!(values, [2, 4, 6, 8, 10]);
//! ```
//!
//! You can also use fixed-size types such as [u32x8] instead of using the hardware's native SIMD width.
//!
//! # Explicit intrinsics
//!
//! If you need access to raw intrinsics, [`kernel!`][kernel] creates a function where they can be called safely:
//!
//! ```rust
//! use fearless_simd::{prelude::*, Level, u32x4};
//!
//! fearless_simd::kernel!(
//!     fn double_u32s_neon(neon: Neon, values: &mut [u32]) {
//!         use core::arch::aarch64::*;
//!
//!         let mut chunks = values.chunks_exact_mut(4);
//!         for chunk in &mut chunks {
//!             let v: uint32x4_t = u32x4::from_slice(neon, chunk).into(); // safe load
//!             let doubled = vmulq_u32(v, vdupq_n_u32(2)); // safe access to a NEON intrinsic
//!             let doubled: u32x4<_> = doubled.simd_into(neon);
//!             doubled.store_slice(chunk);
//!         }
//!         for value in chunks.into_remainder() {
//!             *value = *value * 2;
//!         }
//!     }
//! );
//!
//! #[cfg(target_arch = "aarch64")]
//! {
//!     let level = Level::new(); // Detect SIMD available on the CPU. Expensive, so do it once.
//!     if let Some(neon) = level.as_neon() {
//!         let mut values = [1, 2, 3, 4, 5];
//!         double_u32s_neon(neon, &mut values);
//!         assert_eq!(values, [2, 4, 6, 8, 10]);
//!     }
//! }
//! ```
//!
//! You can also [mix and match](https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/srgb.rs)
//! intrinsics with the other approaches, using high-level code most of the time and dropping down to
//! hardware-specific intrinsics only when necessary.
//!
//! # Inlining
//!
//! Fearless SIMD relies heavily on Rust's inlining support to create functions which have the given target features enabled.
//!
//! As a rule of thumb:
//!
//! - All SIMD functions need `#[inline(always)]`.
//! - Use [`dispatch`] when calling SIMD code from non-SIMD code.
//! - Use [`vectorize()`][Simd::vectorize] when calling SIMD from SIMD if you don't want to force inlining.
//!
//! [The article describing the design](https://gist.github.com/Shnatsel/61fc294987a1e051ce3835c97dc0fc19#the-abi-would-like-a-word) covers why this is the
//! case. There's also Q&A on [Zulip](https://xi.zulipchat.com/#narrow/channel/514230-simd/topic/inlining/with/546913433).
//!
//! # Instruction set support
//!
//! - x86/x86-64: [v2](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) (SSE4.2), [v3](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) (AVX2), [Ice Lake](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512) (AVX-512, avoiding early slow implementations)
//! - Aarch64: Baseline [NEON](https://en.wikipedia.org/wiki/Arm_architecture_family#Advanced_SIMD_(Neon))
//! - WebAssembly: [128-bit packed SIMD](https://github.com/WebAssembly/spec/blob/main/proposals/simd/SIMD.md), [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)
//!
//! A scalar fallback is also provided for platforms, so your code still works even if SIMD is not available.
//!
//! # WebAssembly
//!
//! WASM SIMD doesn't have feature detection, and so you need to compile two versions of your bundle for WASM, one with SIMD and one without,
//! then select the appropriate one for your user's browser. This can be done via [the `wasm-feature-detect`
//! library](https://github.com/GoogleChromeLabs/wasm-feature-detect).
//!
//! You can compile WebAssembly with the SIMD128 feature enabled via the `RUSTFLAGS` environment variable
//! (`RUSTFLAGS="-Ctarget-feature=+simd128"`), or by adding the compiler flags in your [Cargo
//! config.toml](https://doc.rust-lang.org/cargo/reference/config.html):
//!
//! ```toml
//! [target.'cfg(target_arch = "wasm32")']
//! rustflags = ["-Ctarget-feature=+simd128"]
//! rustdocflags = ["-Ctarget-feature=+simd128"]
//! ```
//!
//! If you want to compile both SIMD and non-SIMD versions of your WebAssembly library, your best option right now is to create a shell script
//! that builds it once with the `RUSTFLAGS` specified, and once without. [Cargo currently does not allow specifying compiler flags
//! per-profile.](https://github.com/rust-lang/cargo/issues/10271)
//!
//! ## Relaxed SIMD
//!
//! Fearless SIMD can make use of the [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)
//! WebAssembly instructions, if the requisite target feature is enabled. These instructions can return implementation-dependent results
//! depending on what is fastest on the underlying hardware. They are only used for operations where we already give hardware-dependent results.
//!
//! At the time of writing, relaxed SIMD is only supported in Chrome. To make use of it, you'll need to build two versions of your library, one
//! with relaxed SIMD enabled (`RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd"`) and one with it disabled, and then feature-detect at
//! runtime.
//!
//! # Credits
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
//! - `force_support_fallback`: Force scalar fallback, to be supported, even if your compilation target has a better baseline.
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
#![cfg_attr(not(test), deny(clippy::disallowed_methods))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(non_camel_case_types, reason = "TODO")]
#![expect(clippy::unused_unit, reason = "easier for code generation")]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

#[cfg(all(not(feature = "libm"), not(feature = "std")))]
compile_error!("fearless_simd requires either the `std` or `libm` feature");

// Suppress the unused_crate_dependencies lint when both std and libm are specified.
#[cfg(all(feature = "std", feature = "libm"))]
use libm as _;

mod generated;
mod kernel_macros;
mod macros;
mod support;
mod traits;
mod transmute;

pub use generated::*;
pub use traits::*;

/// This prelude module re-exports every SIMD trait defined in this library. It's useful for accessing trait methods.
///
/// Only traits are exported through the prelude; types must be exported separately.
pub mod prelude {
    pub use crate::generated::simd_trait::*;
    pub use crate::traits::*;
}

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
    pub use crate::generated::Avx512;
    pub use crate::generated::Sse4_2;
}

// Sourced from `rustc --print=cfg --target x86_64-unknown-linux-gnu -C target-cpu=icelake-server`
// and pruned against the features implied by `avx512f` which can be viewed via
// `rustc --print=cfg --target x86_64-unknown-linux-gnu -C target-feature='+avx2'`
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn x86_detects_icelake_avx512() -> bool {
    std::arch::is_x86_feature_detected!("adx")
        && std::arch::is_x86_feature_detected!("aes")
        && std::arch::is_x86_feature_detected!("avx512bitalg")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512cd")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512ifma")
        && std::arch::is_x86_feature_detected!("avx512vbmi")
        && std::arch::is_x86_feature_detected!("avx512vbmi2")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("avx512vnni")
        && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("cmpxchg16b")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("gfni")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("movbe")
        && std::arch::is_x86_feature_detected!("pclmulqdq")
        && std::arch::is_x86_feature_detected!("popcnt")
        && std::arch::is_x86_feature_detected!("rdrand")
        && std::arch::is_x86_feature_detected!("rdseed")
        && std::arch::is_x86_feature_detected!("sha")
        && std::arch::is_x86_feature_detected!("vaes")
        && std::arch::is_x86_feature_detected!("vpclmulqdq")
        && std::arch::is_x86_feature_detected!("xsave")
        && std::arch::is_x86_feature_detected!("xsavec")
        && std::arch::is_x86_feature_detected!("xsaveopt")
        && std::arch::is_x86_feature_detected!("xsaves")
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
            not(all(
                target_feature = "sse4.2",
                target_feature = "cmpxchg16b",
                target_feature = "popcnt"
            ))
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
    /// The SSE4.2 instruction set on (32 and 64 bit) x86, plus `popcnt` and `cmpxchg16b`.
    /// Also known as x86-64-v2.
    ///
    /// All production CPUs with SSE4.2 also support the other two extensions, so it is safe to require them.
    // We don't need to support this if the compilation target definitely supports something better.
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(all(
            target_feature = "avx2",
            target_feature = "bmi1",
            target_feature = "bmi2",
            target_feature = "cmpxchg16b",
            target_feature = "f16c",
            target_feature = "fma",
            target_feature = "lzcnt",
            target_feature = "movbe",
            target_feature = "popcnt",
            target_feature = "xsave"
        ))
    ))]
    Sse4_2(Sse4_2),
    /// Ice Lake-class AVX-512 on (32 and 64 bit) x86.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512(Avx512),
    /// The x86-64-v3 instruction set on (32 and 64 bit) x86, including AVX2 and FMA.
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
    /// This value should be passed to [`dispatch`].
    #[cfg(any(feature = "std", target_arch = "wasm32"))]
    #[must_use]
    #[expect(
        clippy::new_without_default,
        reason = "The `Level::new()` function is not always available, and we also want to be explicit about when runtime feature detection happens"
    )]
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { Self::Neon(Neon::new_unchecked()) };
        }
        #[cfg(target_arch = "wasm32")]
        {
            // WASM always either has the SIMD feature compiled in or not.
            #[cfg(target_feature = "simd128")]
            return Self::WasmSimd128(WasmSimd128::new_unchecked());
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if x86_detects_icelake_avx512() {
                return unsafe { Self::Avx512(Avx512::new_unchecked()) };
            }

            // Feature list sourced from `rustc --print=cfg --target x86_64-unknown-linux-gnu -C target-cpu=x86-64-v3`
            // However, the following features are implied by avx2 and do not need to be spelled out:
            // avx,fxsr,sse,sse2,sse3,sse4.1,sse4.2,ssse3
            // This can be verified by running:
            // rustc --print=cfg --target x86_64-unknown-linux-gnu -C target-feature='+avx2'
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("bmi1")
                && std::arch::is_x86_feature_detected!("bmi2")
                && std::arch::is_x86_feature_detected!("cmpxchg16b")
                && std::arch::is_x86_feature_detected!("f16c")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("lzcnt")
                && std::arch::is_x86_feature_detected!("movbe")
                && std::arch::is_x86_feature_detected!("popcnt")
                && std::arch::is_x86_feature_detected!("xsave")
            {
                return unsafe { Self::Avx2(Avx2::new_unchecked()) };
            // All x86 CPUs that ever shipped with sse4.2 also have cmpxchg16b and popcnt:
            // Intel Nehalem, AMD Bulldozer and VIA Isaiah II were the first with SSE4.2
            // and have these extensions already.
            } else if std::arch::is_x86_feature_detected!("sse4.2")
                && std::arch::is_x86_feature_detected!("cmpxchg16b")
                && std::arch::is_x86_feature_detected!("popcnt")
            {
                #[cfg(not(all(
                    target_feature = "avx2",
                    target_feature = "bmi1",
                    target_feature = "bmi2",
                    target_feature = "cmpxchg16b",
                    target_feature = "f16c",
                    target_feature = "fma",
                    target_feature = "lzcnt",
                    target_feature = "movbe",
                    target_feature = "popcnt",
                    target_feature = "xsave"
                )))]
                return unsafe { Self::Sse4_2(Sse4_2::new_unchecked()) };
            }
        }
        #[cfg(any(
            all(target_arch = "aarch64", not(target_feature = "neon")),
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                not(all(
                    target_feature = "sse4.2",
                    target_feature = "cmpxchg16b",
                    target_feature = "popcnt"
                ))
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
            return Self::Fallback(Fallback::new());
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

    /// Check whether this is the `Fallback` level; that is, whether no better feature level could
    /// be statically or dynamically detected. This is useful if there's a scalarized version of
    /// your algorithm that runs faster if SIMD isn't supported.
    ///
    /// This method is always available, even in cases where `Fallback` is not; for instance, if
    /// you're targeting a platform that always supports some level of SIMD. In such cases, it will
    /// always return false.
    pub fn is_fallback(self) -> bool {
        #[cfg(any(
            all(target_arch = "aarch64", not(target_feature = "neon")),
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                not(all(
                    target_feature = "sse4.2",
                    target_feature = "cmpxchg16b",
                    target_feature = "popcnt"
                ))
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
        return matches!(self, Self::Fallback(_));

        #[allow(unreachable_code, reason = "Fallback unreachable in some cfgs.")]
        false
    }

    /// If this is a proof that Neon (or better) is available, access that instruction set.
    ///
    /// This method should be preferred over matching against the `Neon` variant of self,
    /// because if Fearless SIMD gets support for an instruction set which is a superset of Neon,
    /// this method will return the Neon token even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub fn as_neon(self) -> Option<Neon> {
        #[allow(
            unreachable_patterns,
            reason = "On machines which statically support `neon`, there is only one variant."
        )]
        match self {
            Self::Neon(neon) => Some(neon),
            _ => None,
        }
    }

    /// If this is a proof that SIMD 128 (or better) is available, access that instruction set.
    ///
    /// This method should be preferred over matching against the `WasmSimd128` variant of self,
    /// because if Fearless SIMD gets support for an instruction set which is a superset of SIMD 128,
    /// this method will return the SIMD 128 token even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline]
    pub fn as_wasm_simd128(self) -> Option<WasmSimd128> {
        #[allow(
            unreachable_patterns,
            reason = "On machines which statically support `simd128`, there is only one variant."
        )]
        match self {
            Self::WasmSimd128(simd128) => Some(simd128),
            _ => None,
        }
    }

    /// If this is a proof that x86-64-v2 feature set (or better) is available, access that
    /// instruction set.
    ///
    /// See [`Sse4_2::new_unchecked`] for the exact list of CPU features this token enables.
    ///
    /// This method should be preferred over matching against the `Sse4_2` variant of self,
    /// because if the CPU supports a superset of SSE4.2 (e.g. AVX2 or AVX-512),
    /// this method will return the SSE4.2 token even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn as_sse4_2(self) -> Option<Sse4_2> {
        match self {
            // Safety: The Avx512 struct represents an Ice Lake feature set, which includes the
            // `sse4.2`, `cmpxchg16b`, and `popcnt` features required by Sse4_2.
            Self::Avx512(_avx512) => unsafe { Some(Sse4_2::new_unchecked()) },
            // Safety: The Avx2 struct represents the x86-64-v3 feature set being enabled, which
            // includes the `sse4.2`, `cmpxchg16b`, and `popcnt` features required by Sse4_2.
            Self::Avx2(_avx) => unsafe { Some(Sse4_2::new_unchecked()) },
            #[cfg(not(all(
                target_feature = "avx2",
                target_feature = "bmi1",
                target_feature = "bmi2",
                target_feature = "cmpxchg16b",
                target_feature = "f16c",
                target_feature = "fma",
                target_feature = "lzcnt",
                target_feature = "movbe",
                target_feature = "popcnt",
                target_feature = "xsave"
            )))]
            Self::Sse4_2(sse42) => Some(sse42),
            #[allow(
                unreachable_patterns,
                reason = "This arm is reachable on baseline x86/x86_64."
            )]
            _ => None,
        }
    }

    /// If this is a proof that the x86-64-v3 feature set (or better) is available, access that
    /// instruction set.
    ///
    /// See [`Avx2::new_unchecked`] for the exact list of CPU features this token enables.
    ///
    /// This method should be preferred over matching against the `Avx2` variant of self,
    /// because if the CPU supports a superset of AVX2 (e.g. AVX-512),
    /// this method will return the AVX2 token even if that "better" instruction set is available.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn as_avx2(self) -> Option<Avx2> {
        #[allow(
            unreachable_patterns,
            reason = "On machines which statically support `avx2`, there is only one variant."
        )]
        match self {
            // Safety: The Ice Lake AVX-512 feature set includes the x86-64-v3 features required by Avx2.
            Self::Avx512(_avx512) => unsafe { Some(Avx2::new_unchecked()) },
            Self::Avx2(avx2) => Some(avx2),
            _ => None,
        }
    }

    /// If this is a proof that the Ice Lake AVX-512 feature set is available, access that
    /// instruction set.
    ///
    /// See [`Avx512::new_unchecked`] for the exact list of CPU features this token enables.
    ///
    /// This can be used in combination with the [kernel] macro to safely access level-specific
    /// SIMD intrinsics.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn as_avx512(self) -> Option<Avx512> {
        match self {
            Self::Avx512(avx512) => Some(avx512),
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
        // Note that you can use the `check_targets.sh` script to at least ensure that it compiles in all reasonable cases.
        #[cfg(not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )))]
        {
            return Self::Fallback(Fallback::new());
        }
        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(target_feature = "neon")]
            return unsafe { Self::Neon(Neon::new_unchecked()) };
            #[cfg(not(target_feature = "neon"))]
            return Self::Fallback(Fallback::new());
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(all(
                target_feature = "adx",
                target_feature = "aes",
                target_feature = "avx512bitalg",
                target_feature = "avx512bw",
                target_feature = "avx512cd",
                target_feature = "avx512dq",
                target_feature = "avx512f",
                target_feature = "avx512ifma",
                target_feature = "avx512vbmi",
                target_feature = "avx512vbmi2",
                target_feature = "avx512vl",
                target_feature = "avx512vnni",
                target_feature = "avx512vpopcntdq",
                target_feature = "bmi1",
                target_feature = "bmi2",
                target_feature = "cmpxchg16b",
                target_feature = "fma",
                target_feature = "gfni",
                target_feature = "lzcnt",
                target_feature = "movbe",
                target_feature = "pclmulqdq",
                target_feature = "popcnt",
                target_feature = "rdrand",
                target_feature = "rdseed",
                target_feature = "sha",
                target_feature = "vaes",
                target_feature = "vpclmulqdq",
                target_feature = "xsave",
                target_feature = "xsavec",
                target_feature = "xsaveopt",
                target_feature = "xsaves"
            ))]
            return unsafe { Self::Avx512(Avx512::new_unchecked()) };
            #[cfg(all(
                target_feature = "avx2",
                target_feature = "bmi1",
                target_feature = "bmi2",
                target_feature = "cmpxchg16b",
                target_feature = "f16c",
                target_feature = "fma",
                target_feature = "lzcnt",
                target_feature = "movbe",
                target_feature = "popcnt",
                target_feature = "xsave",
                not(all(
                    target_feature = "adx",
                    target_feature = "aes",
                    target_feature = "avx512bitalg",
                    target_feature = "avx512bw",
                    target_feature = "avx512cd",
                    target_feature = "avx512dq",
                    target_feature = "avx512f",
                    target_feature = "avx512ifma",
                    target_feature = "avx512vbmi",
                    target_feature = "avx512vbmi2",
                    target_feature = "avx512vl",
                    target_feature = "avx512vnni",
                    target_feature = "avx512vpopcntdq",
                    target_feature = "bmi1",
                    target_feature = "bmi2",
                    target_feature = "cmpxchg16b",
                    target_feature = "fma",
                    target_feature = "gfni",
                    target_feature = "lzcnt",
                    target_feature = "movbe",
                    target_feature = "pclmulqdq",
                    target_feature = "popcnt",
                    target_feature = "rdrand",
                    target_feature = "rdseed",
                    target_feature = "sha",
                    target_feature = "vaes",
                    target_feature = "vpclmulqdq",
                    target_feature = "xsave",
                    target_feature = "xsavec",
                    target_feature = "xsaveopt",
                    target_feature = "xsaves"
                ))
            ))]
            return unsafe { Self::Avx2(Avx2::new_unchecked()) };
            #[cfg(all(
                all(
                    target_feature = "sse4.2",
                    target_feature = "cmpxchg16b",
                    target_feature = "popcnt"
                ),
                not(all(
                    target_feature = "avx2",
                    target_feature = "bmi1",
                    target_feature = "bmi2",
                    target_feature = "cmpxchg16b",
                    target_feature = "f16c",
                    target_feature = "fma",
                    target_feature = "lzcnt",
                    target_feature = "movbe",
                    target_feature = "popcnt",
                    target_feature = "xsave"
                ))
            ))]
            return unsafe { Self::Sse4_2(Sse4_2::new_unchecked()) };
            #[cfg(not(all(
                target_feature = "sse4.2",
                target_feature = "cmpxchg16b",
                target_feature = "popcnt"
            )))]
            return Self::Fallback(Fallback::new());
        }
        #[cfg(target_arch = "wasm32")]
        {
            #[cfg(target_feature = "simd128")]
            return Self::WasmSimd128(WasmSimd128::new_unchecked());
            #[cfg(not(target_feature = "simd128"))]
            return Self::Fallback(Fallback::new());
        }
    }

    /// Create a scalar fallback level, which uses no SIMD instructions.
    ///
    /// This is primarily intended for tests; most users should prefer [`Level::new`] or [`Level::baseline`].
    ///
    /// Note that enabling the scalar fallback does *not* mean that the fallback branch will not
    /// contain SIMD instructions. This is because the "ambient" compilation environment has SIMD
    /// instructions available, which may be utilised by LLVM to auto-vectorise that path.
    #[inline]
    #[cfg(feature = "force_support_fallback")]
    pub const fn fallback() -> Self {
        Self::Fallback(Fallback::new())
    }

    /// Dispatch `f` to a context where the target features which this `Level` proves are available are [enabled].
    ///
    /// Most users of Fearless SIMD should prefer to use [`dispatch`] to
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
