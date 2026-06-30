// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros publicly exported

/// Access the applicable [`Simd`] for a given `level`, and perform an operation using it.
///
/// This macro is the root of how any explicitly written SIMD functions in this crate are
/// called from a non-SIMD context.
///
/// The first parameter to the macro is the [`Level`].
/// You should prefer to construct a [`Level`] once and pass it around, rather than
/// frequently calling [`Level::new()`].
/// This is because `Level::new` has to detect which target features are available, which can be slow.
///
/// The code of the operation will be repeated literally several times in the output, so you should prefer
/// to keep this code small (as it will be type-checked, etc. for each supported SIMD level on your target).
/// In most cases, it should be a single call to a function which is generic over `Simd` implementations,
/// as seen in [the examples](#examples).
/// For clarity, it will only be executed once per execution of `dispatch`.
///
/// To guarantee target-feature-specific code generation, any functions called within the operation should
/// be `#[inline(always)]`.
///
/// Note that as an implementation detail of this macro, the operation will be executed inside a closure.
/// This is what enables the target features to be enabled for the code inside the operation.
/// A consequence of this is that early `return` and `?` will not work as expected.
/// Note that in cases where you use `dispatch` to call a single function (which we expect to be the
/// majority of cases), you can use `?` on the return value of dispatch instead.
/// To emulate early return, you can use [`ControlFlow`](core::ops::ControlFlow) instead.
///
/// # Example
///
/// ```rust
/// use fearless_simd::{Level, Simd, dispatch};
///
/// #[inline(always)]
/// fn sigmoid<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) { /* ... */ }
///
/// let level = Level::new();
///
/// dispatch!(level, simd => sigmoid(simd, &[/*...*/], &mut [/*...*/]));
/// ```
///
/// [`Level`]: crate::Level
/// [`Level::new()`]: crate::Level::new
/// [`Simd`]: crate::Simd
#[macro_export]
macro_rules! dispatch {
    // This falls through to the next branch, but with `forced_fallback_arm` turned into a boolean literal
    // indicating whether or not the `force_support_fallback` crate feature is enabled.
    ($level:expr, $simd:pat => $op:expr) => {{ $crate::internal_unstable_dispatch_inner!($level, $simd => $op) }};
    (@impl $level:expr, $simd:pat => $op:expr; $forced_fallback_arm: literal) => {{
        match $level {
            #[cfg(target_arch = "aarch64")]
            $crate::Level::Neon(neon) => {
                $crate::__fearless_simd_dispatch_with_token!(neon, $simd => $op)
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            $crate::Level::WasmSimd128(wasm) => {
                $crate::__fearless_simd_dispatch_with_token!(wasm, $simd => $op)
            }
            // This fallthrough logic is documented at the definition site of `Level`.
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
            $crate::Level::Sse4_2(sse4_2) => {
                $crate::__fearless_simd_dispatch_dispatch_sse4_2!(sse4_2, $simd => $op)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Avx512(avx512) => {
                $crate::__fearless_simd_dispatch_dispatch_avx512!(avx512, $simd => $op)
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Avx2(avx2) => {
                $crate::__fearless_simd_dispatch_dispatch_avx2!(avx2, $simd => $op)
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
                $forced_fallback_arm
            ))]
            $crate::Level::Fallback(fb) => {
                // This vectorize call does nothing for Fallback, but it is reasonable to be consistent here.
                $crate::__fearless_simd_dispatch_with_token!(fb, $simd => $op)
            }
            _ => unreachable!(),
        }
    }};
}

// The x86 multiversion helpers are split into cfg-selected macro definitions
// because exported macro bodies are expanded in the downstream crate,
// so putting `#[cfg(feature = "...")]` directly inside `dispatch!` would test
// the downstream crate's features instead of `fearless_simd`'s features.

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
macro_rules! __fearless_simd_dispatch_with_token {
    ($token:expr, $simd:pat => $op:expr) => {{
        /// Convert the `Simd` value into an `impl Simd`, which enforces that
        /// it is correctly handled.
        // TODO: Just make into a `pub` function in fearless_simd itself?
        #[inline(always)]
        fn launder<S: $crate::Simd>(x: S) -> impl $crate::Simd {
            x
        }

        // $token can be an arbitrary expression,
        // so bind the result of evaluating it to a variable before use
        // so that it's evaluated only once
        let __fearless_simd_token = $token;
        let $simd = launder(__fearless_simd_token);
        $crate::Simd::vectorize(
            __fearless_simd_token,
            #[inline(always)]
            || $op,
        )
    }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(feature = "dispatch_avx512")]
macro_rules! __fearless_simd_dispatch_dispatch_avx512 {
    ($avx512:expr, $simd:pat => $op:expr) => {
        $crate::__fearless_simd_dispatch_with_token!($avx512, $simd => $op)
    };
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(not(feature = "dispatch_avx512"))]
macro_rules! __fearless_simd_dispatch_dispatch_avx512 {
    ($avx512:expr, $simd:pat => $op:expr) => {{
        $crate::__fearless_simd_dispatch_dispatch_avx2_from_superset!($avx512, $simd => $op)
    }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(feature = "dispatch_avx2")]
macro_rules! __fearless_simd_dispatch_dispatch_avx2 {
    ($avx2:expr, $simd:pat => $op:expr) => {
        $crate::__fearless_simd_dispatch_with_token!($avx2, $simd => $op)
    };
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(not(feature = "dispatch_avx2"))]
macro_rules! __fearless_simd_dispatch_dispatch_avx2 {
    ($avx2:expr, $simd:pat => $op:expr) => {{
        $crate::__fearless_simd_dispatch_dispatch_sse4_2_from_superset!($avx2, $simd => $op)
    }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(feature = "dispatch_avx2")]
macro_rules! __fearless_simd_dispatch_dispatch_avx2_from_superset {
    ($proof:expr, $simd:pat => $op:expr) => {{
        let __fearless_simd_proof = $proof;
        let __fearless_simd_token = $crate::Simd::level(__fearless_simd_proof)
            .as_avx2()
            .expect("a superset x86 SIMD level should provide an AVX2 token");
        $crate::__fearless_simd_dispatch_with_token!(__fearless_simd_token, $simd => $op)
    }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(not(feature = "dispatch_avx2"))]
macro_rules! __fearless_simd_dispatch_dispatch_avx2_from_superset {
    ($proof:expr, $simd:pat => $op:expr) => {{
        $crate::__fearless_simd_dispatch_dispatch_sse4_2_from_superset!($proof, $simd => $op)
    }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(feature = "dispatch_sse4_2")]
macro_rules! __fearless_simd_dispatch_dispatch_sse4_2 {
    ($sse4_2:expr, $simd:pat => $op:expr) => {
        $crate::__fearless_simd_dispatch_with_token!($sse4_2, $simd => $op)
    };
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(not(feature = "dispatch_sse4_2"))]
macro_rules! __fearless_simd_dispatch_dispatch_sse4_2 {
    ($sse4_2:expr, $simd:pat => $op:expr) => {{
        let __fearless_simd_proof = $sse4_2;
        let _ = __fearless_simd_proof;
        $crate::__fearless_simd_dispatch_with_token!($crate::Fallback::new(), $simd => $op)
    }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(feature = "dispatch_sse4_2")]
macro_rules! __fearless_simd_dispatch_dispatch_sse4_2_from_superset {
    ($proof:expr, $simd:pat => $op:expr) => {{
        let __fearless_simd_proof = $proof;
        let __fearless_simd_token = $crate::Simd::level(__fearless_simd_proof)
            .as_sse4_2()
            .expect("a superset x86 SIMD level should provide an SSE4.2 token");
        $crate::__fearless_simd_dispatch_with_token!(__fearless_simd_token, $simd => $op)
    }};
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(not(feature = "dispatch_sse4_2"))]
macro_rules! __fearless_simd_dispatch_dispatch_sse4_2_from_superset {
    ($proof:expr, $simd:pat => $op:expr) => {{
        let __fearless_simd_proof = $proof;
        let _ = __fearless_simd_proof;
        $crate::__fearless_simd_dispatch_with_token!($crate::Fallback::new(), $simd => $op)
    }};
}

// This macro turns whether the `force_support_fallback` macro is enabled into a boolean literal
// in `dispatch`, which allows it to be used correctly cross-crate.
// This trickery is required because macros are expanded in the context of the calling crate, including for
// evaluating `cfg`s.

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(feature = "force_support_fallback")]
macro_rules! internal_unstable_dispatch_inner {
    ($level:expr, $simd:pat => $op:expr) => {
        $crate::dispatch!(
            @impl $level, $simd => $op; true
        )
    };
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(not(feature = "force_support_fallback"))]
macro_rules! internal_unstable_dispatch_inner {
    ($level:expr, $simd:pat => $op:expr) => {
        $crate::dispatch!(@impl $level, $simd => $op; false)
    };
}

#[cfg(test)]
// This expect also validates that we haven't missed any levels!
#[expect(
    unreachable_patterns,
    reason = "Level is non_exhaustive, but you must be exhaustive within the same crate."
)]
mod tests {
    use crate::{Level, Simd};

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum X86DispatchBackend {
        Fallback,
        Sse4_2,
        Avx2,
        Avx512,
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn x86_dispatch_backend<S: Simd>(_: S) -> X86DispatchBackend {
        use core::any::TypeId;

        if TypeId::of::<S>() == TypeId::of::<crate::Fallback>() {
            X86DispatchBackend::Fallback
        } else if TypeId::of::<S>() == TypeId::of::<crate::Sse4_2>() {
            X86DispatchBackend::Sse4_2
        } else if TypeId::of::<S>() == TypeId::of::<crate::Avx2>() {
            X86DispatchBackend::Avx2
        } else if TypeId::of::<S>() == TypeId::of::<crate::Avx512>() {
            X86DispatchBackend::Avx512
        } else {
            unreachable!("unexpected x86 dispatch backend")
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn expected_x86_dispatch_backend(level: Level) -> X86DispatchBackend {
        if cfg!(feature = "dispatch_avx512") && level.as_avx512().is_some() {
            X86DispatchBackend::Avx512
        } else if cfg!(feature = "dispatch_avx2") && level.as_avx2().is_some() {
            X86DispatchBackend::Avx2
        } else if cfg!(feature = "dispatch_sse4_2") && level.as_sse4_2().is_some() {
            X86DispatchBackend::Sse4_2
        } else {
            X86DispatchBackend::Fallback
        }
    }

    #[allow(dead_code, reason = "Compile test")]
    fn dispatch_generic() {
        fn generic<S: Simd, T>(_: S, x: T) -> T {
            x
        }
        dispatch!(Level::new(), simd => generic::<_, ()>(simd, ()));
    }

    #[allow(dead_code, reason = "Compile test")]
    fn dispatch_value() {
        fn make_fn<S: Simd>() -> impl FnOnce(S) {
            |_| ()
        }
        dispatch!(Level::new(), simd => (make_fn())(simd));
    }

    #[test]
    fn dispatch_output() {
        assert_eq!(42, dispatch!(Level::new(), _simd => 42));
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn dispatch_respects_x86_multiversion_features() {
        let level = Level::new();
        let actual = dispatch!(level, simd => x86_dispatch_backend(simd));

        assert_eq!(actual, expected_x86_dispatch_backend(level));
    }

    #[cfg(all(
        feature = "std",
        any(target_arch = "x86", target_arch = "x86_64"),
        not(feature = "dispatch_avx512")
    ))]
    #[test]
    fn disabled_avx512_multiversioning_does_not_filter_avx512_token_access() {
        if crate::x86_detects_icelake_avx512() {
            assert!(
                Level::new().as_avx512().is_some(),
                "`dispatch_avx512` controls dispatch multiversioning, not AVX-512 token access"
            );
        }
    }

    mod no_import_simd {
        /// We should be able to use [`dispatch`] in a scope which doesn't import anything.
        #[test]
        fn dispatch_with_no_imports() {
            let res = dispatch!(crate::Level::new(), _ => 1 + 2);
            assert_eq!(res, 3);
        }
    }
}
