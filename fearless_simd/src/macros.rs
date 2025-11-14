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
/// ```rust,ignore
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
        /// Convert the `Simd` value into an `impl Simd`, which enforces that
        /// it is correctly handled.
        // TODO: Just make into a `pub` function in fearless_simd itself?
        #[inline(always)]
        fn launder<S: $crate::Simd>(x: S) -> impl $crate::Simd {
            x
        }

        match $level {
            #[cfg(target_arch = "aarch64")]
            $crate::Level::Neon(neon) => {
                let $simd = launder(neon);
                $crate::Simd::vectorize(
                    neon,
                    #[inline(always)]
                    || $op,
                )
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            $crate::Level::WasmSimd128(wasm) => {
                let $simd = launder(wasm);
                $crate::Simd::vectorize(
                    wasm,
                    #[inline(always)]
                    || $op,
                )
            }
            // This fallthrough logic is documented at the definition site of `Level`.
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                not(all(target_feature = "avx2", target_feature = "fma"))
            ))]
            $crate::Level::Sse4_2(sse4_2) => {
                let $simd = launder(sse4_2);
                $crate::Simd::vectorize(
                    sse4_2,
                    #[inline(always)]
                    || $op,
                )
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Avx2(avx2) => {
                let $simd = launder(avx2);
                $crate::Simd::vectorize(
                    avx2,
                    #[inline(always)]
                    || $op,
                )
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
                $forced_fallback_arm
            ))]
            $crate::Level::Fallback(fb) => {
                let $simd = launder(fb);
                // This vectorize call does nothing, but it is reasonable to be consistent here.
                $crate::Simd::vectorize(
                    fb,
                    #[inline(always)]
                    || $op,
                )
            }
            _ => unreachable!(),
        }
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

    mod no_import_simd {
        /// We should be able to use [`dispatch`] in a scope which doesn't import anything.
        #[test]
        fn dispatch_with_no_imports() {
            let res = dispatch!(crate::Level::new(), _ => 1 + 2);
            assert_eq!(res, 3);
        }
    }
}
