// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros publicly exported

/// Defines a new function which dispatches to a SIMD-generic function, enabling the correct
/// target features.
///
/// The `fn` token in the definition can be prefixed with a visibility (e.g. `pub`),
/// to set the visibility of the outer function.
/// We recommend that the implementation function remains private, and
/// should only be called through the dispatch function.
/// (The exact patterns for SIMD functions using Fearleess SIMD have not
/// yet been designed/enumerated).
///
/// The implementation function (which is outside of this macro) *should* have the
/// `#[inline(always)]` attribute.
/// There are likely to be severe performance consequences if this is not the case, as
/// Rust will be unable to inline SIMD intrinsics in that case.
///
/// The `fn` token in the definition can be prefixed with `unsafe`, to allow an unsafe inner function.
/// The safety comment added by you in the call to  `simd_dispatch` the function must have
/// the preconditions required to call the inner function.
///
/// # Examples
///
/// ```rust
/// use fearless_simd::{Simd, simd_dispatch};
///
/// #[inline(always)]
/// fn sigmoid_impl<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) { /* ... */ }
///
/// simd_dispatch!(fn sigmoid(level, x: &[f32], out: &mut [f32]) = sigmoid_impl);
/// ```
///
/// The signature of the generated function will be:
///
/// ```rust
/// use fearless_simd::Level;
/// fn sigmoid(level: Level, x: &[f32], out: &mut [f32]) { /* ... */ }
/// ```
#[macro_export]
#[deprecated = "use dispatch!(level, function) instead"]
macro_rules! simd_dispatch {
    (
        $( #[$meta:meta] )* $vis:vis
        unsafe fn $func:ident ( level $( , $arg:ident : $ty:ty $(,)? )* ) $( -> $ret:ty )?
        = $inner:ident
    ) => {
        simd_dispatch!{@impl => $(#[$meta])* $vis (unsafe) fn $func (level, $(,$arg:$ty,)*) $(->$ret)? = $inner}
    };
    (
        $( #[$meta:meta] )* $vis:vis
        fn $func:ident ( level $( , $arg:ident : $ty:ty $(,)? )* ) $( -> $ret:ty )?
        = $inner:ident
    ) => {
        simd_dispatch!{@impl => $(#[$meta])* $vis () fn $func (level $(,$arg:$ty)*) $(->$ret)? = $inner}
    };
    (
        @impl => $( #[$meta:meta] )* $vis:vis
        ($($unsafe: ident)?) fn $func:ident ( level $( , $arg:ident : $ty:ty $(,)? )* ) $( -> $ret:ty )?
        = $inner:ident
    ) => {
        $( #[$meta] )* $vis
        $($unsafe)? fn $func(level: $crate::Level $(, $arg: $ty )*) $( -> $ret )? {
            #[cfg(target_arch = "aarch64")]
            #[target_feature(enable = "neon")]
            #[inline]
            $($unsafe)? fn inner_neon(neon: $crate::aarch64::Neon $( , $arg: $ty )* ) $( -> $ret )? {
                $($unsafe)? {
                    $inner( neon $( , $arg )* )
                }
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            #[inline]
            $($unsafe)? fn inner_wasm_simd128(simd128: $crate::wasm32::WasmSimd128 $( , $arg: $ty )* ) $( -> $ret )? {
                $($unsafe)? {
                    $inner( simd128 $( , $arg )* )
                }
            }
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            #[target_feature(enable = "sse4.2")]
            #[inline]
            $($unsafe)? fn inner_sse4_2(sse4_2: $crate::x86::Sse4_2 $( , $arg: $ty )* ) $( -> $ret )? {
                $($unsafe)? {
                    $inner( sse4_2 $( , $arg )* )
                }
            }
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            #[target_feature(enable = "avx2,fma")]
            #[inline]
            $($unsafe)? fn inner_avx2(avx2: $crate::x86::Avx2 $( , $arg: $ty )* ) $( -> $ret )? {
                $($unsafe)? {
                    $inner( avx2 $( , $arg )* )
                }
            }
            match level {
                $crate::Level::Fallback(fb) => {
                    $($unsafe)? {
                        $inner(fb $( , $arg )* )
                    }
                },
                #[cfg(target_arch = "aarch64")]
                $crate::Level::Neon(neon) => unsafe { inner_neon (neon $( , $arg )* ) }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $crate::Level::WasmSimd128(wasm) => unsafe { inner_wasm_simd128 (wasm $( , $arg )* ) }
                #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
                $crate::Level::Sse4_2(sse4_2) => unsafe { inner_sse4_2(sse4_2 $( , $arg)* ) }
                #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
                $crate::Level::Avx2(avx2) => unsafe { inner_avx2(avx2 $( , $arg)* ) }
                _ => unreachable!()
            }
        }
    };
}

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
/// ```
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
    ($level:expr, $simd:pat => $op:expr) => {{
        /// Convert the `Simd` value into an `impl Simd`, which enforces that
        /// it is correctly handled.
        #[inline(always)]
        fn launder<S: $crate::Simd>(x: S) -> impl $crate::Simd {
            x
        }

        match $level {
            $crate::Level::Fallback(fb) => {
                let $simd = launder(fb);
                // This vectorize call does nothing, but it is reasonable to be consistent here.
                fb.vectorize(
                    #[inline(always)]
                    || $op,
                )
            }
            #[cfg(target_arch = "aarch64")]
            $crate::Level::Neon(neon) => {
                let $simd = launder(neon);
                neon.vectorize(
                    #[inline(always)]
                    || $op,
                )
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            $crate::Level::WasmSimd128(wasm) => {
                let $simd = launder(wasm);
                wasm.vectorize(
                    #[inline(always)]
                    || $op,
                )
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Sse4_2(sse4_2) => {
                let $simd = launder(sse4_2);
                sse4_2.vectorize(
                    #[inline(always)]
                    || $op,
                )
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Avx2(avx2) => {
                let $simd = launder(avx2);
                avx2.vectorize(
                    #[inline(always)]
                    || $op,
                )
            }
            _ => unreachable!(),
        }
    }};
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
}
