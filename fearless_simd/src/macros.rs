// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros publicly exported

/// Defines a new function which dispatches to a SIMD-generic function, enabling the correct
/// target features.
///
/// The implementation function (which is outside of this macro) *should* have the
/// `#[inline(always)]` attribute.
/// There are likely to be severe performance consequences if this is not the case, as
/// Rust will be unable to inline SIMD intrinsics in that case.
///
/// The `fn` token in the definition can be prefixed with `unsafe`, to allow an unsafe inner function.
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
