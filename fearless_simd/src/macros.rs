// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros publicly exported

#[cfg(feature = "std")]
#[macro_export]
macro_rules! simd_dispatch {
    (
        $( #[$meta:meta] )* $vis:vis
        $func:ident ( level $( , $arg:ident : $ty:ty $(,)? )* ) $( -> $ret:ty )?
        = $inner:ident
    ) => {
        $( #[$meta] )* $vis
        fn $func(level: $crate::Level $(, $arg: $ty )*) $( -> $ret )? {
            #[cfg(target_arch = "aarch64")]
            #[target_feature(enable = "neon")]
            #[inline]
            unsafe fn inner_neon(neon: $crate::aarch64::Neon $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( neon $( , $arg )* )
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            #[inline]
            unsafe fn inner_wasm_simd128(simd128: $crate::wasm32::WasmSimd128 $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( simd128 $( , $arg )* )
            }
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            #[target_feature(enable = "sse4.2")]
            #[inline]
            unsafe fn inner_sse4_2(sse4_2: $crate::x86::Sse4_2 $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( sse4_2 $( , $arg )* )
            }
            match level {
                Level::Fallback(fb) => {
                    $inner(fb $( , $arg )* )
                },
                #[cfg(target_arch = "aarch64")]
                Level::Neon(neon) => unsafe { inner_neon (neon $( , $arg )* ) }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                Level::WasmSimd128(wasm) => unsafe { inner_wasm_simd128 (wasm $( , $arg )* ) }
                #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
                Level::Sse4_2(sse4_2) => unsafe { inner_sse4_2(sse4_2 $( , $arg)* ) }
            }
        }
    };
}

#[cfg(not(feature = "std"))]
#[macro_export]
macro_rules! simd_dispatch {
    (
        $( #[$meta:meta] )* $vis:vis
        $func:ident ( level $( , $arg:ident : $ty:ty $(,)? )* ) $( -> $ret:ty )?
        = $inner:ident
    ) => {
        $( #[$meta] )* $vis
        fn $func(level: $crate::Level $(, $arg: $ty )*) $( -> $ret )? {
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            #[inline]
            unsafe fn inner_wasm_simd128(simd128: $crate::wasm32::WasmSimd128 $( , $arg: $ty )* ) $( -> $ret )? {
                $inner( simd128 $( , $arg )* )
            }
            match level {
                Level::Fallback(fb) => $inner(fb $( , $arg )* ),
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                Level::WasmSimd128(wasm) => unsafe { inner_wasm_simd128 (wasm $( , $arg )* ) }
            }
        }
    };
}
