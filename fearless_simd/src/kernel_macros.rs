// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[doc = "Creates a context where you can safely call intrinsics"]
#[doc = "available at the SIMD level named by the function's first argument."]
#[doc = ""]
#[doc = "This is useful if the portable abstractions are not enough, and you need to"]
#[doc = "use platform-specific intrinsics for parts of the computation."]
#[doc = ""]
#[doc = "The first argument must be a SIMD token written as `token: Neon`,"]
#[doc = "`token: WasmSimd128`, `token: Sse4_2`, or `token: Avx2`."]
#[doc = "The generated wrapper uses the corresponding `$crate::` token type in its"]
#[doc = "actual signature."]
#[doc = ""]
#[doc = "For levels with runtime-detected target features, the macro runs your body"]
#[doc = "inside an inner function annotated with the appropriate `#[target_feature]`"]
#[doc = "attributes. That makes platform-specific intrinsics from `core::arch` or"]
#[doc = "`std::arch` safe to call in the body, as long as they do not have safety"]
#[doc = "requirements beyond those target features."]
#[doc = ""]
#[doc = "## Example"]
#[doc = ""]
#[doc = "```rust"]
#[doc = "# #[allow(unused_imports)]"]
#[doc = "use fearless_simd::{i32x8, prelude::*};"]
#[doc = "#[cfg(target_arch = \"x86\")]"]
#[doc = "use std::arch::x86::{__m256i, _mm256_add_epi32};"]
#[doc = "#[cfg(target_arch = \"x86_64\")]"]
#[doc = "use std::arch::x86_64::{__m256i, _mm256_add_epi32};"]
#[doc = ""]
#[doc = "fearless_simd::kernel! {"]
#[doc = "    fn add_i32x8(avx2: Avx2, a: __m256i, b: __m256i) -> __m256i {"]
#[doc = "        _mm256_add_epi32(a, b)"]
#[doc = "    }"]
#[doc = "}"]
#[doc = ""]
#[doc = "# fn main() {"]
#[doc = "#[cfg(any(target_arch = \"x86\", target_arch = \"x86_64\"))]"]
#[doc = "if let Some(avx2) = fearless_simd::Level::new().as_avx2() {"]
#[doc = "    let a: i32x8<_> = [1, 2, 3, 4, 5, 6, 7, 8].simd_into(avx2);"]
#[doc = "    let b: i32x8<_> = [10, 20, 30, 40, 50, 60, 70, 80].simd_into(avx2);"]
#[doc = "    let sum: i32x8<_> = add_i32x8(avx2, a.into(), b.into()).simd_into(avx2);"]
#[doc = ""]
#[doc = "    assert_eq!(<[i32; 8]>::from(sum), [11, 22, 33, 44, 55, 66, 77, 88]);"]
#[doc = "}"]
#[doc = "# }"]
#[doc = "```"]
#[doc = ""]
#[doc = "See the [sRGB example] for an end-to-end use of kernel macros."]
#[doc = ""]
#[doc = "[sRGB example]: https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/srgb.rs"]
#[doc = ""]
#[doc = "## Limitations"]
#[doc = ""]
#[doc = "The macro only accepts a single plain, safe, non-generic function item with simple named parameters."]
#[doc = "However, the body of the function can be as complex as you like."]
#[doc = ""]
#[doc = "The SIMD token type must be written as a bare supported name:"]
#[doc = "literally `Neon`, `WasmSimd128`, `Sse4_2`, or `Avx2`. No paths or aliases."]
#[doc = ""]
#[doc = "For soundness, this macro only accepts safe functions."]
#[doc = ""]
#[doc = "```compile_fail"]
#[doc = "fearless_simd::kernel! {"]
#[doc = "    unsafe fn should_not_compile(avx2: Avx2) {}"]
#[doc = "}"]
#[macro_export]
macro_rules! kernel {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(
            $token:ident : $token_ty:ident $(, $arg:ident : $arg_ty:ty)* $(,)?
        ) $(-> $ret:ty)? {
            $($kernel_body:tt)*
        }
    ) => {
        $crate::__fearless_simd_kernel_dispatch! {
            $token_ty,
            $(#[$meta])*
            $vis fn $name(
                $token $(, $arg: $arg_ty)*
            ) $(-> $ret)? {
                $($kernel_body)*
            }
        }
    };

    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(
            $token:ident : $token_ty:ty $(, $arg:ident : $arg_ty:ty)* $(,)?
        ) $(-> $ret:ty)? {
            $($kernel_body:tt)*
        }
    ) => {
        compile_error!(concat!(
            "fearless_simd::kernel! expects its SIMD token argument type to be written as ",
            "one of `Neon`, `WasmSimd128`, `Sse4_2`, or `Avx2`; got `",
            stringify!($token_ty),
            "`",
        ));
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __fearless_simd_kernel_dispatch {
    (
        Neon,
        $($body:tt)*
    ) => {
        $crate::__fearless_simd_kernel_impl! {
            @cfg target_arch = "aarch64";
            @token_ty $crate::Neon;
            @kernel_attrs #[target_feature(enable = "neon")];
            $($body)*
        }
    };

    (
        WasmSimd128,
        $($body:tt)*
    ) => {
        $crate::__fearless_simd_kernel_impl! {
            @cfg all(target_arch = "wasm32", target_feature = "simd128");
            @token_ty $crate::WasmSimd128;
            @kernel_attrs;
            $($body)*
        }
    };

    (
        Sse4_2,
        $($body:tt)*
    ) => {
        $crate::__fearless_simd_kernel_impl! {
            @cfg any(target_arch = "x86", target_arch = "x86_64");
            @token_ty $crate::Sse4_2;
            @kernel_attrs #[target_feature(enable = "sse4.2,cmpxchg16b,popcnt")];
            $($body)*
        }
    };

    (
        Avx2,
        $($body:tt)*
    ) => {
        $crate::__fearless_simd_kernel_impl! {
            @cfg any(target_arch = "x86", target_arch = "x86_64");
            @token_ty $crate::Avx2;
            @kernel_attrs #[target_feature(
                enable = "avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,lzcnt,movbe,popcnt,xsave"
            )];
            $($body)*
        }
    };

    (
        $token_ty:ident,
        $($body:tt)*
    ) => {
        compile_error!(concat!(
            "fearless_simd::kernel! expects its SIMD token argument type to be written as ",
            "one of `Neon`, `WasmSimd128`, `Sse4_2`, or `Avx2`; got `",
            stringify!($token_ty),
            "`",
        ));
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __fearless_simd_kernel_impl {
    (
        @cfg $cfg:meta;
        @token_ty $token_ty:ty;
        @kernel_attrs $(#[$kernel_attr:meta])*;
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(
            $token:ident $(, $arg:ident : $arg_ty:ty)* $(,)?
        ) $(-> $ret:ty)? {
            $($kernel_body:tt)*
        }
    ) => {
        #[cfg($cfg)]
        $(#[$meta])*
        $vis fn $name(
            $token: $token_ty $(, $arg: $arg_ty)*
        ) $(-> $ret)? {
            #[inline] // can't use `#[inline(always)]` with target features
            $(#[$kernel_attr])*
            fn __fearless_simd_kernel(
                $token: $token_ty $(, $arg: $arg_ty)*
            ) $(-> $ret)? {
                let _ = $token;
                $($kernel_body)*
            }

            // SAFETY: the SIMD token proves that the required target features are available.
            #[allow(unused_unsafe)] // for WASM which has no target feature requirements and is safe to call
            unsafe { __fearless_simd_kernel($token $(, $arg)*) }
        }
    };
}
