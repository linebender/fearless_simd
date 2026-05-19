// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// Creates a context where you can safely call intrinsics
/// available at the SIMD level named by the function's first argument.
///
/// This is useful if the portable abstractions are not enough, and you need to
/// use platform-specific intrinsics for parts of the computation.
///
/// The first argument must be a SIMD token written as `token: Neon`,
/// `token: WasmSimd128`, `token: Sse4_2`, or `token: Avx2`.
///
/// For levels with runtime-detected target features, the macro runs your body
/// inside an inner function annotated with the appropriate `#[target_feature]`
/// attributes. That makes platform-specific intrinsics from `core::arch` or
/// `std::arch` safe to call in the body, as long as they do not have safety
/// requirements beyond those target features.
///
/// ## Example
///
/// ```rust
/// # #[allow(unused_imports)]
/// use fearless_simd::{i32x8, prelude::*};
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::{__m256i, _mm256_add_epi32};
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::{__m256i, _mm256_add_epi32};
///
/// fearless_simd::kernel! {
///     fn add_i32x8(avx2: Avx2, a: __m256i, b: __m256i) -> __m256i {
///         _mm256_add_epi32(a, b)
///     }
/// }
///
/// # fn main() {
/// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// if let Some(avx2) = fearless_simd::Level::new().as_avx2() {
///     let a: i32x8<_> = [1, 2, 3, 4, 5, 6, 7, 8].simd_into(avx2);
///     let b: i32x8<_> = [10, 20, 30, 40, 50, 60, 70, 80].simd_into(avx2);
///     let sum: i32x8<_> = add_i32x8(avx2, a.into(), b.into()).simd_into(avx2);
///
///     assert_eq!(<[i32; 8]>::from(sum), [11, 22, 33, 44, 55, 66, 77, 88]);
/// }
/// # }
/// ```
///
/// See the [sRGB example] for an end-to-end use of kernel macros.
///
/// [sRGB example]: https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/srgb.rs
///
/// ## Limitations
///
/// The macro only accepts a single plain, safe, non-generic function item with simple named parameters.
/// However, the body of the function can be as complex as you like.
///
/// The SIMD token type must be written as a bare supported name:
/// literally `Neon`, `WasmSimd128`, `Sse4_2`, or `Avx2`. No paths or aliases.
///
/// For soundness, this macro only accepts safe functions.
///
/// ```compile_fail
/// fearless_simd::kernel! {
///     unsafe fn should_not_compile(avx2: Avx2) {}
/// }
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
            #[allow(unused_unsafe, reason = "for WASM which has no target feature requirements and is safe to call")]
            unsafe { __fearless_simd_kernel($token $(, $arg)*) }
        }
    };
}

#[cfg(test)]
mod tests {
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "x86",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    use crate::prelude::*;

    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::{float32x4_t, vaddq_f32};
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    use core::arch::wasm32::{f32x4_add, v128};
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{__m256i, _mm256_add_epi32};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{__m256i, _mm256_add_epi32};

    crate::kernel! {
        fn add_f32x4_neon(neon: Neon, a: float32x4_t, b: float32x4_t) -> float32x4_t {
            vaddq_f32(a, b)
        }
    }

    crate::kernel! {
        fn add_f32x4_wasm(wasm: WasmSimd128, a: v128, b: v128) -> v128 {
            f32x4_add(a, b)
        }
    }

    crate::kernel! {
        fn add_i32x8_avx2(avx2: Avx2, a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi32(a, b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn kernel_instantiates_for_neon() {
        let Some(neon) = crate::Level::new().as_neon() else {
            return;
        };

        let a: crate::f32x4<_> = [1.0, 2.0, 3.0, 4.0].simd_into(neon);
        let b: crate::f32x4<_> = [10.0, 20.0, 30.0, 40.0].simd_into(neon);
        let sum: crate::f32x4<_> = add_f32x4_neon(neon, a.into(), b.into()).simd_into(neon);

        assert_eq!(
            <[f32; 4]>::from(sum),
            [11.0, 22.0, 33.0, 44.0],
            "`kernel!` should instantiate a working NEON kernel"
        );
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn kernel_instantiates_for_wasm_simd128() {
        let wasm = crate::Level::new()
            .as_wasm_simd128()
            .expect("WASM SIMD128 should be available when +simd128 is enabled");

        let a: crate::f32x4<_> = [1.0, 2.0, 3.0, 4.0].simd_into(wasm);
        let b: crate::f32x4<_> = [10.0, 20.0, 30.0, 40.0].simd_into(wasm);
        let sum: crate::f32x4<_> = add_f32x4_wasm(wasm, a.into(), b.into()).simd_into(wasm);

        assert_eq!(
            <[f32; 4]>::from(sum),
            [11.0, 22.0, 33.0, 44.0],
            "`kernel!` should instantiate a working WASM SIMD128 kernel"
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn kernel_instantiates_for_avx2() {
        let Some(avx2) = crate::Level::new().as_avx2() else {
            return;
        };

        let a: crate::i32x8<_> = [1, 2, 3, 4, 5, 6, 7, 8].simd_into(avx2);
        let b: crate::i32x8<_> = [10, 20, 30, 40, 50, 60, 70, 80].simd_into(avx2);
        let sum: crate::i32x8<_> = add_i32x8_avx2(avx2, a.into(), b.into()).simd_into(avx2);

        assert_eq!(
            <[i32; 8]>::from(sum),
            [11, 22, 33, 44, 55, 66, 77, 88],
            "`kernel!` should instantiate a working AVX2 kernel"
        );
    }
}
