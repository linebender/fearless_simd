// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::x86::{
    self, coarse_type, extend_intrinsic, intrinsic_ident, pack_intrinsic, set1_intrinsic,
    simple_intrinsic,
};
use crate::generic::{generic_combine, generic_op, generic_split, scalar_binary};
use crate::mk_sse4_2;
use crate::ops::{Op, OpSig, ops_for_type};
use crate::types::{SIMD_TYPES, ScalarType, VecType, type_imports};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

#[derive(Clone, Copy)]
pub(crate) struct Level;

impl Level {
    fn name(self) -> &'static str {
        "Avx2"
    }

    fn token(self) -> TokenStream {
        let ident = Ident::new(self.name(), Span::call_site());
        quote! { #ident }
    }
}

pub(crate) fn mk_avx2_impl() -> TokenStream {
    let imports = type_imports();
    let simd_impl = mk_simd_impl();
    let ty_impl = mk_type_impl();

    quote! {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        use core::ops::*;
        use crate::{seal::Seal, Level, Simd, SimdFrom, SimdInto};

        #imports

        /// The SIMD token for the "AVX2" and "FMA" level.
        #[derive(Clone, Copy, Debug)]
        pub struct Avx2 {
            pub avx2: crate::core_arch::x86::Avx2,
        }

        impl Avx2 {
            /// Create a SIMD token.
            ///
            /// # Safety
            ///
            /// The AVX2 and FMA CPU feature must be available.
            #[inline]
            pub const unsafe fn new_unchecked() -> Self {
                Avx2 {
                    avx2: unsafe { crate::core_arch::x86::Avx2::new_unchecked() },
                }
            }
        }

        impl Seal for Avx2 {}

        #simd_impl

        #ty_impl
    }
}

fn mk_simd_impl() -> TokenStream {
    let level_tok = Level.token();
    let mut methods = vec![];
    for vec_ty in SIMD_TYPES {
        for Op { method, sig, .. } in ops_for_type(vec_ty) {
            let too_wide = (vec_ty.n_bits() > 256 && !matches!(method, "split" | "narrow"))
                || vec_ty.n_bits() > 512;

            let acceptable_wide_op = matches!(method, "load_interleaved_128")
                || matches!(method, "store_interleaved_128");

            if too_wide && !acceptable_wide_op {
                methods.push(generic_op(method, sig, vec_ty));
                continue;
            }

            let method = make_method(method, sig, vec_ty);

            methods.push(method);
        }
    }

    // Note: the `vectorize` implementation is pretty boilerplate and should probably
    // be factored out for DRY.
    quote! {
        impl Simd for #level_tok {
            type f32s = f32x8<Self>;
            type f64s = f64x4<Self>;
            type u8s = u8x32<Self>;
            type i8s = i8x32<Self>;
            type u16s = u16x16<Self>;
            type i16s = i16x16<Self>;
            type u32s = u32x8<Self>;
            type i32s = i32x8<Self>;
            type mask8s = mask8x32<Self>;
            type mask16s = mask16x16<Self>;
            type mask32s = mask32x8<Self>;
            type mask64s = mask64x4<Self>;
            #[inline(always)]
            fn level(self) -> Level {
                Level::#level_tok(self)
            }

            #[inline]
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                #[target_feature(enable = "avx2,fma")]
                #[inline]
                unsafe fn vectorize_avx2<F: FnOnce() -> R, R>(f: F) -> R {
                    f()
                }
                unsafe { vectorize_avx2(f) }
            }

            #( #methods )*
        }
    }
}

fn mk_type_impl() -> TokenStream {
    let mut result = vec![];
    for ty in SIMD_TYPES {
        let n_bits = ty.n_bits();
        if n_bits != 256 {
            continue;
        }
        let simd = ty.rust();
        let arch = x86::arch_ty(ty);
        result.push(quote! {
            impl<S: Simd> SimdFrom<#arch, S> for #simd<S> {
                #[inline(always)]
                fn simd_from(arch: #arch, simd: S) -> Self {
                    Self {
                        val: unsafe { core::mem::transmute(arch) },
                        simd
                    }
                }
            }
            impl<S: Simd> From<#simd<S>> for #arch {
                #[inline(always)]
                fn from(value: #simd<S>) -> Self {
                    unsafe { core::mem::transmute(value.val) }
                }
            }
        });
    }
    quote! {
        #( #result )*
    }
}

fn make_method(method: &str, sig: OpSig, vec_ty: &VecType) -> TokenStream {
    let scalar_bits = vec_ty.scalar_bits;
    let ty_name = vec_ty.rust_name();
    let method_name = format!("{method}_{ty_name}");
    let method_ident = Ident::new(&method_name, Span::call_site());
    let ret_ty = sig.simd_impl_ret_ty(vec_ty);
    let args = sig.simd_trait_args(vec_ty);
    let method_sig = quote! {
        #[inline(always)]
        fn #method_ident(#args) -> #ret_ty
    };

    if method == "shrv" && scalar_bits < 32 {
        return scalar_binary(&method_ident, quote!(core::ops::Shr::shr), vec_ty);
    }

    match sig {
        OpSig::Splat => mk_sse4_2::handle_splat(method_sig, vec_ty),
        OpSig::Compare => mk_sse4_2::handle_compare(method_sig, method, vec_ty),
        OpSig::Unary => mk_sse4_2::handle_unary(method_sig, method, vec_ty),
        OpSig::WidenNarrow { target_ty } => {
            handle_widen_narrow(method_sig, method, vec_ty, target_ty)
        }
        OpSig::Binary => mk_sse4_2::handle_binary(method_sig, method, vec_ty),
        OpSig::Shift => mk_sse4_2::handle_shift(method_sig, method, vec_ty),
        OpSig::Ternary => match method {
            "madd" => {
                let intrinsic = simple_intrinsic("fmadd", vec_ty);
                quote! {
                    #method_sig {
                        unsafe { #intrinsic(a.into(), b.into(), c.into()).simd_into(self) }
                    }
                }
            }
            "msub" => {
                let intrinsic = simple_intrinsic("fmsub", vec_ty);
                quote! {
                    #method_sig {
                        unsafe { #intrinsic(a.into(), b.into(), c.into()).simd_into(self) }
                    }
                }
            }
            _ => mk_sse4_2::handle_ternary(method_sig, &method_ident, method, vec_ty),
        },
        OpSig::Select => mk_sse4_2::handle_select(method_sig, vec_ty),
        OpSig::Combine { combined_ty } => handle_combine(method_sig, vec_ty, &combined_ty),
        OpSig::Split { half_ty } => handle_split(method_sig, vec_ty, &half_ty),
        OpSig::Zip { select_low } => mk_sse4_2::handle_zip(method_sig, vec_ty, select_low),
        OpSig::Unzip { select_even } => mk_sse4_2::handle_unzip(method_sig, vec_ty, select_even),
        OpSig::Cvt {
            target_ty,
            scalar_bits,
        } => mk_sse4_2::handle_cvt(method_sig, vec_ty, target_ty, scalar_bits),
        OpSig::Reinterpret {
            target_ty,
            scalar_bits,
        } => mk_sse4_2::handle_reinterpret(method_sig, vec_ty, target_ty, scalar_bits),
        OpSig::MaskReduce {
            quantifier,
            condition,
        } => mk_sse4_2::handle_mask_reduce(method_sig, vec_ty, quantifier, condition),
        OpSig::LoadInterleaved {
            block_size,
            block_count,
        } => mk_sse4_2::handle_load_interleaved(method_sig, vec_ty, block_size, block_count),
        OpSig::StoreInterleaved {
            block_size,
            block_count,
        } => mk_sse4_2::handle_store_interleaved(method_sig, vec_ty, block_size, block_count),
    }
}

pub(crate) fn handle_split(
    method_sig: TokenStream,
    vec_ty: &VecType,
    half_ty: &VecType,
) -> TokenStream {
    if vec_ty.n_bits() == 256 {
        let extract_op = match vec_ty.scalar {
            ScalarType::Float => "extractf128",
            _ => "extracti128",
        };
        let extract_intrinsic = intrinsic_ident(extract_op, coarse_type(vec_ty), 256);
        quote! {
            #method_sig {
                unsafe {
                    (
                        #extract_intrinsic::<0>(a.into()).simd_into(self),
                        #extract_intrinsic::<1>(a.into()).simd_into(self),
                    )
                }
            }
        }
    } else {
        generic_split(vec_ty, half_ty)
    }
}

pub(crate) fn handle_combine(
    method_sig: TokenStream,
    vec_ty: &VecType,
    combined_ty: &VecType,
) -> TokenStream {
    if vec_ty.n_bits() == 128 {
        let suffix = match (vec_ty.scalar, vec_ty.scalar_bits) {
            (ScalarType::Float, 32) => "m128",
            (ScalarType::Float, 64) => "m128d",
            _ => "m128i",
        };
        let set_intrinsic = intrinsic_ident("setr", suffix, 256);
        quote! {
            #method_sig {
                unsafe {
                    #set_intrinsic(a.into(), b.into()).simd_into(self)
                }
            }
        }
    } else {
        generic_combine(vec_ty, combined_ty)
    }
}

pub(crate) fn handle_widen_narrow(
    method_sig: TokenStream,
    method: &str,
    vec_ty: &VecType,
    target_ty: VecType,
) -> TokenStream {
    let expr = match method {
        "widen" => {
            let dst_width = target_ty.n_bits();
            match (dst_width, vec_ty.n_bits()) {
                (256, 128) => {
                    let extend = extend_intrinsic(
                        vec_ty.scalar,
                        vec_ty.scalar_bits,
                        target_ty.scalar_bits,
                        dst_width,
                    );
                    quote! {
                        unsafe {
                            #extend(a.into()).simd_into(self)
                        }
                    }
                }
                (512, 256) => {
                    let extend = extend_intrinsic(
                        vec_ty.scalar,
                        vec_ty.scalar_bits,
                        target_ty.scalar_bits,
                        vec_ty.n_bits(),
                    );
                    let combine = format_ident!(
                        "combine_{}",
                        VecType {
                            len: vec_ty.len / 2,
                            scalar_bits: vec_ty.scalar_bits * 2,
                            ..*vec_ty
                        }
                        .rust_name()
                    );
                    let split = format_ident!("split_{}", vec_ty.rust_name());
                    quote! {
                        unsafe {
                            let (a0, a1) = self.#split(a);
                            let high = #extend(a0.into()).simd_into(self);
                            let low = #extend(a1.into()).simd_into(self);
                            self.#combine(high, low)
                        }
                    }
                }
                _ => unimplemented!(),
            }
        }
        "narrow" => {
            let dst_width = target_ty.n_bits();
            match (dst_width, vec_ty.n_bits()) {
                (128, 256) => {
                    let mask = match target_ty.scalar_bits {
                        8 => {
                            quote! { 0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1 }
                        }
                        _ => unimplemented!(),
                    };
                    quote! {
                        unsafe {
                            let mask = _mm256_setr_epi8(#mask, #mask);

                            let shuffled = _mm256_shuffle_epi8(a.into(), mask);
                            let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(shuffled);

                            _mm256_castsi256_si128(packed).simd_into(self)
                        }
                    }
                }
                (256, 512) => {
                    let mask = set1_intrinsic(&VecType::new(
                        vec_ty.scalar,
                        vec_ty.scalar_bits,
                        vec_ty.len / 2,
                    ));
                    let pack = pack_intrinsic(
                        vec_ty.scalar_bits,
                        matches!(vec_ty.scalar, ScalarType::Int),
                        target_ty.n_bits(),
                    );
                    let split = format_ident!("split_{}", vec_ty.rust_name());
                    quote! {
                        let (a, b) = self.#split(a);
                        unsafe {
                            // Note that AVX2 only has an intrinsic for saturating cast,
                            // but not wrapping.
                            let mask = #mask(0xFF);
                            let lo_masked = _mm256_and_si256(a.into(), mask);
                            let hi_masked = _mm256_and_si256(b.into(), mask);
                            // The 256-bit version of packus_epi16 operates lane-wise, so we need to arrange things
                            // properly afterwards.
                            let result = _mm256_permute4x64_epi64::<0b_11_01_10_00>(#pack(lo_masked, hi_masked));
                            result.simd_into(self)
                        }
                    }
                }
                _ => unimplemented!(),
            }
        }
        _ => unreachable!(),
    };

    quote! {
        #method_sig {
            #expr
        }
    }
}
