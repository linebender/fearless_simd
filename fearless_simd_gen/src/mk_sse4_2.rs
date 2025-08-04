// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::Arch;
use crate::arch::sse4_2::{
    Sse4_2, cvt_intrinsic, extend_intrinsic, op_suffix, pack_intrinsic, set1_intrinsic,
    simple_intrinsic, unpack_intrinsic,
};
use crate::generic::{generic_combine, generic_op, generic_split};
use crate::ops::{
    OpSig, TyFlavor, load_interleaved_arg_ty, ops_for_type, reinterpret_ty,
    store_interleaved_arg_ty, valid_reinterpret,
};
use crate::types::{SIMD_TYPES, ScalarType, VecType, type_imports};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

#[derive(Clone, Copy)]
pub(crate) struct Level;

impl Level {
    fn name(self) -> &'static str {
        "Sse4_2"
    }

    fn token(self) -> TokenStream {
        let ident = Ident::new(self.name(), Span::call_site());
        quote! { #ident }
    }
}

pub(crate) fn mk_sse4_2_impl() -> TokenStream {
    let imports = type_imports();
    let simd_impl = mk_simd_impl();
    let ty_impl = mk_type_impl();

    quote! {
        // Until we have implemented all functions.
        #![expect(
            unused_variables,
            clippy::todo,
            reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
        )]

        use core::arch::x86_64::*;
        use core::ops::*;
        use crate::{seal::Seal, Level, Simd, SimdFrom, SimdInto};

        #imports

        /// The SIMD token for the "SSE 4.2" level.
        #[derive(Clone, Copy, Debug)]
        pub struct Sse4_2 {
            pub sse4_2: crate::core_arch::x86_64::Sse4_2,
        }

        impl Sse4_2 {
            /// Create a SIMD token.
            ///
            /// # Safety
            ///
            /// The SSE4.2 CPU feature must be available.
            #[inline]
            pub unsafe fn new_unchecked() -> Self {
                Sse4_2 {
                    sse4_2: unsafe { crate::core_arch::x86_64::Sse4_2::new_unchecked() },
                }
            }
        }

        impl Seal for Sse4_2 {}

        #simd_impl

        #ty_impl
    }
}

fn mk_simd_impl() -> TokenStream {
    let level_tok = Level.token();
    let mut methods = vec![];
    for vec_ty in SIMD_TYPES {
        let scalar_bits = vec_ty.scalar_bits;
        let ty_name = vec_ty.rust_name();
        let ty = vec_ty.rust();
        for (method, sig) in ops_for_type(vec_ty, true) {
            let b1 = (vec_ty.n_bits() > 128 && !matches!(method, "split" | "narrow"))
                || vec_ty.n_bits() > 256;

            let b2 = !matches!(method, "load_interleaved_128")
                && !matches!(method, "store_interleaved_128");

            if b1 && b2 {
                methods.push(generic_op(method, sig, vec_ty));
                continue;
            }

            let method_name = format!("{method}_{ty_name}");
            let method_ident = Ident::new(&method_name, Span::call_site());
            let ret_ty = sig.ret_ty(vec_ty, TyFlavor::SimdTrait);
            let method = match sig {
                OpSig::Splat => {
                    let scalar = vec_ty.scalar.rust(scalar_bits);
                    let intrinsic = set1_intrinsic(vec_ty.scalar, scalar_bits);
                    let cast = match vec_ty.scalar {
                        ScalarType::Unsigned => quote!(as _),
                        _ => quote!(),
                    };
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, val: #scalar) -> #ret_ty {
                            unsafe {
                                #intrinsic(val #cast).simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Compare => {
                    let args = [quote! { a.into() }, quote! { b.into() }];

                    let mut expr = if matches!(method, "simd_le" | "simd_ge")
                        && vec_ty.scalar != ScalarType::Float
                    {
                        let patched_method = match method {
                            "simd_le" => "simd_lt",
                            "simd_ge" => "simd_gt",
                            _ => method,
                        };
                        let expr = Sse4_2.expr(patched_method, vec_ty, &args);

                        let or_intrinsic = format_ident!("_mm_or_si128");

                        let eq_expr = Sse4_2.expr("simd_eq", vec_ty, &args);
                        quote! { #or_intrinsic(#expr, #eq_expr) }
                    } else {
                        Sse4_2.expr(method, vec_ty, &args)
                    };

                    if vec_ty.scalar == ScalarType::Float {
                        let suffix = op_suffix(vec_ty.scalar, scalar_bits, false);
                        let ident = format_ident!("_mm_cast{suffix}_si128");
                        expr = quote! { #ident(#expr) }
                    }

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            unsafe { #expr.simd_into(self) }
                        }
                    }
                }
                OpSig::Unary => match method {
                    "fract" => {
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                a - a.trunc()
                            }
                        }
                    }
                    "not" => {
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                a ^ !0
                            }
                        }
                    }
                    _ => {
                        let args = [quote! { a.into() }];
                        let expr = Sse4_2.expr(method, vec_ty, &args);
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                unsafe { #expr.simd_into(self) }
                            }
                        }
                    }
                },
                OpSig::WidenNarrow(t) => match method {
                    "widen" => {
                        let extend = extend_intrinsic(vec_ty.scalar, scalar_bits, t.scalar_bits);
                        let combine = format_ident!(
                            "combine_{}",
                            VecType {
                                len: vec_ty.len / 2,
                                scalar_bits: scalar_bits * 2,
                                ..*vec_ty
                            }
                            .rust_name()
                        );
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                unsafe {
                                    let raw = a.into();
                                    let high = #extend(raw).simd_into(self);
                                    // Shift by 8 since we want to get the higher part into the
                                    // lower position.
                                    let low = #extend(_mm_srli_si128::<8>(raw)).simd_into(self);
                                    self.#combine(high, low)
                                }
                            }
                        }
                    }
                    "narrow" => {
                        let mask = set1_intrinsic(vec_ty.scalar, scalar_bits);
                        let pack =
                            pack_intrinsic(scalar_bits, matches!(vec_ty.scalar, ScalarType::Int));
                        let split = format_ident!("split_{}", vec_ty.rust_name());
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                let (a, b) = self.#split(a);
                                unsafe {
                                    // Note that SSE4.2 only has an intrinsic for saturating cast,
                                    // but not wrapping.
                                    let mask = #mask(0xFF);
                                    let lo_masked = _mm_and_si128(a.into(), mask);
                                    let hi_masked = _mm_and_si128(b.into(), mask);
                                    let result = #pack(lo_masked, hi_masked);
                                    result.simd_into(self)
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                },
                OpSig::Binary => {
                    if method == "mul" && (vec_ty.scalar_bits == 8 || vec_ty.scalar_bits == 16) {
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                                todo!()
                            }
                        }
                    } else {
                        let args = [quote! { a.into() }, quote! { b.into() }];
                        let expr = Sse4_2.expr(method, vec_ty, &args);
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                                unsafe { #expr.simd_into(self) }
                            }
                        }
                    }
                }
                OpSig::Shift => {
                    let op = match vec_ty.scalar {
                        ScalarType::Unsigned => "srl",
                        ScalarType::Int => "sra",
                        _ => unreachable!(),
                    };
                    let suffix = op_suffix(vec_ty.scalar, scalar_bits.max(16), false);
                    let shift_intrinsic = format_ident!("_mm_{op}_{suffix}");

                    if scalar_bits == 8 {
                        // SSE doesn't have shifting for 8-bit, so we first convert into
                        // 16 bit, shift, and then back to 8-bit

                        let unpack_hi = unpack_intrinsic(ScalarType::Int, 8, false);
                        let unpack_lo = unpack_intrinsic(ScalarType::Int, 8, true);

                        let extend_expr = |expr| match vec_ty.scalar {
                            ScalarType::Unsigned => quote! {
                                #expr(val, _mm_setzero_si128())
                            },
                            ScalarType::Int => quote! {
                                 #expr(val, _mm_cmplt_epi8(val, _mm_setzero_si128()))
                            },
                            _ => unimplemented!(),
                        };

                        let extend_intrinsic_lo = extend_expr(unpack_lo);
                        let extend_intrinsic_hi = extend_expr(unpack_hi);
                        let pack_intrinsic = pack_intrinsic(16, vec_ty.scalar == ScalarType::Int);

                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: u32) -> #ret_ty {
                                unsafe {
                                    let val = a.into();
                                    let shift_count = _mm_cvtsi32_si128(b as i32);

                                    let lo_16 = #extend_intrinsic_lo;
                                    let hi_16 = #extend_intrinsic_hi;

                                    let lo_shifted = #shift_intrinsic(lo_16, shift_count);
                                    let hi_shifted = #shift_intrinsic(hi_16, shift_count);

                                    #pack_intrinsic(lo_shifted, hi_shifted).simd_into(self)
                                }
                            }
                        }
                    } else {
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: u32) -> #ret_ty {
                                unsafe { #shift_intrinsic(a.into(), _mm_cvtsi32_si128(b as _)).simd_into(self) }
                            }
                        }
                    }
                }
                OpSig::Ternary => match method {
                    "madd" => {
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                                a + b * c
                            }
                        }
                    }
                    "msub" => {
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                                a - b * c
                            }
                        }
                    }
                    _ => {
                        let args = [
                            quote! { a.into() },
                            quote! { b.into() },
                            quote! { c.into() },
                        ];

                        let expr = Sse4_2.expr(method, vec_ty, &args);
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                               #expr.simd_into(self)
                            }
                        }
                    }
                },
                OpSig::Select => {
                    let mask_ty = vec_ty.mask_ty().rust();

                    let expr = if vec_ty.scalar == ScalarType::Float {
                        let suffix = op_suffix(vec_ty.scalar, scalar_bits, false);
                        let (i1, i2, i3, i4) = (
                            format_ident!("_mm_castsi128_{suffix}"),
                            format_ident!("_mm_or_{suffix}"),
                            format_ident!("_mm_and_{suffix}"),
                            format_ident!("_mm_andnot_{suffix}"),
                        );
                        quote! {
                            let mask = #i1(a.into());

                            #i2(
                                #i3(mask, b.into()),
                                #i4(mask, c.into())
                            )
                        }
                    } else {
                        quote! {
                            _mm_or_si128(
                                _mm_and_si128(a.into(), b.into()),
                                _mm_andnot_si128(a.into(), c.into())
                            )
                        }
                    };

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #mask_ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                           unsafe {
                                 #expr.simd_into(self)
                            }
                        }
                    }
                }
                OpSig::Combine => generic_combine(vec_ty),
                OpSig::Split => generic_split(vec_ty),
                OpSig::Zip(zip1) => {
                    let op = if zip1 { "lo" } else { "hi" };

                    let suffix = op_suffix(vec_ty.scalar, scalar_bits, false);
                    let intrinsic = format_ident!("_mm_unpack{op}_{suffix}");

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                           unsafe {  #intrinsic(a.into(), b.into()).simd_into(self) }
                        }
                    }
                }
                OpSig::Unzip(select_even) => {
                    let expr = if vec_ty.scalar == ScalarType::Float {
                        let suffix = op_suffix(vec_ty.scalar, scalar_bits, false);
                        let intrinsic = format_ident!("_mm_shuffle_{suffix}");

                        let mask = match (vec_ty.scalar_bits, select_even) {
                            (32, true) => quote! { 0b10_00_10_00 },
                            (32, false) => quote! { 0b11_01_11_01 },
                            (64, true) => quote! { 0b00 },
                            (64, false) => quote! { 0b11 },
                            _ => unimplemented!(),
                        };

                        quote! { unsafe { #intrinsic::<#mask>(a.into(), b.into()).simd_into(self) } }
                    } else {
                        match vec_ty.scalar_bits {
                            32 => {
                                let op = if select_even { "lo" } else { "hi" };

                                let intrinsic = format_ident!("_mm_unpack{op}_epi64");

                                quote! {
                                      unsafe {
                                          let t1 = _mm_shuffle_epi32::<0b11_01_10_00>(a.into());
                                          let t2 = _mm_shuffle_epi32::<0b11_01_10_00>(b.into());
                                          #intrinsic(t1, t2).simd_into(self)
                                    }
                                }
                            }
                            16 | 8 => {
                                let mask = match (scalar_bits, select_even) {
                                    (8, true) => {
                                        quote! { 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14  }
                                    }
                                    (8, false) => {
                                        quote! { 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15  }
                                    }
                                    (16, true) => {
                                        quote! { 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13 }
                                    }
                                    (16, false) => {
                                        quote! {  2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15 }
                                    }
                                    _ => unreachable!(),
                                };

                                quote! {
                                    unsafe {
                                        let mask = _mm_setr_epi8(#mask);

                                        let t1 = _mm_shuffle_epi8(a.into(), mask);
                                        let t2 = _mm_shuffle_epi8(b.into(), mask);
                                        _mm_unpacklo_epi64(t1, t2).simd_into(self)
                                    }
                                }
                            }
                            _ => quote! { todo!() },
                        }
                    };

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            #expr
                        }
                    }
                }
                OpSig::Cvt(scalar, scalar_bits) => {
                    // IMPORTANT TODO: for f32 to u32, we are currently converting it to i32 instead
                    // of u32. We need to properly polyfill this.
                    let cvt_intrinsic =
                        cvt_intrinsic(*vec_ty, VecType::new(scalar, scalar_bits, vec_ty.len));

                    let mut expr = quote! { a.into() };

                    if vec_ty.scalar == ScalarType::Float {
                        let floor_intrinsic =
                            simple_intrinsic("floor", vec_ty.scalar, vec_ty.scalar_bits);
                        expr = quote! { #floor_intrinsic(#expr) };
                    }

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            unsafe { #cvt_intrinsic(#expr).simd_into(self) }
                        }
                    }
                }
                OpSig::Reinterpret(scalar, scalar_bits) => {
                    if valid_reinterpret(vec_ty, scalar, scalar_bits) {
                        let to_ty = reinterpret_ty(vec_ty, scalar, scalar_bits).rust();

                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                #to_ty {
                                    val: bytemuck::cast(a.val),
                                    simd: a.simd,
                                }
                            }
                        }
                    } else {
                        quote! {}
                    }
                }
                OpSig::LoadInterleaved(block_size, count) => {
                    let arg = load_interleaved_arg_ty(block_size, count, vec_ty);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, #arg) -> #ret_ty {
                            crate::Fallback::new().#method_ident(src).val.simd_into(self)
                        }
                    }
                }
                OpSig::StoreInterleaved(block_size, count) => {
                    let arg = store_interleaved_arg_ty(block_size, count, vec_ty);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, #arg) -> #ret_ty {
                            let fb = crate::Fallback::new();
                            fb.#method_ident(a.val.simd_into(fb), dest)
                        }
                    }
                }
            };
            methods.push(method);
        }
    }
    // Note: the `vectorize` implementation is pretty boilerplate and should probably
    // be factored out for DRY.
    quote! {
        impl Simd for #level_tok {
            type f32s = f32x4<Self>;
            type u8s = u8x16<Self>;
            type i8s = i8x16<Self>;
            type u16s = u16x8<Self>;
            type i16s = i16x8<Self>;
            type u32s = u32x4<Self>;
            type i32s = i32x4<Self>;
            type mask8s = mask8x16<Self>;
            type mask16s = mask16x8<Self>;
            type mask32s = mask32x4<Self>;
            #[inline(always)]
            fn level(self) -> Level {
                Level::#level_tok(self)
            }

            #[inline]
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                f()
            }

            #( #methods )*
        }
    }
}

fn mk_type_impl() -> TokenStream {
    let mut result = vec![];
    for ty in SIMD_TYPES {
        let n_bits = ty.n_bits();
        if n_bits != 128 {
            continue;
        }
        let simd = ty.rust();
        let arch = Sse4_2.arch_ty(ty);
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
