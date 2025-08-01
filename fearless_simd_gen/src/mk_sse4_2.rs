// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::Arch;
use crate::arch::sse4_2::{Sse4_2, extend_intrinsic, op_suffix, pack_intrinsic, set1_intrinsic, cvt_intrinsic};
use crate::generic::{generic_combine, generic_op, generic_split};
use crate::ops::{
    OpSig, TyFlavor, load_interleaved_arg_ty, ops_for_type, reinterpret_ty,
    store_interleaved_arg_ty, valid_reinterpret,
};
use crate::types::{SIMD_TYPES, ScalarType, VecType, type_imports};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

#[derive(Clone, Copy)]
pub struct Level;

impl Level {
    fn name(self) -> &'static str {
        "Sse4_2"
    }

    fn token(self) -> TokenStream {
        let ident = Ident::new(self.name(), Span::call_site());
        quote! { #ident }
    }
}

pub fn mk_sse4_2_impl() -> TokenStream {
    let imports = type_imports();
    let simd_impl = mk_simd_impl();
    let ty_impl = mk_type_impl();

    quote! {
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
                                    let low = #extend(_mm_slli_si128(raw, 8)).simd_into(self);
                                    self.#combine(high, low)
                                }
                            }
                        }
                    }
                    "narrow" => {
                        let pack =
                            pack_intrinsic(scalar_bits, matches!(vec_ty.scalar, ScalarType::Int));
                        let split = format_ident!("split_{}", vec_ty.rust_name());
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                                let (a, b) = self.#split(a);
                                unsafe {
                                    #pack(a.into(), b.into()).simd_into(self)
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
                    }   else {
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
                    if scalar_bits == 8 {
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: u32) -> #ret_ty {
                                todo!()
                            }
                        }
                        // let extend = extend_intrinsic(vec_ty.scalar, 8, 16);
                        // let intrinsic = format_ident!("_mm_{op}_epi16");
                        // let narrow = format_ident!(
                        //     "narrow_{}",
                        //     VecType {
                        //         scalar: ScalarType::Unsigned,
                        //         scalar_bits: 16,
                        //         ..*vec_ty
                        //     }
                        //     .rust_name()
                        // );
                        // let combine = format_ident!(
                        //     "combine_{}",
                        //     VecType {
                        //         len: vec_ty.len / 2,
                        //         scalar_bits: 16,
                        //         ..*vec_ty
                        //     }
                        //     .rust_name()
                        // );
                        // let set1 = set1_intrinsic(vec_ty.scalar, 16);
                        // quote! {
                        //     #[inline(always)]
                        //     fn #method_ident(self, a: #ty<Self>, b: u32) -> #ret_ty {
                        //         unsafe {
                        //             let a1 = a.into();
                        //             let a2 = _mm_slli_si128(a1, 8);
                        //             let b = #set1(b as _);
                        //             self.#narrow(self.#combine(
                        //                 #intrinsic(#extend(a1), b).simd_into(self),
                        //                 #intrinsic(#extend(a2), b).simd_into(self),
                        //             ))
                        //         }
                        //     }
                        // }
                    } else {
                        let suffix = op_suffix(vec_ty.scalar, scalar_bits, false);
                        let intrinsic = format_ident!("_mm_{op}_{suffix}");
                        let set1 = set1_intrinsic(vec_ty.scalar, scalar_bits);
                        quote! {
                            #[inline(always)]
                            fn #method_ident(self, a: #ty<Self>, b: u32) -> #ret_ty {
                                unsafe { #intrinsic(a.into(), #set1(b as _)).simd_into(self) }
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
                                a + b * c
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
                OpSig::Compare => {
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>, b: #ty<Self>) -> #ret_ty {
                            todo!()
                        }
                    }
                }
                OpSig::Select => {
                    let mask_ty = vec_ty.mask_ty().rust();
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #mask_ty<Self>, b: #ty<Self>, c: #ty<Self>) -> #ret_ty {
                            todo!()
                        }
                    }
                }
                OpSig::Combine => generic_combine(vec_ty),
                OpSig::Split => generic_split(vec_ty),
                OpSig::Zip(zip1) => {
                    let op = if zip1 {
                        "lo"
                    }   else {
                        "hi"
                    };

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

                        quote! { unsafe { #intrinsic::<{ #mask }>(a.into(), b.into()).simd_into(self) } }
                    }   else {
                        match vec_ty.scalar_bits {
                            32 => {
                                let op = if select_even {
                                    "lo"
                                }   else {
                                    "hi"
                                };

                                let intrinsic = format_ident!("_mm_unpack{op}_epi64");

                                quote! {
                                      unsafe {
                                          let t1 = _mm_shuffle_epi32::<{0b11_01_10_00}>(a.into());
                                          let t2 = _mm_shuffle_epi32::<{0b11_01_10_00}>(b.into());
                                          #intrinsic(t1, t2).simd_into(self)
                                    }
                                }
                            }
                            16 | 8 => {
                                let mask = match (scalar_bits, select_even) {
                                    (8, true) => quote! { 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14  },
                                    (8, false) => quote! { 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15  },
                                    (16, true) => quote! { 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13 },
                                    (16, false) => quote! {  2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15 },
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
                            },
                            _ => quote! { todo!() }
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
                    let intrinsic = cvt_intrinsic(*vec_ty, VecType::new(scalar, scalar_bits, vec_ty.len));

                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, a: #ty<Self>) -> #ret_ty {
                            unsafe { #intrinsic(a.into()).simd_into(self) }
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
                            todo!()
                        }
                    }
                }
                OpSig::StoreInterleaved(block_size, count) => {
                    let arg = store_interleaved_arg_ty(block_size, count, vec_ty);
                    quote! {
                        #[inline(always)]
                        fn #method_ident(self, #arg) -> #ret_ty {
                            todo!()
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
        })
    }
    quote! {
        #( #result )*
    }
}
