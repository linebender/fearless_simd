// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::x86::{
    self, arch_ty, cast_ident, coarse_type, extend_intrinsic, float_compare_method,
    intrinsic_ident, op_suffix, pack_intrinsic, set1_intrinsic, simple_intrinsic,
    simple_sign_unaware_intrinsic, unpack_intrinsic,
};
use crate::generic::{
    generic_as_array, generic_block_combine, generic_block_split, generic_from_array,
    generic_from_bytes, generic_op, generic_op_name, generic_to_bytes, impl_arch_types,
    scalar_binary,
};
use crate::ops::{Op, OpSig, Quantifier, ops_for_type, valid_reinterpret};
use crate::types::{SIMD_TYPES, ScalarType, VecType, type_imports};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

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
    let arch_types_impl = impl_arch_types(Level.name(), 128, arch_ty);
    let simd_impl = mk_simd_impl();
    let ty_impl = mk_type_impl();

    quote! {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        use core::ops::*;
        use crate::{seal::Seal, arch_types::ArchTypes, Level, Simd, SimdFrom, SimdInto};

        #imports

        /// The SIMD token for the "SSE 4.2" level.
        #[derive(Clone, Copy, Debug)]
        pub struct Sse4_2 {
            pub sse4_2: crate::core_arch::x86::Sse4_2,
        }

        impl Sse4_2 {
            /// Create a SIMD token.
            ///
            /// # Safety
            ///
            /// The SSE4.2 CPU feature must be available.
            #[inline]
            pub const unsafe fn new_unchecked() -> Self {
                Sse4_2 {
                    sse4_2: unsafe { crate::core_arch::x86::Sse4_2::new_unchecked() },
                }
            }
        }

        impl Seal for Sse4_2 {}

        #arch_types_impl

        #simd_impl

        #ty_impl
    }
}

fn mk_simd_impl() -> TokenStream {
    let level_tok = Level.token();
    let mut methods = vec![];
    for vec_ty in SIMD_TYPES {
        for op in ops_for_type(vec_ty) {
            if op.sig.should_use_generic_op(vec_ty, 128) {
                methods.push(generic_op(&op, vec_ty));
                continue;
            }

            let method = make_method(op, vec_ty);

            methods.push(method);
        }
    }
    // Note: the `vectorize` implementation is pretty boilerplate and should probably
    // be factored out for DRY.
    quote! {
        impl Simd for #level_tok {
            type f32s = f32x4<Self>;
            type f64s = f64x2<Self>;
            type u8s = u8x16<Self>;
            type i8s = i8x16<Self>;
            type u16s = u16x8<Self>;
            type i16s = i16x8<Self>;
            type u32s = u32x4<Self>;
            type i32s = i32x4<Self>;
            type mask8s = mask8x16<Self>;
            type mask16s = mask16x8<Self>;
            type mask32s = mask32x4<Self>;
            type mask64s = mask64x2<Self>;
            #[inline(always)]
            fn level(self) -> Level {
                #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
                return Level::#level_tok(self);
                #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
                {
                    Level::baseline()
                }
            }

            #[inline]
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                #[target_feature(enable = "sse4.2")]
                #[inline]
                unsafe fn vectorize_sse4_2<F: FnOnce() -> R, R>(f: F) -> R {
                    f()
                }
                unsafe { vectorize_sse4_2(f) }
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
        let arch = x86::arch_ty(ty);
        result.push(quote! {
            impl<S: Simd> SimdFrom<#arch, S> for #simd<S> {
                #[inline(always)]
                fn simd_from(arch: #arch, simd: S) -> Self {
                    Self {
                        val: unsafe { core::mem::transmute_copy(&arch) },
                        simd
                    }
                }
            }
            impl<S: Simd> From<#simd<S>> for #arch {
                #[inline(always)]
                fn from(value: #simd<S>) -> Self {
                    unsafe { core::mem::transmute_copy(&value.val) }
                }
            }
        });
    }
    quote! {
        #( #result )*
    }
}

fn make_method(op: Op, vec_ty: &VecType) -> TokenStream {
    let Op { sig, method, .. } = op;
    let method_sig = op.simd_trait_method_sig(vec_ty);
    let method_sig = quote! {
        #[inline(always)]
        #method_sig
    };

    match sig {
        OpSig::Splat => handle_splat(method_sig, vec_ty),
        OpSig::Compare => handle_compare(method_sig, method, vec_ty),
        OpSig::Unary => handle_unary(method_sig, method, vec_ty),
        OpSig::WidenNarrow { target_ty } => {
            handle_widen_narrow(method_sig, method, vec_ty, target_ty)
        }
        OpSig::Binary => handle_binary(method_sig, method, vec_ty),
        OpSig::Shift => handle_shift(method_sig, method, vec_ty),
        OpSig::Ternary => handle_ternary(method_sig, method, vec_ty),
        OpSig::Select => handle_select(method_sig, vec_ty),
        OpSig::Combine { combined_ty } => generic_block_combine(method_sig, &combined_ty, 128),
        OpSig::Split { half_ty } => generic_block_split(method_sig, &half_ty, 128),
        OpSig::Zip { select_low } => handle_zip(method_sig, vec_ty, select_low),
        OpSig::Unzip { select_even } => handle_unzip(method_sig, vec_ty, select_even),
        OpSig::Cvt {
            target_ty,
            scalar_bits,
            precise,
        } => handle_cvt(method_sig, vec_ty, target_ty, scalar_bits, precise),
        OpSig::Reinterpret {
            target_ty,
            scalar_bits,
        } => handle_reinterpret(method_sig, vec_ty, target_ty, scalar_bits),
        OpSig::MaskReduce {
            quantifier,
            condition,
        } => handle_mask_reduce(method_sig, vec_ty, quantifier, condition),
        OpSig::LoadInterleaved {
            block_size,
            block_count,
        } => handle_load_interleaved(method_sig, vec_ty, block_size, block_count),
        OpSig::StoreInterleaved {
            block_size,
            block_count,
        } => handle_store_interleaved(method_sig, vec_ty, block_size, block_count),
        OpSig::FromArray { kind } => {
            generic_from_array(method_sig, vec_ty, kind, 128, |block_ty| {
                intrinsic_ident("loadu", coarse_type(block_ty), block_ty.n_bits())
            })
        }
        OpSig::AsArray { kind } => generic_as_array(method_sig, vec_ty, kind, 128, arch_ty),
        OpSig::FromBytes => generic_from_bytes(method_sig, vec_ty),
        OpSig::ToBytes => generic_to_bytes(method_sig, vec_ty),
    }
}

pub(crate) fn handle_splat(method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
    let intrinsic = set1_intrinsic(vec_ty);
    let cast = match vec_ty.scalar {
        ScalarType::Unsigned => quote!(as _),
        _ => quote!(),
    };
    quote! {
        #method_sig {
            unsafe {
                #intrinsic(val #cast).simd_into(self)
            }
        }
    }
}

pub(crate) fn handle_compare(
    method_sig: TokenStream,
    method: &str,
    vec_ty: &VecType,
) -> TokenStream {
    let args = [quote! { a.into() }, quote! { b.into() }];

    let expr = if vec_ty.scalar != ScalarType::Float {
        match method {
            "simd_le" | "simd_ge" => {
                let max_min = match method {
                    "simd_le" => "min",
                    "simd_ge" => "max",
                    _ => unreachable!(),
                };

                let eq_intrinsic = simple_sign_unaware_intrinsic("cmpeq", vec_ty);

                let max_min_expr = x86::expr(max_min, vec_ty, &args);
                quote! { #eq_intrinsic(#max_min_expr, a.into()) }
            }
            "simd_lt" | "simd_gt" => {
                let gt = simple_sign_unaware_intrinsic("cmpgt", vec_ty);

                if vec_ty.scalar == ScalarType::Unsigned {
                    // SSE4.2 only has signed GT/LT, but not unsigned.
                    let set = set1_intrinsic(vec_ty);
                    let sign = match vec_ty.scalar_bits {
                        8 => quote! { 0x80u8 },
                        16 => quote! { 0x8000u16 },
                        32 => quote! { 0x80000000u32 },
                        _ => unimplemented!(),
                    };
                    let xor_op = intrinsic_ident("xor", coarse_type(vec_ty), vec_ty.n_bits());
                    let args = if method == "simd_lt" {
                        quote! { b_signed, a_signed }
                    } else {
                        quote! { a_signed, b_signed }
                    };

                    quote! {
                        let sign_bit = #set(#sign as _);
                        let a_signed = #xor_op(a.into(), sign_bit);
                        let b_signed = #xor_op(b.into(), sign_bit);

                        #gt(#args)
                    }
                } else {
                    let args = if method == "simd_lt" {
                        quote! { b.into(), a.into() }
                    } else {
                        quote! { a.into(), b.into() }
                    };
                    quote! {
                        #gt(#args)
                    }
                }
            }
            "simd_eq" => x86::expr(method, vec_ty, &args),
            _ => unreachable!(),
        }
    } else {
        let compare_op = float_compare_method(method, vec_ty);
        let ident = cast_ident(
            ScalarType::Float,
            ScalarType::Mask,
            vec_ty.scalar_bits,
            vec_ty.scalar_bits,
            vec_ty.n_bits(),
        );
        quote! { #ident(#compare_op(a.into(), b.into())) }
    };

    quote! {
        #method_sig {
            unsafe { #expr.simd_into(self) }
        }
    }
}

pub(crate) fn handle_unary(method_sig: TokenStream, method: &str, vec_ty: &VecType) -> TokenStream {
    match method {
        "fract" => {
            let trunc_op = generic_op_name("trunc", vec_ty);
            quote! {
                #method_sig {
                    a - self.#trunc_op(a)
                }
            }
        }
        "not" => {
            quote! {
                #method_sig {
                    a ^ !0
                }
            }
        }
        _ => {
            let args = [quote! { a.into() }];
            let expr = x86::expr(method, vec_ty, &args);
            quote! {
                #method_sig {
                    unsafe { #expr.simd_into(self) }
                }
            }
        }
    }
}

pub(crate) fn handle_widen_narrow(
    method_sig: TokenStream,
    method: &str,
    vec_ty: &VecType,
    target_ty: VecType,
) -> TokenStream {
    match method {
        "widen" => {
            let extend = extend_intrinsic(
                vec_ty.scalar,
                vec_ty.scalar_bits,
                target_ty.scalar_bits,
                vec_ty.n_bits(),
            );
            let combine = generic_op_name(
                "combine",
                &vec_ty.reinterpret(vec_ty.scalar, vec_ty.scalar_bits * 2),
            );
            quote! {
                #method_sig {
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
            let split = generic_op_name("split", vec_ty);
            quote! {
                #method_sig {
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
    }
}

pub(crate) fn handle_binary(
    method_sig: TokenStream,
    method: &str,
    vec_ty: &VecType,
) -> TokenStream {
    let body = match method {
        "mul" if vec_ty.scalar_bits == 8 => {
            // https://stackoverflow.com/questions/8193601/sse-multiplication-16-x-uint8-t
            let mullo = intrinsic_ident("mullo", "epi16", vec_ty.n_bits());
            let set1 = intrinsic_ident("set1", "epi16", vec_ty.n_bits());
            let and = intrinsic_ident("and", coarse_type(vec_ty), vec_ty.n_bits());
            let or = intrinsic_ident("or", coarse_type(vec_ty), vec_ty.n_bits());
            let slli = intrinsic_ident("slli", "epi16", vec_ty.n_bits());
            let srli = intrinsic_ident("srli", "epi16", vec_ty.n_bits());
            quote! {
                unsafe {
                    let dst_even = #mullo(a.into(), b.into());
                    let dst_odd = #mullo(#srli::<8>(a.into()), #srli::<8>(b.into()));

                    #or(#slli(dst_odd, 8), #and(dst_even, #set1(0xFF))).simd_into(self)
                }
            }
        }
        // SSE2 has shift operations, but they shift every lane by the same amount, so we can't use them here.
        "shlv" => scalar_binary(quote!(core::ops::Shl::shl)),
        "shrv" => scalar_binary(quote!(core::ops::Shr::shr)),
        _ => {
            let args = [quote! { a.into() }, quote! { b.into() }];
            let expr = x86::expr(method, vec_ty, &args);
            quote! {
                unsafe { #expr.simd_into(self) }
            }
        }
    };

    quote! {
        #method_sig {
            #body
        }
    }
}

pub(crate) fn handle_shift(method_sig: TokenStream, method: &str, vec_ty: &VecType) -> TokenStream {
    let op = match (method, vec_ty.scalar) {
        ("shr", ScalarType::Unsigned) => "srl",
        ("shr", ScalarType::Int) => "sra",
        ("shl", _) => "sll",
        _ => unreachable!(),
    };
    let ty_bits = vec_ty.n_bits();
    let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits.max(16), false);
    let shift_intrinsic = intrinsic_ident(op, suffix, ty_bits);

    if vec_ty.scalar_bits == 8 {
        // SSE doesn't have shifting for 8-bit, so we first convert into
        // 16 bit, shift, and then back to 8-bit

        let unpack_hi = unpack_intrinsic(ScalarType::Int, 8, false, ty_bits);
        let unpack_lo = unpack_intrinsic(ScalarType::Int, 8, true, ty_bits);

        let set0 = intrinsic_ident("setzero", coarse_type(vec_ty), ty_bits);
        let extend_expr = |expr| match vec_ty.scalar {
            ScalarType::Unsigned => quote! {
                #expr(val, #set0())
            },
            ScalarType::Int => {
                let cmp_intrinsic = intrinsic_ident("cmpgt", "epi8", ty_bits);
                quote! {
                    #expr(val, #cmp_intrinsic(#set0(), val))
                }
            }
            _ => unimplemented!(),
        };

        let extend_intrinsic_lo = extend_expr(unpack_lo);
        let extend_intrinsic_hi = extend_expr(unpack_hi);
        let pack_intrinsic = pack_intrinsic(16, vec_ty.scalar == ScalarType::Int, ty_bits);

        quote! {
            #method_sig {
                unsafe {
                    let val = a.into();
                    let shift_count = _mm_cvtsi32_si128(shift as i32);

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
            #method_sig {
                unsafe { #shift_intrinsic(a.into(), _mm_cvtsi32_si128(shift as _)).simd_into(self) }
            }
        }
    }
}

pub(crate) fn handle_ternary(
    method_sig: TokenStream,
    method: &str,
    vec_ty: &VecType,
) -> TokenStream {
    match method {
        "mul_add" => {
            quote! {
                #method_sig {
                    a * b + c
                }
            }
        }
        "mul_sub" => {
            quote! {
                #method_sig {
                    a * b - c
                }
            }
        }
        _ => {
            let args = [
                quote! { a.into() },
                quote! { b.into() },
                quote! { c.into() },
            ];

            let expr = x86::expr(method, vec_ty, &args);
            quote! {
                #method_sig {
                   #expr.simd_into(self)
                }
            }
        }
    }
}

pub(crate) fn handle_select(method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
    // Our select ops' argument order is mask, a, b; Intel's intrinsics are b, a, mask
    let args = [
        quote! { c.into() },
        quote! { b.into() },
        match vec_ty.scalar {
            ScalarType::Float => {
                let ident = cast_ident(
                    ScalarType::Mask,
                    ScalarType::Float,
                    vec_ty.scalar_bits,
                    vec_ty.scalar_bits,
                    vec_ty.n_bits(),
                );
                quote! { #ident(a.into()) }
            }
            _ => quote! { a.into() },
        },
    ];
    let expr = x86::expr("select", vec_ty, &args);

    quote! {
        #method_sig {
            unsafe { #expr.simd_into(self) }
        }
    }
}

pub(crate) fn handle_zip(
    method_sig: TokenStream,
    vec_ty: &VecType,
    select_low: bool,
) -> TokenStream {
    let expr = match vec_ty.n_bits() {
        128 => {
            let op = if select_low { "unpacklo" } else { "unpackhi" };

            let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
            let unpack_intrinsic = intrinsic_ident(op, suffix, vec_ty.n_bits());
            quote! {
                unsafe {  #unpack_intrinsic(a.into(), b.into()).simd_into(self) }
            }
        }
        256 => {
            let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
            let lo = intrinsic_ident("unpacklo", suffix, vec_ty.n_bits());
            let hi = intrinsic_ident("unpackhi", suffix, vec_ty.n_bits());
            let shuffle_immediate = if select_low {
                quote! { 0b0010_0000 }
            } else {
                quote! { 0b0011_0001 }
            };

            let shuffle = intrinsic_ident(
                match vec_ty.scalar {
                    ScalarType::Float => "permute2f128",
                    _ => "permute2x128",
                },
                coarse_type(vec_ty),
                256,
            );

            quote! {
                unsafe {
                    let lo = #lo(a.into(), b.into());
                    let hi = #hi(a.into(), b.into());

                    #shuffle::<#shuffle_immediate>(lo, hi).simd_into(self)
                }
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

pub(crate) fn handle_unzip(
    method_sig: TokenStream,
    vec_ty: &VecType,
    select_even: bool,
) -> TokenStream {
    let expr = match (vec_ty.scalar, vec_ty.n_bits(), vec_ty.scalar_bits) {
        (ScalarType::Float, 128, _) => {
            // 128-bit shuffle of floats or doubles; there are built-in SSE intrinsics for this
            let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
            let intrinsic = intrinsic_ident("shuffle", suffix, vec_ty.n_bits());

            let mask = match (vec_ty.scalar_bits, select_even) {
                (32, true) => quote! { 0b10_00_10_00 },
                (32, false) => quote! { 0b11_01_11_01 },
                (64, true) => quote! { 0b00 },
                (64, false) => quote! { 0b11 },
                _ => unimplemented!(),
            };

            quote! { unsafe { #intrinsic::<#mask>(a.into(), b.into()).simd_into(self) } }
        }
        (ScalarType::Int | ScalarType::Mask | ScalarType::Unsigned, 128, 32) => {
            // 128-bit shuffle of 32-bit integers; unlike with floats, there is no single shuffle instruction that
            // combines two vectors
            let op = if select_even { "unpacklo" } else { "unpackhi" };
            let intrinsic = intrinsic_ident(op, "epi64", vec_ty.n_bits());

            quote! {
                    unsafe {
                        let t1 = _mm_shuffle_epi32::<0b11_01_10_00>(a.into());
                        let t2 = _mm_shuffle_epi32::<0b11_01_10_00>(b.into());
                        #intrinsic(t1, t2).simd_into(self)
                }
            }
        }
        (ScalarType::Int | ScalarType::Mask | ScalarType::Unsigned, 128, 16 | 8) => {
            // Separate out the even-indexed and odd-indexed elements
            let mask = match vec_ty.scalar_bits {
                8 => {
                    quote! { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 }
                }
                16 => {
                    quote! { 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 }
                }
                _ => unreachable!(),
            };
            let mask_reg = match vec_ty.n_bits() {
                128 => quote! { _mm_setr_epi8(#mask) },
                256 => quote! { _mm256_setr_epi8(#mask, #mask) },
                _ => unreachable!(),
            };
            let shuffle_epi8 = intrinsic_ident("shuffle", "epi8", vec_ty.n_bits());

            // Select either the low or high half of each one
            let op = if select_even { "unpacklo" } else { "unpackhi" };
            let unpack_epi64 = intrinsic_ident(op, "epi64", vec_ty.n_bits());

            quote! {
                unsafe {
                    let mask = #mask_reg;

                    let t1 = #shuffle_epi8(a.into(), mask);
                    let t2 = #shuffle_epi8(b.into(), mask);
                    #unpack_epi64(t1, t2).simd_into(self)
                }
            }
        }
        (_, 256, 64 | 32) => {
            // First we perform a lane-crossing shuffle to move the even-indexed elements of each input to the lower
            // half, and the odd-indexed ones to the upper half.
            // e.g. [0, 1, 2, 3, 4, 5, 6, 7] becomes [0, 2, 4, 6, 1, 3, 5, 7]).
            let low_shuffle_kind = match vec_ty.scalar_bits {
                32 => "permutevar8x32",
                64 => "permute4x64",
                _ => unreachable!(),
            };
            let low_shuffle_suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
            let low_shuffle_intrinsic = intrinsic_ident(low_shuffle_kind, low_shuffle_suffix, 256);
            let low_shuffle = |input_name: TokenStream| match vec_ty.scalar_bits {
                32 => {
                    quote! { #low_shuffle_intrinsic(#input_name, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)) }
                }
                64 => quote! { #low_shuffle_intrinsic::<0b11_01_10_00>(#input_name) },
                _ => unreachable!(),
            };
            let shuf_t1 = low_shuffle(quote! { a.into() });
            let shuf_t2 = low_shuffle(quote! { b.into() });

            // Then we combine the lower or upper halves.
            let high_shuffle = intrinsic_ident(
                match vec_ty.scalar {
                    ScalarType::Float => "permute2f128",
                    _ => "permute2x128",
                },
                coarse_type(vec_ty),
                256,
            );
            let high_shuffle_immediate = if select_even {
                quote! { 0b0010_0000 }
            } else {
                quote! { 0b0011_0001 }
            };

            quote! {
                unsafe {
                    let t1 = #shuf_t1;
                    let t2 = #shuf_t2;

                    #high_shuffle::<#high_shuffle_immediate>(t1, t2).simd_into(self)
                }
            }
        }
        (_, 256, 16 | 8) => {
            // Separate out the even-indexed and odd-indexed elements within each 128-bit lane
            let mask = match vec_ty.scalar_bits {
                8 => {
                    quote! { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 }
                }
                16 => {
                    quote! { 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 }
                }
                _ => unreachable!(),
            };

            // We then permute the even-indexed and odd-indexed blocks across lanes, and finally do a 2x128 permute to
            // select either the even- or odd-indexed elements
            let high_shuffle_immediate = if select_even {
                quote! { 0b0010_0000 }
            } else {
                quote! { 0b0011_0001 }
            };

            quote! {
                unsafe {
                    let mask = _mm256_setr_epi8(#mask, #mask);
                    let a_shuffled = _mm256_shuffle_epi8(a.into(), mask);
                    let b_shuffled = _mm256_shuffle_epi8(b.into(), mask);

                    let packed = _mm256_permute2x128_si256::<#high_shuffle_immediate>(
                        _mm256_permute4x64_epi64::<0b11_01_10_00>(a_shuffled),
                        _mm256_permute4x64_epi64::<0b11_01_10_00>(b_shuffled)
                    );

                    packed.simd_into(self)
                }
            }
        }
        _ => unimplemented!(),
    };

    quote! {
        #method_sig {
            #expr
        }
    }
}

pub(crate) fn handle_cvt(
    method_sig: TokenStream,
    vec_ty: &VecType,
    target_scalar: ScalarType,
    target_scalar_bits: usize,
    precise: bool,
) -> TokenStream {
    assert_eq!(
        vec_ty.scalar_bits, target_scalar_bits,
        "we currently only support converting between types of the same width"
    );
    let expr = match (vec_ty.scalar, target_scalar) {
        (ScalarType::Float, ScalarType::Int | ScalarType::Unsigned) => {
            let target_ty = vec_ty.reinterpret(target_scalar, target_scalar_bits);
            let max = simple_intrinsic("max", vec_ty);
            let set0 = intrinsic_ident("setzero", coarse_type(vec_ty), vec_ty.n_bits());
            let cmplt = float_compare_method("simd_lt", vec_ty);
            let cmpord = float_compare_method("ord", vec_ty);
            let set1_float = set1_intrinsic(vec_ty);
            let set1_int = set1_intrinsic(&target_ty);
            let movemask = simple_intrinsic("movemask", vec_ty);
            let all_ones = match (vec_ty.n_bits(), vec_ty.scalar_bits) {
                (128, 32) => quote! { 0b1111 },
                (256, 32) => quote! { 0b11111111 },
                _ => unimplemented!(),
            };
            let convert = simple_sign_unaware_intrinsic("cvttps", &target_ty);
            let cast_to_int = cast_ident(
                vec_ty.scalar,
                target_scalar,
                vec_ty.scalar_bits,
                vec_ty.scalar_bits,
                vec_ty.n_bits(),
            );
            let blend = intrinsic_ident("blendv", "epi8", vec_ty.n_bits());
            let and = intrinsic_ident("and", coarse_type(&target_ty), vec_ty.n_bits());
            let andnot = simple_intrinsic("andnot", vec_ty);
            let add_int = simple_sign_unaware_intrinsic("add", &target_ty);
            let sub_float = simple_intrinsic("sub", vec_ty);

            match (target_scalar, precise) {
                (ScalarType::Int, false) => {
                    quote! {
                        unsafe {
                            #convert(a.into()).simd_into(self)
                        }
                    }
                }
                (ScalarType::Unsigned, false) => {
                    quote! {
                        unsafe {
                            let mut converted = #convert(a.into());

                            // In the common case where everything is in range of an i32, we don't need to do anything else.
                            let in_range = #cmplt(a.into(), #set1_float(2147483648.0));
                            let all_in_range = #movemask(in_range) == #all_ones;

                            if !all_in_range {
                                // Add any excess (beyond the maximum value)
                                let excess = #sub_float(a.into(), #set1_float(2147483648.0));
                                let excess_converted = #convert(#andnot(in_range, excess));
                                converted = #add_int(converted, excess_converted);
                            }

                            converted.simd_into(self)
                        }
                    }
                }
                (ScalarType::Int, true) => {
                    quote! {
                        unsafe {
                            let a = a.into();

                            let mut converted = #convert(a);

                            // In the common case where everything is in range, we don't need to do anything else.
                            let in_range = #cmplt(a, #set1_float(2147483648.0));
                            let all_in_range = #movemask(in_range) == #all_ones;

                            if !all_in_range {
                                // If we are above i32::MAX (2147483647), clamp to it.
                                converted = #blend(#set1_int(i32::MAX), converted, #cast_to_int(in_range));
                                // Set NaN to 0. Using `and` seems slightly faster than `blend`.
                                let is_not_nan = #cast_to_int(#cmpord(a, a));
                                converted = #and(converted, is_not_nan);
                                // We don't need to handle negative overflow because Intel's "invalid result" sentinel
                                // value is -2147483648, which is what we want anyway.
                            }

                            converted.simd_into(self)
                        }
                    }
                }
                (ScalarType::Unsigned, true) => {
                    quote! {
                        unsafe {
                            // Clamp out-of-range values (and NaN) to 0. Intel's `_mm_max_ps` always takes the second
                            // operand if the first is NaN.
                            let a = #max(a.into(), #set0());
                            let mut converted = #convert(a);

                            // In the common case where everything is in range of an i32, we don't need to do anything else.
                            let in_range = #cmplt(a, #set1_float(2147483648.0));
                            let all_in_range = #movemask(in_range) == #all_ones;

                            if !all_in_range {
                                let exceeds_unsigned_range = #cast_to_int(#cmplt(#set1_float(4294967040.0), a));
                                // Add any excess (beyond the maximum value)
                                let excess = #sub_float(a, #set1_float(2147483648.0));
                                let excess_converted = #convert(#andnot(in_range, excess));

                                // Clamp to u32::MAX.
                                converted = #add_int(converted, excess_converted);
                                converted = #blend(converted, #set1_int(u32::MAX as i32), exceeds_unsigned_range);
                            }

                            converted.simd_into(self)
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        (ScalarType::Int, ScalarType::Float) => {
            assert_eq!(
                vec_ty.scalar_bits, 32,
                "i64 to f64 conversions do not exist until AVX-512 and require special consideration"
            );
            let target_ty = vec_ty.reinterpret(target_scalar, target_scalar_bits);
            let intrinsic = simple_intrinsic("cvtepi32", &target_ty);
            quote! {
                unsafe {
                    #intrinsic(a.into()).simd_into(self)
                }
            }
        }
        (ScalarType::Unsigned, ScalarType::Float) => {
            assert_eq!(
                vec_ty.scalar_bits, 32,
                "u64 to f64 conversions do not exist until AVX-512 and require special consideration"
            );

            let target_ty = vec_ty.reinterpret(target_scalar, target_scalar_bits);
            let set1_int = set1_intrinsic(vec_ty);
            let set1_float = set1_intrinsic(&target_ty);
            let add_float = simple_intrinsic("add", &target_ty);
            let sub_float = simple_intrinsic("sub", &target_ty);
            let blend = intrinsic_ident("blend", "epi16", vec_ty.n_bits());
            let srli = intrinsic_ident("srli", "epi32", vec_ty.n_bits());
            let cast_to_float = cast_ident(
                vec_ty.scalar,
                target_scalar,
                vec_ty.scalar_bits,
                vec_ty.scalar_bits,
                vec_ty.n_bits(),
            );

            // Magical mystery algorithm taken from LLVM:
            // https://github.com/llvm/llvm-project/blob/6f8e87b9d097c5ef631f24d2eb2f34eb31b54d3b/llvm/lib/Target/X86/X86ISelLowering.cpp
            // (The file is too big for GitHub to show a preview, so no line numbers.)
            quote! {
                unsafe {
                    let a = a.into();
                    let lo = #blend::<0xAA>(a, #set1_int(0x4B000000));
                    let hi = #blend::<0xAA>(#srli::<16>(a), #set1_int(0x53000000));

                    let fhi = #sub_float(#cast_to_float(hi), #set1_float(f32::from_bits(0x53000080)));
                    let result = #add_float(#cast_to_float(lo), fhi);

                    result.simd_into(self)
                }
            }
        }
        _ => unimplemented!(),
    };

    quote! {
        #method_sig {
            #expr
        }
    }
}

pub(crate) fn handle_reinterpret(
    method_sig: TokenStream,
    vec_ty: &VecType,
    target_ty: ScalarType,
    scalar_bits: usize,
) -> TokenStream {
    let dst_ty = vec_ty.reinterpret(target_ty, scalar_bits);
    assert!(
        valid_reinterpret(vec_ty, target_ty, scalar_bits),
        "{vec_ty:?} must be reinterpretable as {dst_ty:?}"
    );

    if coarse_type(vec_ty) == coarse_type(&dst_ty) {
        let arch_ty = x86::arch_ty(vec_ty);
        quote! {
            #method_sig {
                #arch_ty::from(a).simd_into(self)
            }
        }
    } else {
        let ident = cast_ident(
            vec_ty.scalar,
            target_ty,
            vec_ty.scalar_bits,
            scalar_bits,
            vec_ty.n_bits(),
        );
        quote! {
            #method_sig {
                unsafe {
                    #ident(a.into()).simd_into(self)
                }
            }
        }
    }
}

pub(crate) fn handle_mask_reduce(
    method_sig: TokenStream,
    vec_ty: &VecType,
    quantifier: Quantifier,
    condition: bool,
) -> TokenStream {
    assert_eq!(
        vec_ty.scalar,
        ScalarType::Mask,
        "mask reduce ops only operate on masks"
    );

    let (movemask, all_ones) = match vec_ty.scalar_bits {
        32 | 64 => {
            let float_ty = vec_ty.cast(ScalarType::Float);
            let movemask = simple_intrinsic("movemask", &float_ty);
            let cast = cast_ident(
                ScalarType::Mask,
                ScalarType::Float,
                vec_ty.scalar_bits,
                vec_ty.scalar_bits,
                vec_ty.n_bits(),
            );
            let movemask = quote! { #movemask(#cast(a.into())) };
            let all_ones = match vec_ty.len {
                2 => quote! { 0b11 },
                4 => quote! { 0b1111 },
                8 => quote! { 0b11111111 },
                _ => unimplemented!(),
            };

            (movemask, all_ones)
        }
        8 | 16 => {
            let bits_ty = vec_ty.reinterpret(ScalarType::Int, 8);
            let movemask = simple_intrinsic("movemask", &bits_ty);
            let movemask = quote! { #movemask(a.into()) };
            let all_ones = match vec_ty.n_bits() {
                128 => quote! { 0xffff },
                256 => quote! { 0xffffffff },
                _ => unimplemented!(),
            };

            (movemask, all_ones)
        }
        _ => unreachable!(),
    };

    let op = match (quantifier, condition) {
        (Quantifier::Any, true) => quote! { != 0 },
        (Quantifier::Any, false) => quote! { != #all_ones },
        (Quantifier::All, true) => quote! { == #all_ones },
        (Quantifier::All, false) => quote! { == 0 },
    };

    quote! {
        #method_sig {
            unsafe {
                #movemask as u32 #op
            }
        }
    }
}

pub(crate) fn handle_load_interleaved(
    method_sig: TokenStream,
    vec_ty: &VecType,
    block_size: u16,
    block_count: u16,
) -> TokenStream {
    assert_eq!(
        block_size, 128,
        "only 128-bit blocks are currently supported"
    );
    assert_eq!(block_count, 4, "only count of 4 is currently supported");
    let expr = match vec_ty.scalar_bits {
        32 | 16 | 8 => {
            let block_ty =
                VecType::new(vec_ty.scalar, vec_ty.scalar_bits, 128 / vec_ty.scalar_bits);
            let load_unaligned =
                intrinsic_ident("loadu", coarse_type(&block_ty), block_ty.n_bits());
            let vec_32 = block_ty.reinterpret(block_ty.scalar, 32);
            let unpacklo_32 = simple_sign_unaware_intrinsic("unpacklo", &vec_32);
            let unpackhi_32 = simple_sign_unaware_intrinsic("unpackhi", &vec_32);
            let vec_64 = block_ty.reinterpret(block_ty.scalar, 64);
            let unpacklo_64 = simple_sign_unaware_intrinsic("unpacklo", &vec_64);
            let unpackhi_64 = simple_sign_unaware_intrinsic("unpackhi", &vec_64);

            let vec_combined =
                VecType::new(block_ty.scalar, block_ty.scalar_bits, block_ty.len * 2);
            let combine_half = Ident::new(
                &format!("combine_{}", block_ty.rust_name()),
                Span::call_site(),
            );
            let combine_full = Ident::new(
                &format!("combine_{}", vec_combined.rust_name()),
                Span::call_site(),
            );
            let block_len = block_size as usize / vec_ty.scalar_bits;

            let init_shuffle = match vec_ty.scalar_bits {
                16 => Some(quote! {
                    let mask = _mm_setr_epi8(
                        0, 1, 8, 9,
                        2, 3, 10, 11,
                        4, 5, 12, 13,
                        6, 7, 14, 15,
                    );
                    let v0 = _mm_shuffle_epi8(v0, mask);
                    let v1 = _mm_shuffle_epi8(v1, mask);
                    let v2 = _mm_shuffle_epi8(v2, mask);
                    let v3 = _mm_shuffle_epi8(v3, mask);
                }),
                8 => Some(quote! {
                    let mask = _mm_setr_epi8(
                        0, 4, 8, 12,
                        1, 5, 9, 13,
                        2, 6, 10, 14,
                        3, 7, 11, 15,
                    );
                    let v0 = _mm_shuffle_epi8(v0, mask);
                    let v1 = _mm_shuffle_epi8(v1, mask);
                    let v2 = _mm_shuffle_epi8(v2, mask);
                    let v3 = _mm_shuffle_epi8(v3, mask);
                }),
                _ => None,
            };

            let final_unpack = if vec_ty.scalar == ScalarType::Float && vec_ty.scalar_bits == 32 {
                let cast_32 = cast_ident(
                    ScalarType::Float,
                    ScalarType::Float,
                    64,
                    32,
                    block_ty.n_bits(),
                );
                let cast_64 = cast_ident(
                    ScalarType::Float,
                    ScalarType::Float,
                    32,
                    64,
                    block_ty.n_bits(),
                );

                quote! {
                    let out0 = #cast_32(#unpacklo_64(#cast_64(tmp0), #cast_64(tmp2))); // [0,4,8,12]
                    let out1 = #cast_32(#unpackhi_64(#cast_64(tmp0), #cast_64(tmp2))); // [1,5,9,13]
                    let out2 = #cast_32(#unpacklo_64(#cast_64(tmp1), #cast_64(tmp3))); // [2,6,10,14]
                    let out3 = #cast_32(#unpackhi_64(#cast_64(tmp1), #cast_64(tmp3))); // [3,7,11,15]
                }
            } else {
                quote! {
                    let out0 = #unpacklo_64(tmp0, tmp2); // [0,4,8,12]
                    let out1 = #unpackhi_64(tmp0, tmp2); // [1,5,9,13]
                    let out2 = #unpacklo_64(tmp1, tmp3); // [2,6,10,14]
                    let out3 = #unpackhi_64(tmp1, tmp3); // [3,7,11,15]
                }
            };

            quote! {
                unsafe {
                    let v0 = #load_unaligned(src.as_ptr() as *const _);
                    let v1 = #load_unaligned(src.as_ptr().add(#block_len) as *const _);
                    let v2 = #load_unaligned(src.as_ptr().add(2 * #block_len) as *const _);
                    let v3 = #load_unaligned(src.as_ptr().add(3 * #block_len) as *const _);

                    #init_shuffle

                    let tmp0 = #unpacklo_32(v0, v1); // [0,4,1,5]
                    let tmp1 = #unpackhi_32(v0, v1); // [2,6,3,7]
                    let tmp2 = #unpacklo_32(v2, v3); // [8,12,9,13]
                    let tmp3 = #unpackhi_32(v2, v3); // [10,14,11,15]

                    #final_unpack

                    self.#combine_full(
                        self.#combine_half(out0.simd_into(self), out1.simd_into(self)),
                        self.#combine_half(out2.simd_into(self), out3.simd_into(self)),
                    )
                }
            }
        }
        _ => unimplemented!(),
    };

    quote! {
        #method_sig {
            #expr
        }
    }
}

pub(crate) fn handle_store_interleaved(
    method_sig: TokenStream,
    vec_ty: &VecType,
    block_size: u16,
    block_count: u16,
) -> TokenStream {
    assert_eq!(
        block_size, 128,
        "only 128-bit blocks are currently supported"
    );
    assert_eq!(block_count, 4, "only count of 4 is currently supported");
    let expr = match vec_ty.scalar_bits {
        32 | 16 | 8 => {
            let block_ty =
                VecType::new(vec_ty.scalar, vec_ty.scalar_bits, 128 / vec_ty.scalar_bits);
            let store_unaligned =
                intrinsic_ident("storeu", coarse_type(&block_ty), block_ty.n_bits());
            let vec_32 = block_ty.reinterpret(block_ty.scalar, 32);
            let unpacklo_32 = simple_sign_unaware_intrinsic("unpacklo", &vec_32);
            let unpackhi_32 = simple_sign_unaware_intrinsic("unpackhi", &vec_32);
            let vec_64 = block_ty.reinterpret(block_ty.scalar, 64);
            let unpacklo_64 = simple_sign_unaware_intrinsic("unpacklo", &vec_64);
            let unpackhi_64 = simple_sign_unaware_intrinsic("unpackhi", &vec_64);

            let vec_combined =
                VecType::new(block_ty.scalar, block_ty.scalar_bits, block_ty.len * 2);
            let split_half = Ident::new(
                &format!("split_{}", vec_combined.rust_name()),
                Span::call_site(),
            );
            let split_full =
                Ident::new(&format!("split_{}", vec_ty.rust_name()), Span::call_site());
            let block_len = block_size as usize / vec_ty.scalar_bits;

            let post_shuffle = match vec_ty.scalar_bits {
                16 => Some(quote! {
                    let mask = _mm_setr_epi8(
                        0, 1, 4, 5,
                        8, 9, 12, 13,
                        2, 3, 6, 7,
                        10, 11, 14, 15,
                    );
                    let out0 = _mm_shuffle_epi8(out0, mask);
                    let out1 = _mm_shuffle_epi8(out1, mask);
                    let out2 = _mm_shuffle_epi8(out2, mask);
                    let out3 = _mm_shuffle_epi8(out3, mask);
                }),
                8 => Some(quote! {
                    let mask = _mm_setr_epi8(
                        0, 4, 8, 12,
                        1, 5, 9, 13,
                        2, 6, 10, 14,
                        3, 7, 11, 15,
                    );
                    let out0 = _mm_shuffle_epi8(out0, mask);
                    let out1 = _mm_shuffle_epi8(out1, mask);
                    let out2 = _mm_shuffle_epi8(out2, mask);
                    let out3 = _mm_shuffle_epi8(out3, mask);
                }),
                _ => None,
            };

            let final_unpack = if vec_ty.scalar == ScalarType::Float && vec_ty.scalar_bits == 32 {
                let cast_32 = cast_ident(
                    ScalarType::Float,
                    ScalarType::Float,
                    64,
                    32,
                    block_ty.n_bits(),
                );
                let cast_64 = cast_ident(
                    ScalarType::Float,
                    ScalarType::Float,
                    32,
                    64,
                    block_ty.n_bits(),
                );

                quote! {
                    let out0 = #cast_32(#unpacklo_64(#cast_64(tmp0), #cast_64(tmp2))); // [0,4,8,12]
                    let out1 = #cast_32(#unpackhi_64(#cast_64(tmp0), #cast_64(tmp2))); // [1,5,9,13]
                    let out2 = #cast_32(#unpacklo_64(#cast_64(tmp1), #cast_64(tmp3))); // [2,6,10,14]
                    let out3 = #cast_32(#unpackhi_64(#cast_64(tmp1), #cast_64(tmp3))); // [3,7,11,15]
                }
            } else {
                quote! {
                    let out0 = #unpacklo_64(tmp0, tmp2); // [0,4,8,12]
                    let out1 = #unpackhi_64(tmp0, tmp2); // [1,5,9,13]
                    let out2 = #unpacklo_64(tmp1, tmp3); // [2,6,10,14]
                    let out3 = #unpackhi_64(tmp1, tmp3); // [3,7,11,15]
                }
            };

            quote! {
                let (v01, v23) = self.#split_full(a);
                let (v0, v1) = self.#split_half(v01);
                let (v2, v3) = self.#split_half(v23);
                let v0 = v0.into();
                let v1 = v1.into();
                let v2 = v2.into();
                let v3 = v3.into();

                unsafe {
                    let tmp0 = #unpacklo_32(v0, v1); // [0,4,1,5]
                    let tmp1 = #unpackhi_32(v0, v1); // [2,6,3,7]
                    let tmp2 = #unpacklo_32(v2, v3); // [8,12,9,13]
                    let tmp3 = #unpackhi_32(v2, v3); // [10,14,11,15]

                    #final_unpack

                    #post_shuffle

                    #store_unaligned(dest.as_mut_ptr() as *mut _, out0);
                    #store_unaligned(dest.as_mut_ptr().add(#block_len) as *mut _, out1);
                    #store_unaligned(dest.as_mut_ptr().add(2 * #block_len) as *mut _, out2);
                    #store_unaligned(dest.as_mut_ptr().add(3 * #block_len) as *mut _, out3);
                }
            }
        }
        _ => unimplemented!(),
    };

    quote! {
        #method_sig {
            #expr
        }
    }
}
