// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::Arch;
use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

pub(crate) fn translate_op(op: &str) -> Option<&'static str> {
    Some(match op {
        "floor" => "floor",
        "sqrt" => "sqrt",
        "add" => "add",
        "sub" => "sub",
        "div" => "div",
        "and" => "and",
        "simd_eq" => "cmpeq",
        "simd_lt" => "cmplt",
        "simd_le" => "cmple",
        "simd_ge" => "cmpge",
        "simd_gt" => "cmpgt",
        "or" => "or",
        "xor" => "xor",
        "shr" => "shr",
        "max" => "max",
        "min" => "min",
        "max_precise" => "max",
        "min_precise" => "min",
        _ => return None,
    })
}

pub(crate) struct Sse4_2;

impl Arch for Sse4_2 {
    fn arch_ty(&self, ty: &VecType) -> TokenStream {
        let suffix = match (ty.scalar, ty.scalar_bits) {
            (ScalarType::Float, 32) => "",
            (ScalarType::Float, 64) => "d",
            (ScalarType::Float, _) => unimplemented!(),
            (ScalarType::Unsigned | ScalarType::Int | ScalarType::Mask, _) => "i",
        };
        let name = format!("__m{}{}", ty.scalar_bits * ty.len, suffix);
        let ident = Ident::new(&name, Span::call_site());
        quote! { #ident }
    }

    fn expr(&self, op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
        if let Some(op_name) = translate_op(op) {
            let suffix = match op_name {
                "and" | "or" | "xor" => "si128",
                _ => op_suffix(ty.scalar, ty.scalar_bits, false),
            };
            let intrinsic = format_ident!("_mm_{op_name}_{suffix}");
            quote! { #intrinsic ( #( #args ),* ) }
        } else {
            let suffix = op_suffix(ty.scalar, ty.scalar_bits, true);
            match op {
                "trunc" => {
                    let intrinsic = format_ident!("_mm_round_{suffix}");
                    quote! { #intrinsic ( #( #args, )* _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) }
                }
                "neg" => {
                    let set1 = set1_intrinsic(ty.scalar, ty.scalar_bits);
                    let xor = simple_intrinsic("xor", ScalarType::Float, ty.scalar_bits);
                    quote! {
                        #( #xor(#args, #set1(-0.0)) )*
                    }
                }
                "abs" => {
                    let set1 = set1_intrinsic(ty.scalar, ty.scalar_bits);
                    let andnot = simple_intrinsic("andnot", ScalarType::Float, ty.scalar_bits);
                    quote! {
                        #( #andnot(#set1(-0.0), #args) )*
                    }
                }
                "copysign" => {
                    let a = &args[0];
                    let b = &args[1];
                    let set1 = set1_intrinsic(ty.scalar, ty.scalar_bits);
                    let and = simple_intrinsic("and", ScalarType::Float, ty.scalar_bits);
                    let andnot = simple_intrinsic("andnot", ScalarType::Float, ty.scalar_bits);
                    let or = simple_intrinsic("or", ScalarType::Float, ty.scalar_bits);
                    quote! {
                        let mask = #set1(-0.0);
                        #or(#and(mask, #b), #andnot(mask, #a))
                    }
                }
                "mul" => {
                    let suffix = op_suffix(ty.scalar, ty.scalar_bits, false);
                    let intrinsic = if matches!(ty.scalar, ScalarType::Int | ScalarType::Unsigned) {
                        format_ident!("_mm_mullo_{suffix}")
                    } else {
                        format_ident!("_mm_mul_{suffix}")
                    };

                    quote! { #intrinsic ( #( #args ),* ) }
                }
                _ => unimplemented!("{}", op),
            }
        }
    }
}

pub(crate) fn op_suffix(mut ty: ScalarType, bits: usize, sign_aware: bool) -> &'static str {
    use ScalarType::*;
    if !sign_aware && ty == Unsigned {
        ty = Int;
    }
    match (ty, bits) {
        (Float, 32) => "ps",
        (Float, 64) => "pd",
        (Float, _) => unimplemented!("{bits} bit floats"),
        (Int | Mask, 8) => "epi8",
        (Int | Mask, 16) => "epi16",
        (Int | Mask, 32) => "epi32",
        (Int | Mask, 64) => "epi64",
        (Unsigned, 8) => "epu8",
        (Unsigned, 16) => "epu16",
        (Unsigned, 32) => "epu32",
        (Unsigned, 64) => "epu64",
        _ => unreachable!(),
    }
}

pub(crate) fn set1_intrinsic(ty: ScalarType, bits: usize) -> Ident {
    use ScalarType::*;
    let suffix = match (ty, bits) {
        (Int | Unsigned | Mask, 64) => "epi64x",
        _ => op_suffix(ty, bits, false),
    };
    format_ident!("_mm_set1_{suffix}")
}

pub(crate) fn simple_intrinsic(name: &str, ty: ScalarType, bits: usize) -> Ident {
    let suffix = op_suffix(ty, bits, true);
    format_ident!("_mm_{name}_{suffix}")
}

pub(crate) fn extend_intrinsic(ty: ScalarType, from_bits: usize, to_bits: usize) -> Ident {
    let from_suffix = op_suffix(ty, from_bits, true);
    let to_suffix = op_suffix(ty, to_bits, false);
    format_ident!("_mm_cvt{from_suffix}_{to_suffix}")
}

pub(crate) fn cvt_intrinsic(from: VecType, to: VecType) -> Ident {
    let from_suffix = op_suffix(from.scalar, from.scalar_bits, false);
    let to_suffix = op_suffix(to.scalar, to.scalar_bits, false);
    format_ident!("_mm_cvt{from_suffix}_{to_suffix}")
}

pub(crate) fn pack_intrinsic(from_bits: usize, signed: bool) -> Ident {
    let unsigned = match signed {
        true => "",
        false => "u",
    };
    let suffix = op_suffix(ScalarType::Int, from_bits, false);
    format_ident!("_mm_pack{unsigned}s_{suffix}")
}

pub(crate) fn unpack_intrinsic(scalar_type: ScalarType, scalar_bits: usize, low: bool) -> Ident {
    let suffix = op_suffix(scalar_type, scalar_bits, false);

    let low_pref = if low { "lo" } else { "hi" };
    format_ident!("_mm_unpack{low_pref}_{suffix}")
}
