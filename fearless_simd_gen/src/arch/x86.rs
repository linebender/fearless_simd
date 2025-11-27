// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

pub(crate) fn translate_op(op: &str) -> Option<&'static str> {
    Some(match op {
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
        "shl" => "shl",
        "shr" => "shr",
        "max" => "max",
        "min" => "min",
        "select" => "blendv",
        _ => return None,
    })
}

pub(crate) fn arch_ty(ty: &VecType) -> TokenStream {
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

pub(crate) fn expr(op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
    if let Some(op_name) = translate_op(op) {
        let sign_aware = matches!(op, "max" | "min");

        let suffix = match op_name {
            "and" | "or" | "xor" => coarse_type(ty),
            "blendv" if ty.scalar != ScalarType::Float => "epi8",
            _ => op_suffix(ty.scalar, ty.scalar_bits, sign_aware),
        };
        let intrinsic = intrinsic_ident(op_name, suffix, ty.n_bits());
        quote! { #intrinsic ( #( #args ),* ) }
    } else {
        let suffix = op_suffix(ty.scalar, ty.scalar_bits, true);
        match op {
            "floor" | "ceil" | "round_ties_even" | "trunc" => {
                let intrinsic = intrinsic_ident("round", suffix, ty.n_bits());
                let rounding_mode = match op {
                    "floor" => quote! { _MM_FROUND_TO_NEG_INF },
                    "ceil" => quote! { _MM_FROUND_TO_POS_INF },
                    "round_ties_even" => quote! { _MM_FROUND_TO_NEAREST_INT },
                    "trunc" => quote! { _MM_FROUND_TO_ZERO },
                    _ => unreachable!(),
                };
                quote! { #intrinsic::<{#rounding_mode | _MM_FROUND_NO_EXC}>( #( #args, )* ) }
            }
            "neg" => match ty.scalar {
                ScalarType::Float => {
                    let set1 = set1_intrinsic(ty);
                    let xor = simple_intrinsic("xor", ty);
                    quote! {
                        #( #xor(#args, #set1(-0.0)) )*
                    }
                }
                ScalarType::Int => {
                    let set0 = intrinsic_ident("setzero", coarse_type(ty), ty.n_bits());
                    let sub = simple_intrinsic("sub", ty);
                    let arg = &args[0];
                    quote! {
                        #sub(#set0(), #arg)
                    }
                }
                _ => unreachable!(),
            },
            "abs" => {
                let set1 = set1_intrinsic(ty);
                let andnot = simple_intrinsic("andnot", ty);
                quote! {
                    #( #andnot(#set1(-0.0), #args) )*
                }
            }
            "copysign" => {
                let a = &args[0];
                let b = &args[1];
                let set1 = set1_intrinsic(ty);
                let and = simple_intrinsic("and", ty);
                let andnot = simple_intrinsic("andnot", ty);
                let or = simple_intrinsic("or", ty);
                quote! {
                    let mask = #set1(-0.0);
                    #or(#and(mask, #b), #andnot(mask, #a))
                }
            }
            "mul" => {
                let suffix = op_suffix(ty.scalar, ty.scalar_bits, false);
                let intrinsic = if matches!(ty.scalar, ScalarType::Int | ScalarType::Unsigned) {
                    intrinsic_ident("mullo", suffix, ty.n_bits())
                } else {
                    intrinsic_ident("mul", suffix, ty.n_bits())
                };

                quote! { #intrinsic ( #( #args ),* ) }
            }
            "shrv" if ty.scalar_bits > 16 => {
                let suffix = op_suffix(ty.scalar, ty.scalar_bits, false);
                let name = match ty.scalar {
                    ScalarType::Int => "srav",
                    _ => "srlv",
                };
                let intrinsic = intrinsic_ident(name, suffix, ty.n_bits());
                quote! { #intrinsic ( #( #args ),* ) }
            }
            "min_precise" | "max_precise" => {
                assert_eq!(
                    ty.scalar,
                    ScalarType::Float,
                    "[min/max]_precise only makes sense on floats"
                );
                let suffix = op_suffix(ty.scalar, ty.scalar_bits, true);
                let intrinsic = intrinsic_ident(
                    if op == "max_precise" { "max" } else { "min" },
                    suffix,
                    ty.n_bits(),
                );
                let cmpunord = float_compare_method("unord", ty);
                let blend = intrinsic_ident("blendv", suffix, ty.n_bits());
                let a = &args[0];
                let b = &args[1];

                quote! {
                    let intermediate = #intrinsic(#a, #b);
                    // The x86 min/max intrinsics behave like `a < b ? a : b` and `a > b ? a : b` respectively. That
                    // means that if either `a` or `b` is NaN, they return the second argument `b`. So to implement a
                    // min/max where we always return the non-NaN argument, we add an additional check if `b` is NaN,
                    // and select `a` if so.
                    let b_is_nan = #cmpunord(#b, #b);
                    #blend(intermediate, #a, b_is_nan)
                }
            }
            _ => unimplemented!("{}", op),
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

/// Intrinsic name for the "int, float, or double" type (not as fine-grained as [`op_suffix`]).
pub(crate) fn coarse_type(vec_ty: &VecType) -> &'static str {
    use ScalarType::*;
    match (vec_ty.scalar, vec_ty.n_bits()) {
        (Int | Unsigned | Mask, 128) => "si128",
        (Int | Unsigned | Mask, 256) => "si256",
        (Int | Unsigned | Mask, 512) => "si512",
        _ => op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false),
    }
}

pub(crate) fn set1_intrinsic(vec_ty: &VecType) -> Ident {
    use ScalarType::*;
    let suffix = match (vec_ty.scalar, vec_ty.scalar_bits) {
        (Int | Unsigned | Mask, 64) => "epi64x",
        (scalar, bits) => op_suffix(scalar, bits, false),
    };

    intrinsic_ident("set1", suffix, vec_ty.n_bits())
}

pub(crate) fn simple_intrinsic(name: &str, vec_ty: &VecType) -> Ident {
    let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, true);

    intrinsic_ident(name, suffix, vec_ty.n_bits())
}

pub(crate) fn simple_sign_unaware_intrinsic(name: &str, vec_ty: &VecType) -> Ident {
    let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);

    intrinsic_ident(name, suffix, vec_ty.n_bits())
}

pub(crate) fn extend_intrinsic(
    ty: ScalarType,
    from_bits: usize,
    to_bits: usize,
    ty_bits: usize,
) -> Ident {
    let from_suffix = op_suffix(ty, from_bits, true);
    let to_suffix = op_suffix(ty, to_bits, false);

    intrinsic_ident(&format!("cvt{from_suffix}"), to_suffix, ty_bits)
}

pub(crate) fn pack_intrinsic(from_bits: usize, signed: bool, ty_bits: usize) -> Ident {
    let unsigned = match signed {
        true => "",
        false => "u",
    };
    let suffix = op_suffix(ScalarType::Int, from_bits, false);

    intrinsic_ident(&format!("pack{unsigned}s"), suffix, ty_bits)
}

pub(crate) fn unpack_intrinsic(
    scalar_type: ScalarType,
    scalar_bits: usize,
    low: bool,
    ty_bits: usize,
) -> Ident {
    let suffix = op_suffix(scalar_type, scalar_bits, false);

    let low_pref = if low { "lo" } else { "hi" };

    intrinsic_ident(&format!("unpack{low_pref}"), suffix, ty_bits)
}

pub(crate) fn intrinsic_ident(name: &str, suffix: &str, ty_bits: usize) -> Ident {
    let prefix = match ty_bits {
        128 => "",
        256 => "256",
        512 => "512",
        _ => unreachable!(),
    };

    format_ident!("_mm{prefix}_{name}_{suffix}")
}

pub(crate) fn cast_ident(
    src_scalar_ty: ScalarType,
    dst_scalar_ty: ScalarType,
    src_scalar_bits: usize,
    dst_scalar_bits: usize,
    ty_bits: usize,
) -> Ident {
    let prefix = match ty_bits {
        128 => "",
        256 => "256",
        512 => "512",
        _ => unreachable!(),
    };
    let src_name = coarse_type(&VecType::new(
        src_scalar_ty,
        src_scalar_bits,
        ty_bits / src_scalar_bits,
    ));
    let dst_name = coarse_type(&VecType::new(
        dst_scalar_ty,
        dst_scalar_bits,
        ty_bits / dst_scalar_bits,
    ));

    format_ident!("_mm{prefix}_cast{src_name}_{dst_name}")
}

pub(crate) fn float_compare_method(method: &str, vec_ty: &VecType) -> TokenStream {
    match vec_ty.n_bits() {
        128 => {
            let ident = match method {
                "ord" => simple_intrinsic("cmpord", vec_ty),
                "unord" => simple_intrinsic("cmpunord", vec_ty),
                _ => intrinsic_ident(
                    translate_op(method).unwrap(),
                    op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false),
                    vec_ty.n_bits(),
                ),
            };
            quote! { #ident }
        }
        256 => {
            // For AVX2 and up, Intel gives us a generic comparison intrinsic that takes a predicate. There are 32,
            // of which only a few are useful and the rest will violate IEEE754 and/or raise a SIGFPE on NaN.
            //
            // https://www.felixcloutier.com/x86/cmppd#tbl-3-1
            let order_predicate = match method {
                "simd_eq" => 0x00,
                "simd_lt" => 0x11,
                "simd_le" => 0x12,
                "simd_ge" => 0x1D,
                "simd_gt" => 0x1E,
                "ord" => 0x07,
                "unord" => 0x03,
                _ => unreachable!(),
            };
            let intrinsic = simple_intrinsic("cmp", vec_ty);

            quote! {
                #intrinsic::<#order_predicate>
            }
        }
        _ => unimplemented!(),
    }
}
