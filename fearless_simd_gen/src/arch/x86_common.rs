// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::types::{ScalarType, VecType};
use crate::x86_common::{intrinsic_ident, op_suffix, set1_intrinsic, simple_intrinsic};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

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
        "shl" => "shl",
        "shr" => "shr",
        "max" => "max",
        "min" => "min",
        "max_precise" => "max",
        "min_precise" => "min",
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
            "and" | "or" | "xor" => "si128",
            _ => op_suffix(ty.scalar, ty.scalar_bits, sign_aware),
        };
        let intrinsic = intrinsic_ident(op_name, suffix, ty.n_bits());
        quote! { #intrinsic ( #( #args ),* ) }
    } else {
        let suffix = op_suffix(ty.scalar, ty.scalar_bits, true);
        match op {
            "trunc" => {
                let intrinsic = intrinsic_ident("round", suffix, ty.n_bits());
                quote! { #intrinsic ( #( #args, )* _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) }
            }
            "neg" => {
                let set1 = set1_intrinsic(ty.scalar, ty.scalar_bits, ty.n_bits());
                let xor = simple_intrinsic("xor", ScalarType::Float, ty.scalar_bits, ty.n_bits());
                quote! {
                    #( #xor(#args, #set1(-0.0)) )*
                }
            }
            "abs" => {
                let set1 = set1_intrinsic(ty.scalar, ty.scalar_bits, ty.n_bits());
                let andnot =
                    simple_intrinsic("andnot", ScalarType::Float, ty.scalar_bits, ty.n_bits());
                quote! {
                    #( #andnot(#set1(-0.0), #args) )*
                }
            }
            "copysign" => {
                let a = &args[0];
                let b = &args[1];
                let set1 = set1_intrinsic(ty.scalar, ty.scalar_bits, ty.n_bits());
                let and = simple_intrinsic("and", ScalarType::Float, ty.scalar_bits, ty.n_bits());
                let andnot =
                    simple_intrinsic("andnot", ScalarType::Float, ty.scalar_bits, ty.n_bits());
                let or = simple_intrinsic("or", ScalarType::Float, ty.scalar_bits, ty.n_bits());
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
            _ => unimplemented!("{}", op),
        }
    }
}
