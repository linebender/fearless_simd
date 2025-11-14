// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    clippy::match_single_binding,
    unreachable_pub,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub(crate) fn translate_op(op: &str, is_float: bool) -> Option<&'static str> {
    Some(match op {
        "abs" => "abs",
        "copysign" => "copysign",
        "neg" => "neg",
        "floor" => "floor",
        "fract" => "fract",
        "trunc" => "trunc",
        "sqrt" => "sqrt",
        "add" => {
            if is_float {
                "add"
            } else {
                "wrapping_add"
            }
        }
        "sub" => {
            if is_float {
                "sub"
            } else {
                "wrapping_sub"
            }
        }
        "mul" => {
            if is_float {
                "mul"
            } else {
                "wrapping_mul"
            }
        }
        "div" => "div",
        "simd_eq" => "eq",
        "simd_lt" => "lt",
        "simd_le" => "le",
        "simd_ge" => "ge",
        "simd_gt" => "gt",
        "not" => "not",
        "and" => "bitand",
        "or" => "bitor",
        "xor" => "bitxor",
        "shr" => "shr",
        "shl" => "shl",
        "shrv" => "shr",
        // TODO: Do we need to polyfill so behavior is consistent with NEON?
        "max" => "max",
        "min" => "min",
        "max_precise" => "max",
        "min_precise" => "min",
        _ => return None,
    })
}

pub fn simple_intrinsic(name: &str, ty: &VecType) -> TokenStream {
    let ty_prefix = arch_ty(ty);
    let ident = Ident::new(name, Span::call_site());

    quote! {#ty_prefix::#ident}
}

pub(crate) fn arch_ty(ty: &VecType) -> TokenStream {
    let scalar = match ty.scalar {
        ScalarType::Float => "f",
        ScalarType::Unsigned => "u",
        ScalarType::Int | ScalarType::Mask => "i",
    };
    let name = format!("{}{}", scalar, ty.scalar_bits);
    let ident = Ident::new(&name, Span::call_site());
    quote! { #ident }
}

pub(crate) fn expr(op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
    if let Some(translated) = translate_op(op, ty.scalar == ScalarType::Float) {
        let intrinsic = simple_intrinsic(translated, ty);
        quote! { #intrinsic ( #( #args ),* ) }
    } else {
        match op {
            _ => unimplemented!("missing {op}"),
        }
    }
}
