// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub(crate) fn translate_op(op: &str, is_float: bool) -> Option<&'static str> {
    Some(match op {
        "abs" => "abs",
        "copysign" => "copysign",
        "neg" => "neg",
        "floor" => "floor",
        "ceil" => "ceil",
        "round_ties_even" => "round_ties_even",
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
        "shlv" => "shl",
        "max" => "max",
        "min" => "min",
        "max_precise" => "max",
        "min_precise" => "min",
        _ => return None,
    })
}

pub(crate) fn simple_intrinsic(name: &str, ty: &VecType) -> TokenStream {
    let ty_prefix = ty.scalar.rust(ty.scalar_bits);
    let ident = Ident::new(name, Span::call_site());

    quote! {#ty_prefix::#ident}
}

pub(crate) fn expr(op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
    let Some(translated) = translate_op(op, ty.scalar == ScalarType::Float) else {
        unimplemented!("missing {op}");
    };

    let intrinsic = simple_intrinsic(translated, ty);
    quote! { #intrinsic ( #( #args ),* ) }
}
