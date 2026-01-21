// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

fn translate_op(op: &str) -> Option<&'static str> {
    Some(match op {
        "abs" => "abs",
        "neg" => "neg",
        "floor" => "floor",
        "ceil" => "ceil",
        "round_ties_even" => "nearest",
        "trunc" => "trunc",
        "sqrt" => "sqrt",
        "add" => "add",
        // TODO: Is wrapping sub same on WASM?
        "sub" => "sub",
        "mul" => "mul",
        "div" => "div",
        "simd_eq" => "eq",
        "simd_lt" => "lt",
        "simd_le" => "le",
        "simd_ge" => "ge",
        "simd_gt" => "gt",
        "not" => "not",
        "and" => "and",
        "or" => "or",
        "xor" => "xor",
        "shrv" => "shr",
        "max" => "max",
        "min" => "min",
        "splat" => "splat",
        _ => return None,
    })
}

pub(crate) fn simple_intrinsic(name: &str, ty: &VecType) -> Ident {
    let ty_prefix = arch_prefix(ty);
    let ident = Ident::new(name, Span::call_site());
    Ident::new(&format!("{}_{}", ty_prefix, ident), Span::call_site())
}

pub(crate) fn v128_intrinsic(name: &str) -> Ident {
    let ty_prefix = Ident::new("v128", Span::call_site());
    let ident = Ident::new(name, Span::call_site());
    Ident::new(&format!("{}_{}", ty_prefix, ident), Span::call_site())
}

pub(crate) fn arch_prefix(ty: &VecType) -> Ident {
    let scalar = match ty.scalar {
        ScalarType::Float => "f",
        ScalarType::Unsigned => "u",
        ScalarType::Int | ScalarType::Mask => "i",
    };
    let name = format!("{}{}x{}", scalar, ty.scalar_bits, ty.len);
    Ident::new(&name, Span::call_site())
}

// expects args and return value in arch dialect
pub(crate) fn expr(op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
    let intrinsic = match translate_op(op) {
        Some(translated @ ("not" | "and" | "or" | "xor")) => v128_intrinsic(translated),
        Some(translated) => simple_intrinsic(translated, ty),
        None => unimplemented!("missing {op}"),
    };

    quote! { #intrinsic ( #( #args ),* ) }
}
