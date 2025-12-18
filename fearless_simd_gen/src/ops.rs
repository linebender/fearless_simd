// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use anyhow::{Context, anyhow};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use std::fmt::Write;

use crate::{
    generic::generic_op_name,
    types::{ScalarType, VecType},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Quantifier {
    Any,
    All,
}

impl Quantifier {
    pub(crate) fn bool_op(&self) -> TokenStream {
        match self {
            Self::Any => quote! { || },
            Self::All => quote! { && },
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefKind {
    Value,
    Ref,
    Mut,
}

impl RefKind {
    pub(crate) fn token(&self) -> Option<TokenStream> {
        match self {
            Self::Value => None,
            Self::Ref => Some(quote! { & }),
            Self::Mut => Some(quote! { &mut }),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum OpSig {
    /// Takes a single argument of the underlying SIMD element type, and returns the corresponding vector type.
    Splat,
    /// Takes a single argument of the vector type, and returns that same vector type.
    Unary,
    /// Takes two argument of the vector type, and returns that same vector type.
    Binary,
    /// Takes three argument of the vector type, and returns that same vector type.
    Ternary,
    /// Takes two argument of the vector type, and returns the corresponding mask type.
    Compare,
    /// Takes a single argument of the vector type, which must be a mask type, and two elements of another vector type
    /// of the same scalar width and length. Returns that latter vector type.
    Select,
    /// Takes two arguments of a vector type, and returns a vector type that's twice as wide.
    Combine { combined_ty: VecType },
    /// Takes a single argument of a vector type, and returns a tuple of two vector types that are each half as wide.
    Split { half_ty: VecType },
    /// Takes two arguments of a vector type, and returns that same vector type.
    Zip { select_low: bool },
    /// Takes two arguments of a vector type, and returns that same vector type.
    Unzip { select_even: bool },
    /// Takes a single argument of the source vector type, and returns a vector type of the target scalar type and the
    /// same length.
    Cvt {
        target_ty: ScalarType,
        scalar_bits: usize,
        precise: bool,
    },
    /// Takes a single argument of the source vector type, and returns a vector type of the target scalar type and the
    /// same bit width.
    Reinterpret {
        target_ty: ScalarType,
        scalar_bits: usize,
    },
    /// Takes a single argument of the source vector type, and returns a vector type of the target scalar type and the
    /// same length.
    WidenNarrow { target_ty: VecType },
    /// Takes an argument of a vector type and another u32 argument (the shift amount), and returns that same vector
    /// type.
    Shift,
    /// Takes an argument of a mask vector type, and returns a boolean.
    MaskReduce {
        quantifier: Quantifier,
        condition: bool,
    },
    /// Takes an argument of an array of a certain scalar type, with the length (`block_size` * `block_count`) / [scalar
    /// type's byte size]. Returns a vector type of that scalar type and length.
    ///
    /// First argument is the base block size (i.e. 128), second argument is how many blocks. For example,
    /// `LoadInterleaved(128, 4)` would correspond to the NEON instructions `vld4q_f32`, while `LoadInterleaved(64, 4)`
    /// would correspond to `vld4_f32`.
    LoadInterleaved { block_size: u16, block_count: u16 },
    /// The inverse of [`OpSig::LoadInterleaved`]. Takes a vector argument with the length (`block_size` * `block_count`) /
    /// [scalar type's byte size], and a mutable reference to a scalar array of the same length, and returns nothing.
    StoreInterleaved { block_size: u16, block_count: u16 },
    /// Takes a single argument of the array type corresponding to the vector type (e.g. `[f32; 4]` for `f32x4<S>`), or
    /// a reference to it, and returns that vector type.
    FromArray { kind: RefKind },
    /// Takes a single argument of the vector type, or a reference to it, and returns the corresponding array type (e.g.
    /// `[f32; 4]` for `f32x4<S>`) or a reference to it.
    AsArray { kind: RefKind },
    /// Takes a single argument of the vector type, and returns a vector type with `u8` elements and the same bit width.
    FromBytes,
    /// Takes a single argument of a vector type with `u8` elements, and returns a vector type with different elements
    /// and the same bit width.
    ToBytes,
}

/// Where this operation is defined, and how it is called.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpKind {
    /// This operation is implemented as a `core::ops` overloaded operation (e.g. `core::ops::Add`).
    Overloaded(CoreOpTrait),
    /// This operation is a method on the `SimdBase` type.
    BaseTraitMethod,
    /// This operation is a method on the `SimdInt`, `SimdFloat`, or `SimdMask` type.
    VecTraitMethod,
    /// This operation is a method on its own bespoke trait.
    OwnTrait,
    /// This operation is only available as a method on the `Simd` trait.
    AssociatedOnly,
}

#[derive(Clone, Copy)]
pub(crate) struct Op {
    /// The method name. Used for the `Simd` trait's implementation of this method, with the specific vector type
    /// suffixed.
    pub(crate) method: &'static str,
    /// Where the operation is defined.
    pub(crate) kind: OpKind,
    /// The method signature.
    pub(crate) sig: OpSig,
    /// The documentation string for this method. Basic templating facilities are available: currently `{arg0}`,
    /// `{arg1}`, etc. correspond to the argument names, which are different between the `Simd` trait methods and the
    /// ones defined on the vector types themselves (for instance, the first argument is always `self` in the latter).
    pub(crate) doc: &'static str,
}

impl Op {
    const fn new(method: &'static str, kind: OpKind, sig: OpSig, doc: &'static str) -> Self {
        Self {
            method,
            kind,
            sig,
            doc,
        }
    }

    pub(crate) fn simd_trait_method_sig(&self, vec_ty: &VecType) -> TokenStream {
        let ty = vec_ty.rust();
        let arg_names = self
            .sig
            .simd_trait_arg_names()
            .iter()
            .map(|n| Ident::new(n, Span::call_site()))
            .collect::<Vec<_>>();
        let method_ident = generic_op_name(self.method, vec_ty);
        let sig_inner = match &self.sig {
            OpSig::Splat => {
                let arg0 = &arg_names[0];
                let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
                quote! { (self, #arg0: #scalar) -> #ty<Self> }
            }
            OpSig::LoadInterleaved {
                block_size,
                block_count,
            } => {
                let arg0 = &arg_names[0];
                let arg_ty = load_interleaved_arg_ty(*block_size, *block_count, vec_ty);
                quote! { (self, #arg0: #arg_ty) -> #ty<Self> }
            }
            OpSig::StoreInterleaved {
                block_size,
                block_count,
            } => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let arg_ty = store_interleaved_arg_ty(*block_size, *block_count, vec_ty);
                quote! { (self, #arg0: #ty<Self>, #arg1: #arg_ty) -> () }
            }
            OpSig::Compare => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let result = vec_ty.mask_ty().rust();
                quote! { (self, #arg0: #ty<Self>, #arg1: #ty<Self>) -> #result<Self> }
            }
            OpSig::Split { half_ty } => {
                let arg0 = &arg_names[0];
                let result = half_ty.rust();
                quote! { (self, #arg0: #ty<Self>) -> (#result<Self>, #result<Self>) }
            }
            OpSig::Combine { combined_ty } => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let result = combined_ty.rust();
                quote! { (self, #arg0: #ty<Self>, #arg1: #ty<Self>) -> #result<Self> }
            }
            OpSig::Unary => {
                let arg0 = &arg_names[0];
                quote! { (self, #arg0: #ty<Self>) -> #ty<Self> }
            }
            OpSig::Binary | OpSig::Zip { .. } | OpSig::Unzip { .. } => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                quote! { (self, #arg0: #ty<Self>, #arg1: #ty<Self>) -> #ty<Self> }
            }
            OpSig::Cvt {
                target_ty,
                scalar_bits,
                ..
            } => {
                let arg0 = &arg_names[0];
                let result = VecType::new(*target_ty, *scalar_bits, vec_ty.len).rust();
                quote! { (self, #arg0: #ty<Self>) -> #result<Self> }
            }
            OpSig::Reinterpret {
                target_ty,
                scalar_bits,
            } => {
                let arg0 = &arg_names[0];
                let result = vec_ty.reinterpret(*target_ty, *scalar_bits).rust();
                quote! { (self, #arg0: #ty<Self>) -> #result<Self> }
            }
            OpSig::WidenNarrow { target_ty } => {
                let arg0 = &arg_names[0];
                let result = target_ty.rust();
                quote! { (self, #arg0: #ty<Self>) -> #result<Self> }
            }
            OpSig::MaskReduce { .. } => {
                let arg0 = &arg_names[0];
                quote! { (self, #arg0: #ty<Self>) -> bool }
            }
            OpSig::Shift => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                quote! { (self, #arg0: #ty<Self>, #arg1: u32) -> #ty<Self> }
            }
            OpSig::Ternary => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let arg2 = &arg_names[2];
                quote! { (self, #arg0: #ty<Self>, #arg1: #ty<Self>, #arg2: #ty<Self>) -> #ty<Self> }
            }
            OpSig::Select => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let arg2 = &arg_names[2];
                let mask_ty = vec_ty.mask_ty().rust();
                quote! { (self, #arg0: #mask_ty<Self>, #arg1: #ty<Self>, #arg2: #ty<Self>) -> #ty<Self> }
            }
            OpSig::FromArray { kind } => {
                let arg0 = &arg_names[0];
                let ref_tok = kind.token();
                let rust_scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
                let len = vec_ty.len;
                let array_ty = quote! { [#rust_scalar; #len] };
                quote! { (self, #arg0: #ref_tok #array_ty) -> #ty<Self> }
            }
            OpSig::AsArray { kind } => {
                let arg0 = &arg_names[0];
                let ref_tok = kind.token();
                let rust_scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
                let len = vec_ty.len;
                let array_ty = quote! { [#rust_scalar; #len] };
                quote! { (self, #arg0: #ref_tok #ty<Self>) -> #ref_tok #array_ty }
            }
            OpSig::FromBytes => {
                let arg0 = &arg_names[0];
                let bytes_ty = vec_ty.reinterpret(ScalarType::Unsigned, 8).rust();
                quote! { (self, #arg0: #bytes_ty<Self>) -> #ty<Self> }
            }
            OpSig::ToBytes => {
                let arg0 = &arg_names[0];
                let bytes_ty = vec_ty.reinterpret(ScalarType::Unsigned, 8).rust();
                quote! { (self, #arg0: #ty<Self>) -> #bytes_ty<Self> }
            }
        };

        quote! {
            fn #method_ident #sig_inner
        }
    }

    pub(crate) fn vec_trait_method_sig(&self) -> Option<TokenStream> {
        let arg_names = self
            .sig
            .vec_trait_arg_names()
            .iter()
            .map(|n| Ident::new(n, Span::call_site()))
            .collect::<Vec<_>>();
        let method_ident = Ident::new(self.method, Span::call_site());
        let sig_inner = match &self.sig {
            OpSig::Splat | OpSig::LoadInterleaved { .. } | OpSig::StoreInterleaved { .. } => {
                return None;
            }
            OpSig::Unary
            | OpSig::Cvt { .. }
            | OpSig::Reinterpret { .. }
            | OpSig::WidenNarrow { .. } => {
                let arg0 = &arg_names[0];
                quote! { (#arg0) -> Self }
            }
            OpSig::MaskReduce { .. } => {
                let arg0 = &arg_names[0];
                quote! { (#arg0) -> bool }
            }
            OpSig::Binary | OpSig::Zip { .. } | OpSig::Unzip { .. } => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                quote! { (#arg0, #arg1: impl SimdInto<Self, S>) -> Self }
            }
            OpSig::Compare => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                quote! { (#arg0, #arg1: impl SimdInto<Self, S>) -> Self::Mask }
            }
            OpSig::Shift => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                quote! { (#arg0, #arg1: u32) -> Self }
            }
            OpSig::Ternary => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let arg2 = &arg_names[2];
                quote! { (#arg0, #arg1: impl SimdInto<Self, S>, #arg2: impl SimdInto<Self, S>) -> Self }
            }
            // select is currently done by trait, but maybe we'll implement for
            // masks.
            OpSig::Select => return None,
            // These signatures involve types not in the Simd trait
            OpSig::Split { .. }
            | OpSig::Combine { .. }
            | OpSig::FromArray { .. }
            | OpSig::AsArray { .. }
            | OpSig::FromBytes
            | OpSig::ToBytes => return None,
        };
        Some(quote! { fn #method_ident #sig_inner })
    }

    pub(crate) fn format_docstring(&self, flavor: TyFlavor) -> String {
        let arg_names = match flavor {
            TyFlavor::SimdTrait => self.sig.simd_trait_arg_names(),
            TyFlavor::VecImpl => self.sig.vec_trait_arg_names(),
        };

        let interpolate_var_into = |dest: &mut String, template_var: &str| {
            if let Some(arg_num) = template_var.strip_prefix("arg") {
                let arg_num: usize = arg_num
                    .parse()
                    .with_context(|| format!("Invalid arg number: {arg_num:?}"))?;
                let arg_name = *arg_names.get(arg_num).with_context(|| {
                    format!("Arg number {arg_num} out of range (args are {arg_names:?})")
                })?;
                dest.write_str(arg_name)?;
                Ok(())
            } else {
                Err(anyhow!("Unknown template variable: {template_var:?}"))
            }
        };

        let docstring = self.doc;
        let mut remaining = self.doc;
        let mut dest = String::new();
        loop {
            // Go until we reach the next opening brace. If there is none, push the rest of the string; we're done,
            let Some((left, right)) = remaining.split_once('{') else {
                dest.push_str(remaining);
                break;
            };

            dest.push_str(left);

            let Some((template_var, rest)) = right.split_once('}') else {
                panic!("Unmatched closing brace: {docstring:?}");
            };
            if let Err(e) = interpolate_var_into(&mut dest, template_var) {
                panic!("{e}\nIn docstring: {docstring:?}");
            }

            remaining = rest;
        }

        dest
    }
}

const BASE_OPS: &[Op] = &[
    Op::new(
        "splat",
        OpKind::BaseTraitMethod,
        OpSig::Splat,
        "Create a SIMD vector with all elements set to the given value.",
    ),
    Op::new(
        "load_array",
        OpKind::AssociatedOnly,
        OpSig::FromArray {
            kind: RefKind::Value,
        },
        "Create a SIMD vector from an array of the same length.",
    ),
    Op::new(
        "load_array_ref",
        OpKind::AssociatedOnly,
        OpSig::FromArray { kind: RefKind::Ref },
        "Create a SIMD vector from an array of the same length.",
    ),
    Op::new(
        "as_array",
        OpKind::AssociatedOnly,
        OpSig::AsArray {
            kind: RefKind::Value,
        },
        "Convert a SIMD vector to an array.",
    ),
    Op::new(
        "as_array_ref",
        OpKind::AssociatedOnly,
        OpSig::AsArray { kind: RefKind::Ref },
        "Project a reference to a SIMD vector to a reference to the equivalent array.",
    ),
    Op::new(
        "as_array_mut",
        OpKind::AssociatedOnly,
        OpSig::AsArray { kind: RefKind::Mut },
        "Project a mutable reference to a SIMD vector to a mutable reference to the equivalent array.",
    ),
    Op::new(
        "cvt_from_bytes",
        OpKind::OwnTrait,
        OpSig::FromBytes,
        "Reinterpret a vector of bytes as a SIMD vector of a given type, with the equivalent byte length.",
    ),
    Op::new(
        "cvt_to_bytes",
        OpKind::OwnTrait,
        OpSig::ToBytes,
        "Reinterpret a SIMD vector as a vector of bytes, with the equivalent byte length.",
    ),
];

const FLOAT_OPS: &[Op] = &[
    Op::new(
        "abs",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Compute the absolute value of each element.",
    ),
    Op::new(
        "neg",
        OpKind::Overloaded(CoreOpTrait::Neg),
        OpSig::Unary,
        "Negate each element of the vector.",
    ),
    Op::new(
        "sqrt",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Compute the square root of each element.\n\n\
        Negative elements other than `-0.0` will become NaN.",
    ),
    Op::new(
        "add",
        OpKind::Overloaded(CoreOpTrait::Add),
        OpSig::Binary,
        "Add two vectors element-wise.",
    ),
    Op::new(
        "sub",
        OpKind::Overloaded(CoreOpTrait::Sub),
        OpSig::Binary,
        "Subtract two vectors element-wise.",
    ),
    Op::new(
        "mul",
        OpKind::Overloaded(CoreOpTrait::Mul),
        OpSig::Binary,
        "Multiply two vectors element-wise.",
    ),
    Op::new(
        "div",
        OpKind::Overloaded(CoreOpTrait::Div),
        OpSig::Binary,
        "Divide two vectors element-wise.",
    ),
    Op::new(
        "copysign",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Return a vector with the magnitude of `{arg0}` and the sign of `{arg1}` for each element.\n\n\
        This operation copies the sign bit, so if an input element is NaN, the output element will be a NaN with the same payload and a copied sign bit.",
    ),
    Op::new(
        "simd_eq",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for equality.\n\n\
        Returns a mask where each element is all ones if the corresponding elements are equal, and all zeroes if not.",
    ),
    Op::new(
        "simd_lt",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for less than.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is less than `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_le",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for less than or equal.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is less than or equal to `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_ge",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for greater than or equal.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is greater than or equal to `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_gt",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for greater than.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is greater than `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "zip_low",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: true },
        "Interleave the lower half elements of two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a0, b0, a1, b1]`.",
    ),
    Op::new(
        "zip_high",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: false },
        "Interleave the upper half elements of two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a2, b2, a3, b3]`.",
    ),
    Op::new(
        "unzip_low",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: true },
        "Extract even-indexed elements from two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a0, a2, b0, b2]`.",
    ),
    Op::new(
        "unzip_high",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: false },
        "Extract odd-indexed elements from two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a1, a3, b1, b3]`.",
    ),
    Op::new(
        "max",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Return the element-wise maximum of two vectors.\n\n\
        If either operand is NaN, the result for that lane is implementation-defined-- it could be either the first or second operand. See `max_precise` for a version that returns the non-NaN operand if only one is NaN.\n\n\
        If one operand is positive zero and the other is negative zero, the result is also implementation-defined, and it could be either one.",
    ),
    Op::new(
        "min",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Return the element-wise minimum of two vectors.\n\n\
        If either operand is NaN, the result for that lane is implementation-defined-- it could be either the first or second operand. See `min_precise` for a version that returns the non-NaN operand if only one is NaN.\n\n\
        If one operand is positive zero and the other is negative zero, the result is also implementation-defined, and it could be either one.",
    ),
    Op::new(
        "max_precise",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Return the element-wise maximum of two vectors.\n\n\
        If one operand is a quiet NaN and the other is not, this operation will choose the non-NaN operand.\n\n\
        If one operand is positive zero and the other is negative zero, the result is implementation-defined, and it could be either one.\n\n\
        If an operand is a *signaling* NaN, the result is not just implementation-defined, but fully non-deterministic: it may be either NaN or the non-NaN operand.\n\
        Signaling NaN values are not produced by floating-point math operations, only from manual initialization with specific bit patterns. You probably don't need to worry about them.",
    ),
    Op::new(
        "min_precise",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Return the element-wise minimum of two vectors.\n\n\
        If one operand is a quiet NaN and the other is not, this operation will choose the non-NaN operand.\n\n\
        If one operand is positive zero and the other is negative zero, the result is implementation-defined, and it could be either one.\n\n\
        If an operand is a *signaling* NaN, the result is not just implementation-defined, but fully non-deterministic: it may be either NaN or the non-NaN operand.\n\
        Signaling NaN values are not produced by floating-point math operations, only from manual initialization with specific bit patterns. You probably don't need to worry about them.",
    ),
    Op::new(
        "mul_add",
        OpKind::VecTraitMethod,
        OpSig::Ternary,
        "Compute `({arg0} * {arg1}) + {arg2}` (fused multiply-add) for each element.\n\n\
        Depending on hardware support, the result may be computed with only one rounding error, or may be implemented as a regular multiply followed by an add, which will result in two rounding errors.",
    ),
    Op::new(
        "mul_sub",
        OpKind::VecTraitMethod,
        OpSig::Ternary,
        "Compute `({arg0} * {arg1}) - {arg2}` (fused multiply-subtract) for each element.\n\n\
        Depending on hardware support, the result may be computed with only one rounding error, or may be implemented as a regular multiply followed by a subtract, which will result in two rounding errors.",
    ),
    Op::new(
        "floor",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Return the largest integer less than or equal to each element, that is, round towards negative infinity.",
    ),
    Op::new(
        "ceil",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Return the smallest integer greater than or equal to each element, that is, round towards positive infinity.",
    ),
    Op::new(
        "round_ties_even",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Round each element to the nearest integer, with ties rounding to the nearest even integer.\n\n\
        There is no corresponding `round` operation. Rust's `round` operation rounds ties away from zero, a behavior it inherited from C. That behavior is not implemented across all platforms, whereas round-ties-even is.",
    ),
    Op::new(
        "fract",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Return the fractional part of each element.\n\nThis is equivalent to `{arg0} - {arg0}.trunc()`.",
    ),
    Op::new(
        "trunc",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Return the integer part of each element, rounding towards zero.",
    ),
    Op::new(
        "select",
        OpKind::OwnTrait,
        OpSig::Select,
        "Select elements from {arg1} and {arg2} based on the mask operand {arg0}.\n\n\
    This operation's behavior is unspecified if each lane of {arg0} is not the all-zeroes or all-ones bit pattern. See the [`Select`] trait's documentation for more information.",
    ),
];

const INT_OPS: &[Op] = &[
    Op::new(
        "add",
        OpKind::Overloaded(CoreOpTrait::Add),
        OpSig::Binary,
        "Add two vectors element-wise, wrapping on overflow.",
    ),
    Op::new(
        "sub",
        OpKind::Overloaded(CoreOpTrait::Sub),
        OpSig::Binary,
        "Subtract two vectors element-wise, wrapping on overflow.",
    ),
    Op::new(
        "mul",
        OpKind::Overloaded(CoreOpTrait::Mul),
        OpSig::Binary,
        "Multiply two vectors element-wise, wrapping on overflow.",
    ),
    Op::new(
        "and",
        OpKind::Overloaded(CoreOpTrait::BitAnd),
        OpSig::Binary,
        "Compute the bitwise AND of two vectors.",
    ),
    Op::new(
        "or",
        OpKind::Overloaded(CoreOpTrait::BitOr),
        OpSig::Binary,
        "Compute the bitwise OR of two vectors.",
    ),
    Op::new(
        "xor",
        OpKind::Overloaded(CoreOpTrait::BitXor),
        OpSig::Binary,
        "Compute the bitwise XOR of two vectors.",
    ),
    Op::new(
        "not",
        OpKind::Overloaded(CoreOpTrait::Not),
        OpSig::Unary,
        "Compute the bitwise NOT of the vector.",
    ),
    Op::new(
        "shl",
        OpKind::Overloaded(CoreOpTrait::Shl),
        OpSig::Shift,
        "Shift each element left by the given number of bits.\n\n\
        Bits shifted out of the left side are discarded, and zeros are shifted in on the right.",
    ),
    Op::new(
        "shlv",
        OpKind::Overloaded(CoreOpTrait::ShlVectored),
        OpSig::Binary,
        "Shift each element left by the given number of bits.\n\n\
        Bits shifted out of the left side are discarded, and zeros are shifted in on the right.\n\n\
        This operation is not implemented in hardware on all platforms. On WebAssembly, and on x86 platforms without AVX2, this will use a fallback scalar implementation.",
    ),
    Op::new(
        "shr",
        OpKind::Overloaded(CoreOpTrait::Shr),
        OpSig::Shift,
        "Shift each element right by the given number of bits.\n\n\
        For unsigned integers, zeros are shifted in on the left. For signed integers, the sign bit is replicated.",
    ),
    Op::new(
        "shrv",
        OpKind::Overloaded(CoreOpTrait::ShrVectored),
        OpSig::Binary,
        "Shift each element right by the corresponding element in another vector.\n\n\
        For unsigned integers, zeros are shifted in on the left. For signed integers, the sign bit is replicated.\n\n\
        This operation is not implemented in hardware on all platforms. On WebAssembly, and on x86 platforms without AVX2, this will use a fallback scalar implementation.",
    ),
    Op::new(
        "simd_eq",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for equality.\n\n\
        Returns a mask where each element is all ones if the corresponding elements are equal, and all zeroes if not.",
    ),
    Op::new(
        "simd_lt",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for less than.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is less than `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_le",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for less than or equal.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is less than or equal to `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_ge",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for greater than or equal.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is greater than or equal to `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_gt",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for greater than.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is greater than `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "zip_low",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: true },
        "Interleave the lower half elements of two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a0, b0, a1, b1]`.",
    ),
    Op::new(
        "zip_high",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: false },
        "Interleave the upper half elements of two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a2, b2, a3, b3]`.",
    ),
    Op::new(
        "unzip_low",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: true },
        "Extract even-indexed elements from two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a0, a2, b0, b2]`.",
    ),
    Op::new(
        "unzip_high",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: false },
        "Extract odd-indexed elements from two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a1, a3, b1, b3]`.",
    ),
    Op::new(
        "select",
        OpKind::OwnTrait,
        OpSig::Select,
        "Select elements from {arg1} and {arg2} based on the mask operand {arg0}.\n\n\
    This operation's behavior is unspecified if each lane of {arg0} is not the all-zeroes or all-ones bit pattern. See the [`Select`] trait's documentation for more information.",
    ),
    Op::new(
        "min",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Return the element-wise minimum of two vectors.",
    ),
    Op::new(
        "max",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Return the element-wise maximum of two vectors.",
    ),
];

// Long blurb shared between all the mask reduction operations. Needs to be a macro because consts don't work in the
// `concat!` macro.
macro_rules! mask_reduce_blurb {
    () => {
        "Behavior on mask elements that are not all zeroes or all ones is unspecified. It may vary depending on architecture, feature level, the mask elements' width, the mask vector's width, or library version.\n\n\
        The behavior is also not guaranteed to be logically consistent if mask elements are not all zeroes or all ones. `any_true` may not return the same result as `!all_false`, and `all_true` may not return the same result as `!any_false`.\n\n\
        The [`select`](crate::Select::select) operation also has unspecified behavior for mask elements that are not all zeroes or all ones. That behavior may not match the behavior of this operation."
    }
}

const MASK_OPS: &[Op] = &[
    Op::new(
        "and",
        OpKind::Overloaded(CoreOpTrait::BitAnd),
        OpSig::Binary,
        "Compute the logical AND of two masks.",
    ),
    Op::new(
        "or",
        OpKind::Overloaded(CoreOpTrait::BitOr),
        OpSig::Binary,
        "Compute the logical OR of two masks.",
    ),
    Op::new(
        "xor",
        OpKind::Overloaded(CoreOpTrait::BitXor),
        OpSig::Binary,
        "Compute the logical XOR of two masks.",
    ),
    Op::new(
        "not",
        OpKind::Overloaded(CoreOpTrait::Not),
        OpSig::Unary,
        "Compute the logical NOT of the mask.",
    ),
    Op::new(
        "select",
        OpKind::OwnTrait,
        OpSig::Select,
        "Select elements from `{arg1}` and `{arg2}` based on the mask operand `{arg0}`.\n\n\
    This operation's behavior is unspecified if each lane of {arg0} is not the all-zeroes or all-ones bit pattern. See the [`Select`] trait's documentation for more information.",
    ),
    Op::new(
        "simd_eq",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compare two vectors element-wise for equality.\n\n\
        Returns a mask where each element is all ones if the corresponding elements are equal, and all zeroes if not.",
    ),
    Op::new(
        "any_true",
        OpKind::VecTraitMethod,
        OpSig::MaskReduce {
            quantifier: Quantifier::Any,
            condition: true,
        },
        concat!(
            "Returns true if any elements in this mask are true (all ones).\n\n",
            mask_reduce_blurb!()
        ),
    ),
    Op::new(
        "all_true",
        OpKind::VecTraitMethod,
        OpSig::MaskReduce {
            quantifier: Quantifier::All,
            condition: true,
        },
        concat!(
            "Returns true if all elements in this mask are true (all ones).\n\n",
            mask_reduce_blurb!()
        ),
    ),
    Op::new(
        "any_false",
        OpKind::VecTraitMethod,
        OpSig::MaskReduce {
            quantifier: Quantifier::Any,
            condition: false,
        },
        concat!(
            "Returns true if any elements in this mask are false (all zeroes).\n\n\
            This is logically equivalent to `!all_true`, but may be faster.\n\n",
            mask_reduce_blurb!()
        ),
    ),
    Op::new(
        "all_false",
        OpKind::VecTraitMethod,
        OpSig::MaskReduce {
            quantifier: Quantifier::All,
            condition: false,
        },
        concat!(
            "Returns true if all elements in this mask are false (all zeroes).\n\n\
            This is logically equivalent to `!any_true`, but may be faster.\n\n",
            mask_reduce_blurb!()
        ),
    ),
];

pub(crate) fn vec_trait_ops_for(scalar: ScalarType) -> Vec<Op> {
    let base = match scalar {
        ScalarType::Float => FLOAT_OPS,
        ScalarType::Int | ScalarType::Unsigned => INT_OPS,
        ScalarType::Mask => MASK_OPS,
    };
    BASE_OPS
        .iter()
        .chain(base.iter())
        .filter(|op| matches!(op.kind, OpKind::VecTraitMethod))
        .copied()
        .collect()
}

const NEGATE_INT: Op = Op::new(
    "neg",
    OpKind::Overloaded(CoreOpTrait::Neg),
    OpSig::Unary,
    "Negate each element of the vector, wrapping on overflow.",
);

pub(crate) fn overloaded_ops_for(scalar: ScalarType) -> Vec<Op> {
    let base = match scalar {
        ScalarType::Float => FLOAT_OPS,
        ScalarType::Int | ScalarType::Unsigned => INT_OPS,
        ScalarType::Mask => MASK_OPS,
    };
    // We prepend the negate operation only for signed integer types.
    BASE_OPS
        .iter()
        .copied()
        .chain((scalar == ScalarType::Int).then_some(NEGATE_INT))
        .chain(base.iter().copied())
        .collect()
}

pub(crate) const F32_TO_U32: Op = Op::new(
    "cvt_u32",
    OpKind::OwnTrait,
    OpSig::Cvt {
        target_ty: ScalarType::Unsigned,
        scalar_bits: 32,
        precise: false,
    },
    "Convert each floating-point element to an unsigned 32-bit integer, truncating towards zero.\n\n\
    Out-of-range values or NaN will produce implementation-defined results.\n\n\
    On x86 platforms, this operation will still be slower than converting to `i32`, because there is no native instruction for converting to `u32` (at least until AVX-512, which is currently not supported).\n\
    If you know your values fit within range of an `i32`, you should convert to an `i32` and cast to your desired datatype afterwards.",
);
pub(crate) const F32_TO_U32_PRECISE: Op = Op::new(
    "cvt_u32_precise",
    OpKind::OwnTrait,
    OpSig::Cvt {
        target_ty: ScalarType::Unsigned,
        scalar_bits: 32,
        precise: true,
    },
    "Convert each floating-point element to an unsigned 32-bit integer, truncating towards zero.\n\n\
    Out-of-range values are saturated to the closest in-range value. NaN becomes 0.",
);
pub(crate) const F32_TO_I32: Op = Op::new(
    "cvt_i32",
    OpKind::OwnTrait,
    OpSig::Cvt {
        target_ty: ScalarType::Int,
        scalar_bits: 32,
        precise: false,
    },
    "Convert each floating-point element to a signed 32-bit integer, truncating towards zero.\n\n\
    Out-of-range values or NaN will produce implementation-defined results.",
);
pub(crate) const F32_TO_I32_PRECISE: Op = Op::new(
    "cvt_i32_precise",
    OpKind::OwnTrait,
    OpSig::Cvt {
        target_ty: ScalarType::Int,
        scalar_bits: 32,
        precise: true,
    },
    "Convert each floating-point element to a signed 32-bit integer, truncating towards zero.\n\n\
    Out-of-range values are saturated to the closest in-range value. NaN becomes 0.",
);
pub(crate) const U32_TO_F32: Op = Op::new(
    "cvt_f32",
    OpKind::OwnTrait,
    OpSig::Cvt {
        target_ty: ScalarType::Float,
        scalar_bits: 32,
        precise: false,
    },
    "Convert each unsigned 32-bit integer element to a floating-point value.\n\n\
    Values that cannot be exactly represented are rounded to the nearest representable value.",
);
pub(crate) const I32_TO_F32: Op = Op::new(
    "cvt_f32",
    OpKind::OwnTrait,
    OpSig::Cvt {
        target_ty: ScalarType::Float,
        scalar_bits: 32,
        precise: false,
    },
    "Convert each signed 32-bit integer element to a floating-point value.\n\n\
    Values that cannot be exactly represented are rounded to the nearest representable value.",
);

pub(crate) fn ops_for_type(ty: &VecType) -> Vec<Op> {
    let base = match ty.scalar {
        ScalarType::Float => FLOAT_OPS,
        ScalarType::Int | ScalarType::Unsigned => INT_OPS,
        ScalarType::Mask => MASK_OPS,
    };
    let mut ops: Vec<Op> = BASE_OPS.iter().chain(base.iter()).copied().collect();

    if let Some(combined_ty) = ty.combine_operand() {
        ops.push(Op::new(
            "combine",
            OpKind::OwnTrait,
            OpSig::Combine { combined_ty },
            "Combine two vectors into a single vector with twice the width.\n\n`{arg0}` provides the lower elements and `{arg1}` provides the upper elements.",
        ));
    }
    if let Some(half_ty) = ty.split_operand() {
        ops.push(Op::new(
            "split",
            OpKind::OwnTrait,
            OpSig::Split { half_ty },
            "Split a vector into two vectors of half the width.\n\nReturns a tuple of (lower half, upper half).",
        ));
    }
    if ty.scalar == ScalarType::Int {
        ops.push(NEGATE_INT);
    }

    if ty.scalar == ScalarType::Float {
        if ty.scalar_bits == 64 {
            ops.push(Op::new(
                "reinterpret_f32",
                OpKind::AssociatedOnly,
                OpSig::Reinterpret {
                    target_ty: ScalarType::Float,
                    scalar_bits: 32,
                },
                "Reinterpret the bits of this vector as a vector of `f32` elements.\n\nThe number of elements in the result is twice that of the input.",
            ));
        } else {
            ops.push(Op::new(
                "reinterpret_f64",
                OpKind::AssociatedOnly,
                OpSig::Reinterpret {
                    target_ty: ScalarType::Float,
                    scalar_bits: 64,
                },
                "Reinterpret the bits of this vector as a vector of `f64` elements.\n\nThe number of elements in the result is half that of the input.",
            ));

            ops.push(Op::new(
                "reinterpret_i32",
                OpKind::AssociatedOnly,
                OpSig::Reinterpret {
                    target_ty: ScalarType::Int,
                    scalar_bits: 32,
                },
                "Reinterpret the bits of this vector as a vector of `i32` elements.\n\n\
                This is a bitwise reinterpretation only, and does not perform any conversions.",
            ));
        }

        if ty.scalar_bits == 64 {
            return ops;
        }
    }

    if matches!(ty.scalar, ScalarType::Unsigned | ScalarType::Float) && ty.n_bits() == 512 {
        ops.push(Op::new(
            "load_interleaved_128",
            OpKind::AssociatedOnly,
            OpSig::LoadInterleaved {
                block_size: 128,
                block_count: 4,
            },
            "Load elements from an array with 4-way interleaving.\n\nReads consecutive elements and deinterleaves them into a single vector.",
        ));
    }

    if matches!(ty.scalar, ScalarType::Unsigned | ScalarType::Float) && ty.n_bits() == 512 {
        ops.push(Op::new(
            "store_interleaved_128",
            OpKind::AssociatedOnly,
            OpSig::StoreInterleaved {
                block_size: 128,
                block_count: 4,
            },
            "Store elements to an array with 4-way interleaving.\n\nInterleaves the vector elements and writes them consecutively to memory.",
        ));
    }

    if matches!(ty.scalar, ScalarType::Unsigned) {
        if let Some(target_ty) = ty.widened() {
            ops.push(Op::new(
                "widen",
                OpKind::AssociatedOnly,
                OpSig::WidenNarrow { target_ty },
                "Zero-extend each element to a wider integer type.\n\nThe number of elements in the result is half that of the input.",
            ));
        }

        if let Some(target_ty) = ty.narrowed() {
            ops.push(Op::new(
                "narrow",
                OpKind::AssociatedOnly,
                OpSig::WidenNarrow { target_ty },
                "Truncate each element to a narrower integer type.\n\nThe number of elements in the result is twice that of the input.",
            ));
        }
    }

    if valid_reinterpret(ty, ScalarType::Unsigned, 8) {
        ops.push(Op::new(
            "reinterpret_u8",
            OpKind::AssociatedOnly,
            OpSig::Reinterpret {
                target_ty: ScalarType::Unsigned,
                scalar_bits: 8,
            },
            "Reinterpret the bits of this vector as a vector of `u8` elements.\n\nThe total bit width is preserved; the number of elements changes accordingly.",
        ));
    }

    if valid_reinterpret(ty, ScalarType::Unsigned, 32) {
        ops.push(Op::new(
            "reinterpret_u32",
            OpKind::AssociatedOnly,
            OpSig::Reinterpret {
                target_ty: ScalarType::Unsigned,
                scalar_bits: 32,
            },
            "Reinterpret the bits of this vector as a vector of `u32` elements.\n\nThe total bit width is preserved; the number of elements changes accordingly.",
        ));
    }

    match (ty.scalar, ty.scalar_bits) {
        (ScalarType::Float, 32) => {
            ops.push(F32_TO_U32);
            ops.push(F32_TO_U32_PRECISE);
            ops.push(F32_TO_I32);
            ops.push(F32_TO_I32_PRECISE);
        }
        (ScalarType::Unsigned, 32) => ops.push(U32_TO_F32),
        (ScalarType::Int, 32) => ops.push(I32_TO_F32),
        _ => (),
    }

    ops
}

/// Operations on SIMD types that correspond to `core::ops` traits for overloadable operators.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CoreOpTrait {
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    BitAnd,
    BitOr,
    BitXor,
    Not,
    Shl,
    ShlVectored,
    Shr,
    ShrVectored,
}

impl CoreOpTrait {
    pub(crate) fn trait_name(&self) -> &'static str {
        match self {
            Self::Neg => "Neg",
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::BitAnd => "BitAnd",
            Self::BitOr => "BitOr",
            Self::BitXor => "BitXor",
            Self::Not => "Not",
            Self::Shl | Self::ShlVectored => "Shl",
            Self::Shr | Self::ShrVectored => "Shr",
        }
    }

    pub(crate) fn op_fn(&self) -> &'static str {
        match self {
            Self::BitAnd => "bitand",
            Self::BitOr => "bitor",
            Self::BitXor => "bitxor",
            Self::ShlVectored => "shl",
            Self::ShrVectored => "shr",
            _ => self.simd_name(),
        }
    }

    pub(crate) fn simd_name(&self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::BitAnd => "and",
            Self::BitOr => "or",
            Self::BitXor => "xor",
            Self::Not => "not",
            Self::Shl => "shl",
            Self::ShlVectored => "shlv",
            Self::Shr => "shr",
            Self::ShrVectored => "shrv",
        }
    }

    fn is_unary(&self) -> bool {
        matches!(self, Self::Neg | Self::Not)
    }

    pub(crate) fn trait_bounds(&self) -> impl Iterator<Item = TokenStream> {
        let trait_name = Ident::new(self.trait_name(), Span::call_site());
        let trait_name_assign = format_ident!("{trait_name}Assign");
        match self {
            // Shifts always use a u32 as the shift amount
            Self::Shl | Self::Shr => vec![
                quote! { core::ops::#trait_name<u32, Output = Self> },
                quote! { core::ops::#trait_name_assign<u32> },
            ]
            .into_iter(),
            Self::ShlVectored | Self::ShrVectored => vec![
                quote! { core::ops::#trait_name<Output = Self> },
                quote! { core::ops::#trait_name_assign },
            ]
            .into_iter(),
            _ if self.is_unary() => {
                vec![quote! { core::ops::#trait_name<Output = Self> }].into_iter()
            }
            _ => vec![
                quote! { core::ops::#trait_name<Output = Self> },
                quote! { core::ops::#trait_name_assign },
                quote! { core::ops::#trait_name<Self::Element, Output = Self> },
                quote! { core::ops::#trait_name_assign<Self::Element> },
            ]
            .into_iter(),
        }
    }
}

impl OpSig {
    /// Determine whether a given operation should defer to the generic split/combine implementation, for a given vector
    /// type and the maximum native vector width.
    pub(crate) fn should_use_generic_op(&self, vec_ty: &VecType, native_width: usize) -> bool {
        // For widen/narrow operations, we care about the *target* type's width.
        if let Self::WidenNarrow { target_ty } = self
            && target_ty.n_bits() <= native_width
        {
            return false;
        }

        // These operations need to work on the full vector type.
        if matches!(
            self,
            Self::Split { .. }
                | Self::Combine { .. }
                | Self::LoadInterleaved { .. }
                | Self::StoreInterleaved { .. }
                | Self::FromArray { .. }
                | Self::AsArray { .. }
        ) {
            return false;
        }

        // Otherwise, defer to split/combine if this is a wider operation than natively supported.
        if vec_ty.n_bits() <= native_width {
            return false;
        }

        true
    }

    fn simd_trait_arg_names(&self) -> &'static [&'static str] {
        match self {
            Self::Splat | Self::FromArray { .. } => &["val"],
            Self::Unary
            | Self::Split { .. }
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. }
            | Self::MaskReduce { .. }
            | Self::AsArray { .. }
            | Self::FromBytes
            | Self::ToBytes => &["a"],
            Self::Binary
            | Self::Compare
            | Self::Combine { .. }
            | Self::Zip { .. }
            | Self::Unzip { .. } => &["a", "b"],
            Self::Ternary | Self::Select => &["a", "b", "c"],
            Self::Shift => &["a", "shift"],
            Self::LoadInterleaved { .. } => &["src"],
            Self::StoreInterleaved { .. } => &["a", "dest"],
        }
    }
    fn vec_trait_arg_names(&self) -> &'static [&'static str] {
        match self {
            Self::Splat
            | Self::LoadInterleaved { .. }
            | Self::StoreInterleaved { .. }
            | Self::FromArray { .. }
            | Self::FromBytes { .. } => &[],
            Self::Unary
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. }
            | Self::MaskReduce { .. }
            | Self::AsArray { .. }
            | Self::ToBytes => &["self"],
            Self::Binary | Self::Compare | Self::Zip { .. } | Self::Unzip { .. } => {
                &["self", "rhs"]
            }
            Self::Shift => &["self", "shift"],
            Self::Ternary => &["self", "op1", "op2"],
            Self::Select | Self::Split { .. } | Self::Combine { .. } => &[],
        }
    }

    pub(crate) fn forwarding_call_args(&self) -> Option<TokenStream> {
        let arg_names = self
            .vec_trait_arg_names()
            .iter()
            .map(|n| Ident::new(n, Span::call_site()))
            .collect::<Vec<_>>();
        let args = match self {
            Self::Unary | Self::MaskReduce { .. } => {
                let arg0 = &arg_names[0];
                quote! { #arg0 }
            }
            Self::Binary
            | Self::Compare
            | Self::Combine { .. }
            | Self::Zip { .. }
            | Self::Unzip { .. } => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                quote! { #arg0, #arg1.simd_into(self.simd) }
            }
            Self::Ternary => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let arg2 = &arg_names[2];
                quote! { #arg0, #arg1.simd_into(self.simd), #arg2.simd_into(self.simd) }
            }
            Self::Splat
            | Self::Select
            | Self::Split { .. }
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. }
            | Self::Shift
            | Self::LoadInterleaved { .. }
            | Self::StoreInterleaved { .. }
            | Self::FromArray { .. }
            | Self::AsArray { .. }
            | Self::FromBytes
            | Self::ToBytes => return None,
        };
        Some(args)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TyFlavor {
    /// Types for methods in the `Simd` trait; `f32x4<Self>`
    SimdTrait,
    /// Types for methods in the vec trait; `f32x4<S>`
    VecImpl,
}

fn load_interleaved_arg_ty(block_size: u16, block_count: u16, vec_ty: &VecType) -> TokenStream {
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let len = (block_size * block_count) as usize / vec_ty.scalar_bits;
    quote! { &[#scalar; #len] }
}

fn store_interleaved_arg_ty(block_size: u16, block_count: u16, vec_ty: &VecType) -> TokenStream {
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let len = (block_size * block_count) as usize / vec_ty.scalar_bits;
    quote! { &mut [#scalar; #len] }
}

pub(crate) fn valid_reinterpret(src: &VecType, dst_scalar: ScalarType, dst_bits: usize) -> bool {
    if src.scalar == dst_scalar && src.scalar_bits == dst_bits {
        return false;
    }

    if matches!(src.scalar, ScalarType::Mask) {
        return false;
    }

    true
}
