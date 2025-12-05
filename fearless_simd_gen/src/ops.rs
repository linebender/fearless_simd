// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use anyhow::{Context, anyhow};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use std::fmt::Write;

use crate::types::{ScalarType, VecType};

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
}

const FLOAT_OPS: &[Op] = &[
    Op::new(
        "splat",
        OpKind::BaseTraitMethod,
        OpSig::Splat,
        "Creates a SIMD vector with all elements set to the given value.",
    ),
    Op::new(
        "abs",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Computes the absolute value of each element.",
    ),
    Op::new(
        "neg",
        OpKind::Overloaded(CoreOpTrait::Neg),
        OpSig::Unary,
        "Negates each element of the vector.",
    ),
    Op::new(
        "sqrt",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Computes the square root of each element.\n\n\
        Negative elements other than `-0.0` will become NaN.",
    ),
    Op::new(
        "add",
        OpKind::Overloaded(CoreOpTrait::Add),
        OpSig::Binary,
        "Adds two vectors element-wise.",
    ),
    Op::new(
        "sub",
        OpKind::Overloaded(CoreOpTrait::Sub),
        OpSig::Binary,
        "Subtracts two vectors element-wise.",
    ),
    Op::new(
        "mul",
        OpKind::Overloaded(CoreOpTrait::Mul),
        OpSig::Binary,
        "Multiplies two vectors element-wise.",
    ),
    Op::new(
        "div",
        OpKind::Overloaded(CoreOpTrait::Div),
        OpSig::Binary,
        "Divides two vectors element-wise.",
    ),
    Op::new(
        "copysign",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Returns a vector with the magnitude of `{arg0}` and the sign of `{arg1}` for each element.\n\n\
        This operation copies the sign bit, so if an input element is NaN, the output element will be a NaN with the same payload and a copied sign bit.",
    ),
    Op::new(
        "simd_eq",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for equality.\n\n\
        Returns a mask where each element is all ones if the corresponding elements are equal, and all zeroes if not.",
    ),
    Op::new(
        "simd_lt",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for less than.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is less than `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_le",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for less than or equal.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is less than or equal to `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_ge",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for greater than or equal.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is greater than or equal to `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_gt",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for greater than.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is greater than `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "zip_low",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: true },
        "Interleaves the lower half elements of two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a0, b0, a1, b1]`.",
    ),
    Op::new(
        "zip_high",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: false },
        "Interleaves the upper half elements of two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a2, b2, a3, b3]`.",
    ),
    Op::new(
        "unzip_low",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: true },
        "Extracts even-indexed elements from two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a0, a2, b0, b2]`.",
    ),
    Op::new(
        "unzip_high",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: false },
        "Extracts odd-indexed elements from two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a1, a3, b1, b3]`.",
    ),
    Op::new(
        "max",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Returns the element-wise maximum of two vectors.\n\n\
        If either operand is NaN, the result for that lane is implementation-defined-- it could be either the first or second operand. See `max_precise` for a version that returns the non-NaN operand if only one is NaN.\n\n\
        If one operand is positive zero and the other is negative zero, the result is also implementation-defined, and it could be either one.",
    ),
    Op::new(
        "min",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Returns the element-wise minimum of two vectors.\n\n\
        If either operand is NaN, the result for that lane is implementation-defined-- it could be either the first or second operand. See `min_precise` for a version that returns the non-NaN operand if only one is NaN.\n\n\
        If one operand is positive zero and the other is negative zero, the result is also implementation-defined, and it could be either one.",
    ),
    Op::new(
        "max_precise",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Returns the element-wise maximum of two vectors.\n\n\
        If one operand is a quiet NaN and the other is not, this operation will choose the non-NaN operand.\n\n\
        If one operand is positive zero and the other is negative zero, the result is implementation-defined, and it could be either one.\n\n\
        If an operand is a *signaling* NaN, the result is not just implementation-defined, but fully non-determinstic: it may be either NaN or the non-NaN operand.\n\
        Signaling NaN values are not produced by floating-point math operations, only from manual initialization with specific bit patterns. You probably don't need to worry about them.",
    ),
    Op::new(
        "min_precise",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Returns the element-wise minimum of two vectors.\n\n\
        If one operand is a quiet NaN and the other is not, this operation will choose the non-NaN operand.\n\n\
        If one operand is positive zero and the other is negative zero, the result is implementation-defined, and it could be either one.\n\n\
        If an operand is a *signaling* NaN, the result is not just implementation-defined, but fully non-determinstic: it may be either NaN or the non-NaN operand.\n\
        Signaling NaN values are not produced by floating-point math operations, only from manual initialization with specific bit patterns. You probably don't need to worry about them.",
    ),
    Op::new(
        "madd",
        OpKind::VecTraitMethod,
        OpSig::Ternary,
        "Computes `({arg0} * {arg1}) + {arg2}` (fused multiply-add) for each element.\n\n\
        Depending on hardware support, the result may be computed with only one rounding error, or may be implemented as a regular multiply followed by an add, which will result in two rounding errors.",
    ),
    Op::new(
        "msub",
        OpKind::VecTraitMethod,
        OpSig::Ternary,
        "Computes `({arg0} * {arg1}) + {arg2}` (fused multiply-subtract) for each element.\n\n\
        Depending on hardware support, the result may be computed with only one rounding error, or may be implemented as a regular multiply followed by a subtract, which will result in two rounding errors.",
    ),
    Op::new(
        "floor",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Returns the largest integer less than or equal to each element, that is, it rounds towards negative infinity.",
    ),
    Op::new(
        "ceil",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Returns the smallest integer greater than or equal to each element, that is, it rounds towards positive infinity.",
    ),
    Op::new(
        "round_ties_even",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Rounds each element to the nearest integer, with ties rounding to the nearest even integer.\n\n\
        There is no corresponding `round` operation. Rust's `round` operation rounds ties away from zero, a behavior it inherited from C. That behavior is not implemented across all platforms, whereas round-ties-even is.",
    ),
    Op::new(
        "fract",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Returns the fractional part of each element.\n\nThis is equivalent to `{arg0} - {arg0}.trunc()`.",
    ),
    Op::new(
        "trunc",
        OpKind::VecTraitMethod,
        OpSig::Unary,
        "Returns the integer part of each element, rounding towards zero.",
    ),
    Op::new(
        "select",
        OpKind::OwnTrait,
        OpSig::Select,
        "Selects elements from {arg1} and {arg2} based on the mask operand {arg0}.\n\n\
    This operation's behavior is unspecified if each lane of {arg0} is not the all-zeroes or all-ones bit pattern. See the [`Select`] trait's documentation for more information.",
    ),
];

const INT_OPS: &[Op] = &[
    Op::new(
        "splat",
        OpKind::BaseTraitMethod,
        OpSig::Splat,
        "Creates a SIMD vector with all elements set to the given value.",
    ),
    Op::new(
        "add",
        OpKind::Overloaded(CoreOpTrait::Add),
        OpSig::Binary,
        "Adds two vectors element-wise, wrapping on overflow.",
    ),
    Op::new(
        "sub",
        OpKind::Overloaded(CoreOpTrait::Sub),
        OpSig::Binary,
        "Subtracts two vectors element-wise, wrapping on overflow.",
    ),
    Op::new(
        "mul",
        OpKind::Overloaded(CoreOpTrait::Mul),
        OpSig::Binary,
        "Multiplies two vectors element-wise, wrapping on overflow.",
    ),
    Op::new(
        "and",
        OpKind::Overloaded(CoreOpTrait::BitAnd),
        OpSig::Binary,
        "Computes the bitwise AND of two vectors.",
    ),
    Op::new(
        "or",
        OpKind::Overloaded(CoreOpTrait::BitOr),
        OpSig::Binary,
        "Computes the bitwise OR of two vectors.",
    ),
    Op::new(
        "xor",
        OpKind::Overloaded(CoreOpTrait::BitXor),
        OpSig::Binary,
        "Computes the bitwise XOR of two vectors.",
    ),
    Op::new(
        "not",
        OpKind::Overloaded(CoreOpTrait::Not),
        OpSig::Unary,
        "Computes the bitwise NOT of the vector.",
    ),
    Op::new(
        "shl",
        OpKind::Overloaded(CoreOpTrait::Shl),
        OpSig::Shift,
        "Shifts each element left by the given number of bits.\n\n\
        Bits shifted out of the left side are discarded, and zeros are shifted in on the right.",
    ),
    Op::new(
        "shr",
        OpKind::Overloaded(CoreOpTrait::Shr),
        OpSig::Shift,
        "Shifts each element right by the given number of bits.\n\n\
        For unsigned integers, zeros are shifted in on the left. For signed integers, the sign bit is replicated.",
    ),
    Op::new(
        "shrv",
        OpKind::Overloaded(CoreOpTrait::ShrVectored),
        OpSig::Binary,
        "Shifts each element right by the corresponding element in another vector.\n\n\
        For unsigned integers, zeros are shifted in on the left. For signed integers, the sign bit is replicated.\n\n\
        This operation is not implemented in hardware on all platforms. On WebAssembly, and on x86 platforms without AVX2, this will use a fallback scalar implementation.",
    ),
    Op::new(
        "simd_eq",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for equality.\n\n\
        Returns a mask where each element is all ones if the corresponding elements are equal, and all zeroes if not.",
    ),
    Op::new(
        "simd_lt",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for less than.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is less than `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_le",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for less than or equal.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is less than or equal to `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_ge",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for greater than or equal.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is greater than or equal to `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "simd_gt",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for greater than.\n\n\
        Returns a mask where each element is all ones if `{arg0}` is greater than `{arg1}`, and all zeroes if not.",
    ),
    Op::new(
        "zip_low",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: true },
        "Interleaves the lower half elements of two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a0, b0, a1, b1]`.",
    ),
    Op::new(
        "zip_high",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: false },
        "Interleaves the upper half elements of two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a2, b2, a3, b3]`.",
    ),
    Op::new(
        "unzip_low",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: true },
        "Extracts even-indexed elements from two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a0, a2, b0, b2]`.",
    ),
    Op::new(
        "unzip_high",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: false },
        "Extracts odd-indexed elements from two vectors.\n\n\
        For vectors `[a0, a1, a2, a3]` and `[b0, b1, b2, b3]`, returns `[a1, a3, b1, b3]`.",
    ),
    Op::new(
        "select",
        OpKind::OwnTrait,
        OpSig::Select,
        "Selects elements from {arg1} and {arg2} based on the mask operand {arg0}.\n\n\
    This operation's behavior is unspecified if each lane of {arg0} is not the all-zeroes or all-ones bit pattern. See the [`Select`] trait's documentation for more information.",
    ),
    Op::new(
        "min",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Returns the element-wise minimum of two vectors.",
    ),
    Op::new(
        "max",
        OpKind::VecTraitMethod,
        OpSig::Binary,
        "Returns the element-wise maximum of two vectors.",
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
        "splat",
        OpKind::BaseTraitMethod,
        OpSig::Splat,
        "Creates a SIMD mask with all elements set to the given value.",
    ),
    Op::new(
        "and",
        OpKind::Overloaded(CoreOpTrait::BitAnd),
        OpSig::Binary,
        "Computes the logical AND of two masks.",
    ),
    Op::new(
        "or",
        OpKind::Overloaded(CoreOpTrait::BitOr),
        OpSig::Binary,
        "Computes the logical OR of two masks.",
    ),
    Op::new(
        "xor",
        OpKind::Overloaded(CoreOpTrait::BitXor),
        OpSig::Binary,
        "Computes the logical XOR of two masks.",
    ),
    Op::new(
        "not",
        OpKind::Overloaded(CoreOpTrait::Not),
        OpSig::Unary,
        "Computes the logical NOT of the mask.",
    ),
    Op::new(
        "select",
        OpKind::OwnTrait,
        OpSig::Select,
        "Selects elements from `{arg1}` and `{arg2}` based on the mask operand `{arg0}`.\n\n\
    This operation's behavior is unspecified if each lane of {arg0} is not the all-zeroes or all-ones bit pattern. See the [`Select`] trait's documentation for more information.",
    ),
    Op::new(
        "simd_eq",
        OpKind::VecTraitMethod,
        OpSig::Compare,
        "Compares two vectors element-wise for equality.\n\n\
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
    base.iter()
        .filter(|op| matches!(op.kind, OpKind::VecTraitMethod))
        .copied()
        .collect()
}

const NEGATE_INT: Op = Op::new(
    "neg",
    OpKind::Overloaded(CoreOpTrait::Neg),
    OpSig::Unary,
    "Negates each element of the vector, wrapping on overflow.",
);

pub(crate) fn overloaded_ops_for(scalar: ScalarType) -> Vec<(CoreOpTrait, OpSig, &'static str)> {
    let base = match scalar {
        ScalarType::Float => FLOAT_OPS,
        ScalarType::Int | ScalarType::Unsigned => INT_OPS,
        ScalarType::Mask => MASK_OPS,
    };
    // We prepend the negate operation only for signed integer types.
    (scalar == ScalarType::Int)
        .then_some(NEGATE_INT)
        .into_iter()
        .chain(base.iter().copied())
        .filter_map(|op| match op.kind {
            OpKind::Overloaded(core_op) => Some((core_op, op.sig, op.doc)),
            _ => None,
        })
        .collect()
}

pub(crate) fn ops_for_type(ty: &VecType) -> Vec<Op> {
    let base = match ty.scalar {
        ScalarType::Float => FLOAT_OPS,
        ScalarType::Int | ScalarType::Unsigned => INT_OPS,
        ScalarType::Mask => MASK_OPS,
    };
    let mut ops = base.to_vec();

    if let Some(combined_ty) = ty.combine_operand() {
        ops.push(Op::new(
            "combine",
            OpKind::OwnTrait,
            OpSig::Combine { combined_ty },
            "Combines two vectors into a single vector with twice the width.\n\n`{arg0}` provides the lower elements and `{arg1}` provides the upper elements.",
        ));
    }
    if let Some(half_ty) = ty.split_operand() {
        ops.push(Op::new(
            "split",
            OpKind::OwnTrait,
            OpSig::Split { half_ty },
            "Splits a vector into two vectors of half the width.\n\nReturns a tuple of (lower half, upper half).",
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
                "Reinterprets the bits of this vector as a vector of `f32` elements.\n\nThe number of elements in the result is twice that of the input.",
            ));
        } else {
            ops.push(Op::new(
                "reinterpret_f64",
                OpKind::AssociatedOnly,
                OpSig::Reinterpret {
                    target_ty: ScalarType::Float,
                    scalar_bits: 64,
                },
                "Reinterprets the bits of this vector as a vector of `f64` elements.\n\nThe number of elements in the result is half that of the input.",
            ));

            ops.push(Op::new(
                "reinterpret_i32",
                OpKind::AssociatedOnly,
                OpSig::Reinterpret {
                    target_ty: ScalarType::Int,
                    scalar_bits: 32,
                },
                "Reinterprets the bits of this vector as a vector of `i32` elements.\n\n\
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
            "Loads elements from an array with 4-way interleaving.\n\nReads consecutive elements and deinterleaves them into a single vector.",
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
            "Stores elements to an array with 4-way interleaving.\n\nInterleaves the vector elements and writes them consecutively to memory.",
        ));
    }

    if matches!(ty.scalar, ScalarType::Unsigned) {
        if let Some(target_ty) = ty.widened() {
            ops.push(Op::new(
                "widen",
                OpKind::AssociatedOnly,
                OpSig::WidenNarrow { target_ty },
                "Zero-extends each element to a wider integer type.\n\nThe number of elements in the result is half that of the input.",
            ));
        }

        if let Some(target_ty) = ty.narrowed() {
            ops.push(Op::new(
                "narrow",
                OpKind::AssociatedOnly,
                OpSig::WidenNarrow { target_ty },
                "Truncates each element to a narrower integer type.\n\nThe number of elements in the result is twice that of the input.",
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
            "Reinterprets the bits of this vector as a vector of `u8` elements.\n\nThe total bit width is preserved; the number of elements changes accordingly.",
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
            "Reinterprets the bits of this vector as a vector of `u32` elements.\n\nThe total bit width is preserved; the number of elements changes accordingly.",
        ));
    }

    match (ty.scalar, ty.scalar_bits) {
        (ScalarType::Float, 32) => {
            ops.push(Op::new(
                "cvt_u32",
                OpKind::OwnTrait,
                OpSig::Cvt {
                    target_ty: ScalarType::Unsigned,
                    scalar_bits: 32,
                },
                "Converts each floating-point element to an unsigned 32-bit integer, truncating towards zero.\n\n\
                Out-of-range values are saturated to the closest in-range value. NaN becomes 0.",
            ));
            ops.push(Op::new(
                "cvt_i32",
                OpKind::OwnTrait,
                OpSig::Cvt {
                    target_ty: ScalarType::Int,
                    scalar_bits: 32,
                },
                "Converts each floating-point element to a signed 32-bit integer, truncating towards zero.\n\n\
                Out-of-range values are saturated to the closest in-range value. NaN becomes 0.",
            ));
        }
        (ScalarType::Unsigned, 32) => ops.push(Op::new(
            "cvt_f32",
            OpKind::OwnTrait,
            OpSig::Cvt {
                target_ty: ScalarType::Float,
                scalar_bits: 32,
            },
            "Converts each unsigned 32-bit integer element to a floating-point value.\n\n\
            Values that cannot be exactly represented are rounded to the nearest representable value.",
        )),
        (ScalarType::Int, 32) => ops.push(Op::new(
            "cvt_f32",
            OpKind::OwnTrait,
            OpSig::Cvt {
                target_ty: ScalarType::Float,
                scalar_bits: 32,
            },
            "Converts each signed 32-bit integer element to a floating-point value.\n\n\
            Values that cannot be exactly represented are rounded to the nearest representable value.",
        )),
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
            Self::Shl => "Shl",
            Self::Shr | Self::ShrVectored => "Shr",
        }
    }

    pub(crate) fn op_fn(&self) -> &'static str {
        match self {
            Self::BitAnd => "bitand",
            Self::BitOr => "bitor",
            Self::BitXor => "bitxor",
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
            Self::Shl => vec![
                quote! { core::ops::#trait_name<u32, Output = Self> },
                quote! { core::ops::#trait_name_assign<u32> },
            ]
            .into_iter(),
            Self::Shr => vec![
                quote! { core::ops::#trait_name<u32, Output = Self> },
                quote! { core::ops::#trait_name_assign<u32> },
            ]
            .into_iter(),
            Self::ShrVectored => vec![
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
                quote! { core::ops::#trait_name<Element, Output = Self> },
                quote! { core::ops::#trait_name_assign<Element> },
            ]
            .into_iter(),
        }
    }
}

impl OpSig {
    pub(crate) fn simd_trait_arg_names(&self) -> &'static [&'static str] {
        match self {
            Self::Splat => &["val"],
            Self::Unary
            | Self::Split { .. }
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. }
            | Self::MaskReduce { .. } => &["a"],
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
    pub(crate) fn vec_trait_arg_names(&self) -> &'static [&'static str] {
        match self {
            Self::Splat | Self::LoadInterleaved { .. } | Self::StoreInterleaved { .. } => &[],
            Self::Unary
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. }
            | Self::MaskReduce { .. } => &["self"],
            Self::Binary | Self::Compare | Self::Zip { .. } | Self::Unzip { .. } => {
                &["self", "rhs"]
            }
            Self::Shift => &["self", "shift"],
            Self::Ternary => &["self", "op1", "op2"],
            Self::Select | Self::Split { .. } | Self::Combine { .. } => &[],
        }
    }

    pub(crate) fn simd_trait_args(&self, vec_ty: &VecType) -> TokenStream {
        let ty = vec_ty.rust();
        let arg_names = self
            .simd_trait_arg_names()
            .iter()
            .map(|n| Ident::new(n, Span::call_site()))
            .collect::<Vec<_>>();
        match self {
            Self::Splat => {
                let arg0 = &arg_names[0];
                let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
                quote! { self, #arg0: #scalar }
            }
            Self::LoadInterleaved {
                block_size,
                block_count,
            } => {
                let arg0 = &arg_names[0];
                let arg_ty = load_interleaved_arg_ty(*block_size, *block_count, vec_ty);
                quote! { self, #arg0: #arg_ty }
            }
            Self::StoreInterleaved {
                block_size,
                block_count,
            } => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let arg_ty = store_interleaved_arg_ty(*block_size, *block_count, vec_ty);
                quote! { self, #arg0: #ty<Self>, #arg1: #arg_ty }
            }
            Self::Unary
            | Self::Split { .. }
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. }
            | Self::MaskReduce { .. } => {
                let arg0 = &arg_names[0];
                quote! { self, #arg0: #ty<Self> }
            }
            Self::Binary
            | Self::Compare
            | Self::Combine { .. }
            | Self::Zip { .. }
            | Self::Unzip { .. } => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                quote! { self, #arg0: #ty<Self>, #arg1: #ty<Self> }
            }
            Self::Shift => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                quote! { self, #arg0: #ty<Self>, #arg1: u32 }
            }
            Self::Ternary => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let arg2 = &arg_names[2];
                quote! { self, #arg0: #ty<Self>, #arg1: #ty<Self>, #arg2: #ty<Self> }
            }
            Self::Select => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let arg2 = &arg_names[2];
                let mask_ty = vec_ty.mask_ty().rust();
                quote! { self, #arg0: #mask_ty<Self>, #arg1: #ty<Self>, #arg2: #ty<Self> }
            }
        }
    }

    pub(crate) fn vec_trait_args(&self) -> Option<TokenStream> {
        let arg_names = self
            .vec_trait_arg_names()
            .iter()
            .map(|n| Ident::new(n, Span::call_site()))
            .collect::<Vec<_>>();
        let args = match self {
            Self::Splat | Self::LoadInterleaved { .. } | Self::StoreInterleaved { .. } => {
                return None;
            }
            Self::Unary
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. }
            | Self::MaskReduce { .. } => {
                let arg0 = &arg_names[0];
                quote! { #arg0 }
            }
            Self::Binary | Self::Compare | Self::Zip { .. } | Self::Unzip { .. } => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                quote! { #arg0, #arg1: impl SimdInto<Self, S> }
            }
            Self::Shift => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                quote! { #arg0, #arg1: u32 }
            }
            Self::Ternary => {
                let arg0 = &arg_names[0];
                let arg1 = &arg_names[1];
                let arg2 = &arg_names[2];
                quote! { #arg0, #arg1: impl SimdInto<Self, S>, #arg2: impl SimdInto<Self, S> }
            }
            // select is currently done by trait, but maybe we'll implement for
            // masks.
            Self::Select => return None,
            // These signatures involve types not in the Simd trait
            Self::Split { .. } | Self::Combine { .. } => return None,
        };
        Some(args)
    }

    pub(crate) fn forwarding_call_args(&self) -> Option<TokenStream> {
        let args = match self {
            Self::Unary | Self::MaskReduce { .. } => quote! { self },
            Self::Binary
            | Self::Compare
            | Self::Combine { .. }
            | Self::Zip { .. }
            | Self::Unzip { .. } => {
                quote! { self, rhs.simd_into(self.simd) }
            }
            Self::Ternary => {
                quote! { self, op1.simd_into(self.simd), op2.simd_into(self.simd) }
            }
            Self::Splat
            | Self::Select
            | Self::Split { .. }
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. }
            | Self::Shift
            | Self::LoadInterleaved { .. }
            | Self::StoreInterleaved { .. } => return None,
        };
        Some(args)
    }

    pub(crate) fn simd_impl_ret_ty(&self, ty: &VecType) -> TokenStream {
        let quant = quote! { <Self> };
        match self {
            Self::Splat
            | Self::Unary
            | Self::Binary
            | Self::Select
            | Self::Ternary
            | Self::Shift
            | Self::LoadInterleaved { .. } => {
                let rust = ty.rust();
                quote! { #rust #quant }
            }
            Self::Compare => {
                let rust = ty.mask_ty().rust();
                quote! { #rust #quant }
            }
            Self::Combine { combined_ty } => {
                let result = combined_ty.rust();
                quote! { #result #quant }
            }
            Self::Split { half_ty } => {
                let result = half_ty.rust();
                quote! { ( #result #quant, #result #quant ) }
            }
            Self::Zip { .. } | Self::Unzip { .. } => {
                let rust = ty.rust();
                quote! { #rust #quant }
            }
            Self::Cvt {
                target_ty,
                scalar_bits,
            } => {
                let result = VecType::new(*target_ty, *scalar_bits, ty.len).rust();
                quote! { #result #quant }
            }
            Self::Reinterpret {
                target_ty,
                scalar_bits,
            } => {
                let result = ty.reinterpret(*target_ty, *scalar_bits).rust();
                quote! { #result #quant }
            }
            Self::WidenNarrow { target_ty } => {
                let result = target_ty.rust();
                quote! { #result #quant }
            }
            Self::MaskReduce { .. } => quote! { bool },
            Self::StoreInterleaved { .. } => quote! {()},
        }
    }

    pub(crate) fn trait_ret_ty(&self) -> TokenStream {
        match self {
            Self::Compare => quote! { Self::Mask },
            Self::MaskReduce { .. } => quote! { bool },
            _ => quote! { Self },
        }
    }

    pub(crate) fn format_docstring(&self, docstring: &str, flavor: TyFlavor) -> String {
        let arg_names = match flavor {
            TyFlavor::SimdTrait => self.simd_trait_arg_names(),
            TyFlavor::VecImpl => self.vec_trait_arg_names(),
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

        let mut remaining = docstring;
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TyFlavor {
    /// Types for methods in the `Simd` trait; `f32x4<Self>`
    SimdTrait,
    /// Types for methods in the vec trait; `f32x4<S>`
    VecImpl,
}

pub(crate) fn load_interleaved_arg_ty(
    block_size: u16,
    block_count: u16,
    vec_ty: &VecType,
) -> TokenStream {
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let len = (block_size * block_count) as usize / vec_ty.scalar_bits;
    quote! { &[#scalar; #len] }
}

pub(crate) fn store_interleaved_arg_ty(
    block_size: u16,
    block_count: u16,
    vec_ty: &VecType,
) -> TokenStream {
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
