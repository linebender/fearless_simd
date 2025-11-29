// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::TokenStream;
use quote::quote;

use crate::types::{CoreOpTrait, ScalarType, VecType};

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
    pub(crate) method: &'static str,
    pub(crate) kind: OpKind,
    pub(crate) sig: OpSig,
}

impl Op {
    const fn new(method: &'static str, kind: OpKind, sig: OpSig) -> Self {
        Self { method, kind, sig }
    }
}

const FLOAT_OPS: &[Op] = &[
    Op::new("splat", OpKind::BaseTraitMethod, OpSig::Splat),
    Op::new("abs", OpKind::VecTraitMethod, OpSig::Unary),
    Op::new("neg", OpKind::Overloaded(CoreOpTrait::Neg), OpSig::Unary),
    Op::new("sqrt", OpKind::VecTraitMethod, OpSig::Unary),
    Op::new("add", OpKind::Overloaded(CoreOpTrait::Add), OpSig::Binary),
    Op::new("sub", OpKind::Overloaded(CoreOpTrait::Sub), OpSig::Binary),
    Op::new("mul", OpKind::Overloaded(CoreOpTrait::Mul), OpSig::Binary),
    Op::new("div", OpKind::Overloaded(CoreOpTrait::Div), OpSig::Binary),
    Op::new("copysign", OpKind::VecTraitMethod, OpSig::Binary),
    Op::new("simd_eq", OpKind::VecTraitMethod, OpSig::Compare),
    Op::new("simd_lt", OpKind::VecTraitMethod, OpSig::Compare),
    Op::new("simd_le", OpKind::VecTraitMethod, OpSig::Compare),
    Op::new("simd_ge", OpKind::VecTraitMethod, OpSig::Compare),
    Op::new("simd_gt", OpKind::VecTraitMethod, OpSig::Compare),
    Op::new(
        "zip_low",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: true },
    ),
    Op::new(
        "zip_high",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: false },
    ),
    Op::new(
        "unzip_low",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: true },
    ),
    Op::new(
        "unzip_high",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: false },
    ),
    Op::new("max", OpKind::VecTraitMethod, OpSig::Binary),
    Op::new("min", OpKind::VecTraitMethod, OpSig::Binary),
    Op::new("max_precise", OpKind::VecTraitMethod, OpSig::Binary),
    Op::new("min_precise", OpKind::VecTraitMethod, OpSig::Binary),
    Op::new("madd", OpKind::VecTraitMethod, OpSig::Ternary),
    Op::new("msub", OpKind::VecTraitMethod, OpSig::Ternary),
    Op::new("floor", OpKind::VecTraitMethod, OpSig::Unary),
    Op::new("ceil", OpKind::VecTraitMethod, OpSig::Unary),
    Op::new("round_ties_even", OpKind::VecTraitMethod, OpSig::Unary),
    Op::new("fract", OpKind::VecTraitMethod, OpSig::Unary),
    Op::new("trunc", OpKind::VecTraitMethod, OpSig::Unary),
    Op::new("select", OpKind::OwnTrait, OpSig::Select),
];

const INT_OPS: &[Op] = &[
    Op::new("splat", OpKind::BaseTraitMethod, OpSig::Splat),
    Op::new("add", OpKind::Overloaded(CoreOpTrait::Add), OpSig::Binary),
    Op::new("sub", OpKind::Overloaded(CoreOpTrait::Sub), OpSig::Binary),
    Op::new("mul", OpKind::Overloaded(CoreOpTrait::Mul), OpSig::Binary),
    Op::new(
        "and",
        OpKind::Overloaded(CoreOpTrait::BitAnd),
        OpSig::Binary,
    ),
    Op::new("or", OpKind::Overloaded(CoreOpTrait::BitOr), OpSig::Binary),
    Op::new(
        "xor",
        OpKind::Overloaded(CoreOpTrait::BitXor),
        OpSig::Binary,
    ),
    Op::new("not", OpKind::Overloaded(CoreOpTrait::Not), OpSig::Unary),
    Op::new("shl", OpKind::Overloaded(CoreOpTrait::Shl), OpSig::Shift),
    Op::new("shr", OpKind::Overloaded(CoreOpTrait::Shr), OpSig::Shift),
    Op::new(
        "shrv",
        OpKind::Overloaded(CoreOpTrait::ShrVectored),
        OpSig::Binary,
    ),
    Op::new("simd_eq", OpKind::VecTraitMethod, OpSig::Compare),
    Op::new("simd_lt", OpKind::VecTraitMethod, OpSig::Compare),
    Op::new("simd_le", OpKind::VecTraitMethod, OpSig::Compare),
    Op::new("simd_ge", OpKind::VecTraitMethod, OpSig::Compare),
    Op::new("simd_gt", OpKind::VecTraitMethod, OpSig::Compare),
    Op::new(
        "zip_low",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: true },
    ),
    Op::new(
        "zip_high",
        OpKind::VecTraitMethod,
        OpSig::Zip { select_low: false },
    ),
    Op::new(
        "unzip_low",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: true },
    ),
    Op::new(
        "unzip_high",
        OpKind::VecTraitMethod,
        OpSig::Unzip { select_even: false },
    ),
    Op::new("select", OpKind::OwnTrait, OpSig::Select),
    Op::new("min", OpKind::VecTraitMethod, OpSig::Binary),
    Op::new("max", OpKind::VecTraitMethod, OpSig::Binary),
];

const MASK_OPS: &[Op] = &[
    Op::new("splat", OpKind::BaseTraitMethod, OpSig::Splat),
    Op::new(
        "and",
        OpKind::Overloaded(CoreOpTrait::BitAnd),
        OpSig::Binary,
    ),
    Op::new("or", OpKind::Overloaded(CoreOpTrait::BitOr), OpSig::Binary),
    Op::new(
        "xor",
        OpKind::Overloaded(CoreOpTrait::BitXor),
        OpSig::Binary,
    ),
    Op::new("not", OpKind::Overloaded(CoreOpTrait::Not), OpSig::Unary),
    Op::new("select", OpKind::VecTraitMethod, OpSig::Select),
    Op::new("simd_eq", OpKind::VecTraitMethod, OpSig::Compare),
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

pub(crate) fn overloaded_ops_for(scalar: ScalarType) -> Vec<CoreOpTrait> {
    let base = match scalar {
        ScalarType::Float => FLOAT_OPS,
        ScalarType::Int | ScalarType::Unsigned => INT_OPS,
        ScalarType::Mask => MASK_OPS,
    };
    // We prepend the negate operation only for signed integer types.
    (scalar == ScalarType::Int)
        .then(|| CoreOpTrait::Neg)
        .into_iter()
        .chain(base.iter().filter_map(|op| match op.kind {
            OpKind::Overloaded(core_op) => Some(core_op),
            _ => None,
        }))
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
        ));
    }
    if let Some(half_ty) = ty.split_operand() {
        ops.push(Op::new("split", OpKind::OwnTrait, OpSig::Split { half_ty }));
    }
    if ty.scalar == ScalarType::Int {
        ops.push(Op::new(
            "neg",
            OpKind::Overloaded(CoreOpTrait::Neg),
            OpSig::Unary,
        ));
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
            ));
        } else {
            ops.push(Op::new(
                "reinterpret_f64",
                OpKind::AssociatedOnly,
                OpSig::Reinterpret {
                    target_ty: ScalarType::Float,
                    scalar_bits: 64,
                },
            ));

            ops.push(Op::new(
                "reinterpret_i32",
                OpKind::AssociatedOnly,
                OpSig::Reinterpret {
                    target_ty: ScalarType::Int,
                    scalar_bits: 32,
                },
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
        ));
    }

    if matches!(ty.scalar, ScalarType::Unsigned) {
        if let Some(target_ty) = ty.widened() {
            ops.push(Op::new(
                "widen",
                OpKind::AssociatedOnly,
                OpSig::WidenNarrow { target_ty },
            ));
        }

        if let Some(target_ty) = ty.narrowed() {
            ops.push(Op::new(
                "narrow",
                OpKind::AssociatedOnly,
                OpSig::WidenNarrow { target_ty },
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
            ));
            ops.push(Op::new(
                "cvt_i32",
                OpKind::OwnTrait,
                OpSig::Cvt {
                    target_ty: ScalarType::Int,
                    scalar_bits: 32,
                },
            ));
        }
        (ScalarType::Unsigned, 32) => ops.push(Op::new(
            "cvt_f32",
            OpKind::OwnTrait,
            OpSig::Cvt {
                target_ty: ScalarType::Float,
                scalar_bits: 32,
            },
        )),
        (ScalarType::Int, 32) => ops.push(Op::new(
            "cvt_f32",
            OpKind::OwnTrait,
            OpSig::Cvt {
                target_ty: ScalarType::Float,
                scalar_bits: 32,
            },
        )),
        _ => (),
    }

    ops
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TyFlavor {
    /// Types for methods in the `Simd` trait; `f32x4<Self>`
    SimdTrait,
    /// Types for methods in the vec trait; `f32x4<S>`
    VecImpl,
}

impl OpSig {
    pub(crate) fn simd_trait_args(&self, vec_ty: &VecType) -> TokenStream {
        let ty = vec_ty.rust();
        match self {
            Self::Splat => {
                let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
                quote! { self, val: #scalar }
            }
            Self::LoadInterleaved {
                block_size,
                block_count,
            } => {
                let ty = load_interleaved_arg_ty(*block_size, *block_count, vec_ty);
                quote! { self, #ty }
            }
            Self::StoreInterleaved {
                block_size,
                block_count,
            } => {
                let ty = store_interleaved_arg_ty(*block_size, *block_count, vec_ty);
                quote! { self, #ty }
            }
            Self::Unary
            | Self::Split { .. }
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. } => quote! { self, a: #ty<Self> },
            Self::Binary
            | Self::Compare
            | Self::Combine { .. }
            | Self::Zip { .. }
            | Self::Unzip { .. } => {
                quote! { self, a: #ty<Self>, b: #ty<Self> }
            }
            Self::Shift => {
                quote! { self, a: #ty<Self>, shift: u32 }
            }
            Self::Ternary => {
                quote! { self, a: #ty<Self>, b: #ty<Self>, c: #ty<Self> }
            }
            Self::Select => {
                let mask_ty = vec_ty.mask_ty().rust();
                quote! { self, a: #mask_ty<Self>, b: #ty<Self>, c: #ty<Self> }
            }
        }
    }

    pub(crate) fn vec_trait_args(&self) -> Option<TokenStream> {
        let args = match self {
            Self::Splat | Self::LoadInterleaved { .. } | Self::StoreInterleaved { .. } => {
                return None;
            }
            Self::Unary
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. } => {
                quote! { self }
            }
            Self::Binary | Self::Compare | Self::Zip { .. } | Self::Unzip { .. } => {
                quote! { self, rhs: impl SimdInto<Self, S> }
            }
            Self::Shift => {
                quote! { self, shift: u32 }
            }
            Self::Ternary => {
                quote! { self, op1: impl SimdInto<Self, S>, op2: impl SimdInto<Self, S> }
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
            Self::Unary => quote! { self },
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

    pub(crate) fn ret_ty(&self, ty: &VecType, flavor: TyFlavor) -> TokenStream {
        let quant = match flavor {
            TyFlavor::SimdTrait => quote! { <Self> },
            TyFlavor::VecImpl => quote! { <S> },
        };
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
            Self::StoreInterleaved { .. } => quote! {()},
        }
    }
}

pub(crate) fn load_interleaved_arg_ty(
    block_size: u16,
    block_count: u16,
    vec_ty: &VecType,
) -> TokenStream {
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let len = (block_size * block_count) as usize / vec_ty.scalar_bits;
    quote! { src: &[#scalar; #len] }
}

pub(crate) fn store_interleaved_arg_ty(
    block_size: u16,
    block_count: u16,
    vec_ty: &VecType,
) -> TokenStream {
    let ty = vec_ty.rust();
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let len = (block_size * block_count) as usize / vec_ty.scalar_bits;
    quote! { a: #ty<Self>, dest: &mut [#scalar; #len] }
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
