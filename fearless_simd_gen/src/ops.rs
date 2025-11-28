// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::TokenStream;
use quote::quote;

use crate::types::{ScalarType, VecType};

#[derive(Clone, Copy)]
pub(crate) enum OpSig {
    Splat,
    Unary,
    Binary,
    Ternary,
    Compare,
    Select,
    Combine,
    Split,
    Zip {
        select_low: bool,
    },
    Unzip {
        select_even: bool,
    },
    Cvt {
        target_ty: ScalarType,
        scalar_bits: usize,
    },
    Reinterpret {
        target_ty: ScalarType,
        scalar_bits: usize,
    },
    WidenNarrow {
        target_ty: VecType,
    },
    // TODO: Make clear that this is right-shift
    Shift,
    // First argument is the base block size (i.e. 128), second argument
    // is how many blocks. For example, `LoadInterleaved(128, 4)` would correspond to the
    // NEON instructions `vld4q_f32`, while `LoadInterleaved(64, 4)` would correspond to
    // `vld4_f32`.
    LoadInterleaved {
        block_size: u16,
        block_count: u16,
    },
    StoreInterleaved {
        block_size: u16,
        block_count: u16,
    },
}

pub(crate) const FLOAT_OPS: &[(&str, OpSig)] = &[
    ("splat", OpSig::Splat),
    ("abs", OpSig::Unary),
    ("neg", OpSig::Unary),
    ("sqrt", OpSig::Unary),
    ("add", OpSig::Binary),
    ("sub", OpSig::Binary),
    ("mul", OpSig::Binary),
    ("div", OpSig::Binary),
    ("copysign", OpSig::Binary),
    ("simd_eq", OpSig::Compare),
    ("simd_lt", OpSig::Compare),
    ("simd_le", OpSig::Compare),
    ("simd_ge", OpSig::Compare),
    ("simd_gt", OpSig::Compare),
    ("zip_low", OpSig::Zip { select_low: true }),
    ("zip_high", OpSig::Zip { select_low: false }),
    ("unzip_low", OpSig::Unzip { select_even: true }),
    ("unzip_high", OpSig::Unzip { select_even: false }),
    // The non-precise max/min are *allowed*, but not required, to return NaN if either operand is NaN.
    //
    // TODO: document the behavior of max/min vs max_precise/min_precise once we generate documentation.
    ("max", OpSig::Binary),
    ("min", OpSig::Binary),
    // The precise max/min are guaranteed to return a non-NaN result if at most one operand is non-NaN.
    ("max_precise", OpSig::Binary),
    ("min_precise", OpSig::Binary),
    ("madd", OpSig::Ternary),
    ("msub", OpSig::Ternary),
    ("floor", OpSig::Unary),
    ("ceil", OpSig::Unary),
    ("round_ties_even", OpSig::Unary),
    ("fract", OpSig::Unary),
    ("trunc", OpSig::Unary),
    // TODO: simd_ne, but this requires additional implementation work on Neon
    ("select", OpSig::Select),
];

pub(crate) const INT_OPS: &[(&str, OpSig)] = &[
    ("splat", OpSig::Splat),
    ("not", OpSig::Unary),
    ("add", OpSig::Binary),
    ("sub", OpSig::Binary),
    ("mul", OpSig::Binary),
    ("and", OpSig::Binary),
    ("or", OpSig::Binary),
    ("xor", OpSig::Binary),
    ("shr", OpSig::Shift),
    // Shift right by vector
    ("shrv", OpSig::Binary),
    ("shl", OpSig::Shift),
    ("simd_eq", OpSig::Compare),
    ("simd_lt", OpSig::Compare),
    ("simd_le", OpSig::Compare),
    ("simd_ge", OpSig::Compare),
    ("simd_gt", OpSig::Compare),
    ("zip_low", OpSig::Zip { select_low: true }),
    ("zip_high", OpSig::Zip { select_low: false }),
    ("unzip_low", OpSig::Unzip { select_even: true }),
    ("unzip_high", OpSig::Unzip { select_even: false }),
    ("select", OpSig::Select),
    ("min", OpSig::Binary),
    ("max", OpSig::Binary),
];

pub(crate) const MASK_OPS: &[(&str, OpSig)] = &[
    ("splat", OpSig::Splat),
    ("not", OpSig::Unary),
    ("and", OpSig::Binary),
    ("or", OpSig::Binary),
    ("xor", OpSig::Binary),
    ("select", OpSig::Select),
    ("simd_eq", OpSig::Compare),
];

/// Ops covered by `core::ops`
pub(crate) const CORE_OPS: &[&str] = &[
    "not", "neg", "add", "sub", "mul", "div", "and", "or", "xor", "shr", "shrv", "shl",
];

pub(crate) fn ops_for_type(ty: &VecType, cvt: bool) -> Vec<(&str, OpSig)> {
    let base = match ty.scalar {
        ScalarType::Float => FLOAT_OPS,
        ScalarType::Int | ScalarType::Unsigned => INT_OPS,
        ScalarType::Mask => MASK_OPS,
    };
    let mut ops = base.to_vec();
    if ty.n_bits() < 512 {
        ops.push(("combine", OpSig::Combine));
    }
    if ty.n_bits() > 128 {
        ops.push(("split", OpSig::Split));
    }
    if ty.scalar == ScalarType::Int {
        ops.push(("neg", OpSig::Unary));
    }

    if ty.scalar == ScalarType::Float {
        if cvt {
            if ty.scalar_bits == 64 {
                ops.push((
                    "reinterpret_f32",
                    OpSig::Reinterpret {
                        target_ty: ScalarType::Float,
                        scalar_bits: 32,
                    },
                ));
            } else {
                ops.push((
                    "reinterpret_f64",
                    OpSig::Reinterpret {
                        target_ty: ScalarType::Float,
                        scalar_bits: 64,
                    },
                ));

                ops.push((
                    "reinterpret_i32",
                    OpSig::Reinterpret {
                        target_ty: ScalarType::Int,
                        scalar_bits: 32,
                    },
                ));
            }
        }

        if ty.scalar_bits == 64 {
            return ops;
        }
    }

    if matches!(ty.scalar, ScalarType::Unsigned | ScalarType::Float) && ty.n_bits() == 512 {
        ops.push((
            "load_interleaved_128",
            OpSig::LoadInterleaved {
                block_size: 128,
                block_count: 4,
            },
        ));
    }

    if matches!(ty.scalar, ScalarType::Unsigned | ScalarType::Float) && ty.n_bits() == 512 {
        ops.push((
            "store_interleaved_128",
            OpSig::StoreInterleaved {
                block_size: 128,
                block_count: 4,
            },
        ));
    }

    if cvt {
        if matches!(ty.scalar, ScalarType::Unsigned) {
            if let Some(target_ty) = ty.widened() {
                ops.push(("widen", OpSig::WidenNarrow { target_ty }));
            }

            if let Some(target_ty) = ty.narrowed() {
                ops.push(("narrow", OpSig::WidenNarrow { target_ty }));
            }
        }

        if valid_reinterpret(ty, ScalarType::Unsigned, 8) {
            ops.push((
                "reinterpret_u8",
                OpSig::Reinterpret {
                    target_ty: ScalarType::Unsigned,
                    scalar_bits: 8,
                },
            ));
        }

        if valid_reinterpret(ty, ScalarType::Unsigned, 32) {
            ops.push((
                "reinterpret_u32",
                OpSig::Reinterpret {
                    target_ty: ScalarType::Unsigned,
                    scalar_bits: 32,
                },
            ));
        }

        match (ty.scalar, ty.scalar_bits) {
            (ScalarType::Float, 32) => {
                ops.push((
                    "cvt_u32",
                    OpSig::Cvt {
                        target_ty: ScalarType::Unsigned,
                        scalar_bits: 32,
                    },
                ));
                ops.push((
                    "cvt_i32",
                    OpSig::Cvt {
                        target_ty: ScalarType::Int,
                        scalar_bits: 32,
                    },
                ));
            }
            (ScalarType::Unsigned, 32) => ops.push((
                "cvt_f32",
                OpSig::Cvt {
                    target_ty: ScalarType::Float,
                    scalar_bits: 32,
                },
            )),
            (ScalarType::Int, 32) => ops.push((
                "cvt_f32",
                OpSig::Cvt {
                    target_ty: ScalarType::Float,
                    scalar_bits: 32,
                },
            )),
            _ => (),
        }
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
            | Self::Split
            | Self::Cvt { .. }
            | Self::Reinterpret { .. }
            | Self::WidenNarrow { .. } => quote! { self, a: #ty<Self> },
            Self::Binary
            | Self::Compare
            | Self::Combine
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
            Self::Binary
            | Self::Compare
            | Self::Zip { .. }
            | Self::Combine
            | Self::Unzip { .. } => {
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
            Self::Split => return None,
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
            Self::Combine => {
                let n2 = ty.len * 2;
                let result = VecType::new(ty.scalar, ty.scalar_bits, n2).rust();
                quote! { #result #quant }
            }
            Self::Split => {
                let len = ty.len / 2;
                let result = VecType::new(ty.scalar, ty.scalar_bits, len).rust();
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
                let result = reinterpret_ty(ty, *target_ty, *scalar_bits).rust();
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

pub(crate) fn reinterpret_ty(src: &VecType, dst_scalar: ScalarType, dst_bits: usize) -> VecType {
    VecType::new(dst_scalar, dst_bits, src.n_bits() / dst_bits)
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
