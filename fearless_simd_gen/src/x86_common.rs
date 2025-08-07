use crate::arch::Arch;
use crate::arch::sse4_2::Sse4_2;
use crate::generic::{generic_combine, generic_split};
use crate::ops::{OpSig, TyFlavor, reinterpret_ty, valid_reinterpret};
use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

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

pub(crate) fn set1_intrinsic(ty: ScalarType, bits: usize, ty_bits: usize) -> Ident {
    use ScalarType::*;
    let suffix = match (ty, bits) {
        (Int | Unsigned | Mask, 64) => "epi64x",
        _ => op_suffix(ty, bits, false),
    };

    intrinsic_ident("set1", suffix, ty_bits)
}

pub(crate) fn simple_intrinsic(name: &str, ty: ScalarType, bits: usize, ty_bits: usize) -> Ident {
    let suffix = op_suffix(ty, bits, true);

    intrinsic_ident(name, suffix, ty_bits)
}

pub(crate) fn simple_sign_unaware_intrinsic(
    name: &str,
    ty: ScalarType,
    bits: usize,
    ty_bits: usize,
) -> Ident {
    let suffix = op_suffix(ty, bits, false);

    intrinsic_ident(name, suffix, ty_bits)
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

pub(crate) fn cvt_intrinsic(from: VecType, to: VecType) -> Ident {
    let from_suffix = op_suffix(from.scalar, from.scalar_bits, false);
    let to_suffix = op_suffix(to.scalar, to.scalar_bits, false);

    intrinsic_ident(&format!("cvt{from_suffix}"), to_suffix, from.n_bits())
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
