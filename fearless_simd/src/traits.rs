// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    missing_docs,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]
use crate::{Simd, SimdBase, seal::Seal};
use core::fmt::{Binary, Debug, Display, LowerExp, UpperExp};
use core::iter::{Product, Sum};
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use core::str::FromStr;

/// Element-wise selection between two SIMD vectors using `self`.
pub trait Select<T: Seal>: Seal {
    /// For each logical lane of this mask, select the first operand if the lane is true, and select the second
    /// operand if the lane is false.
    ///
    /// Masks may be converted to and from signed integer lane arrays for compatibility with older APIs. For those
    /// conversions, false is encoded as all zeroes (integer value 0) and true is encoded as all ones (integer value -1).
    /// If a mask is constructed from any other integer bit pattern, the result of this operation is unspecified.
    fn select(self, if_true: T, if_false: T) -> T;
}

/// Conversion of SIMD vectors to and from same-width vectors of `u8` lanes.
///
/// [`Bytes::bitcast`] uses this byte representation to reinterpret any two
/// non-mask SIMD vectors with the same total width and SIMD token. This is a
/// bitwise reinterpretation: it does not perform numeric conversion.
pub trait Bytes: Sized + Seal {
    /// The same-width SIMD vector of `u8` lanes used as the byte representation.
    ///
    /// This type is its own byte representation.
    type Bytes: Bytes<Bytes = Self::Bytes>;

    /// Reinterpret this vector as a same-width vector of `u8` lanes.
    fn to_bytes(self) -> Self::Bytes;

    /// Reinterpret a same-width vector of `u8` lanes as this vector type.
    fn from_bytes(value: Self::Bytes) -> Self;

    #[doc(alias = "reinterpret")]
    #[doc(alias = "transmute")]
    /// Bitcast directly to another SIMD vector with the same byte representation.
    /// This is effectively a safe [transmute](core::mem::transmute) for SIMD types.
    ///
    /// This works in code generic over a [`Simd`] implementation,
    /// including between native-width vectors with different lane types:
    ///
    /// ```
    /// # use fearless_simd::prelude::*;
    /// fn i8s_as_f64s<S: Simd>(value: S::i8s) -> S::f64s {
    ///     value.bitcast()
    /// }
    /// ```
    #[inline(always)]
    fn bitcast<U: Bytes<Bytes = Self::Bytes>>(self) -> U {
        U::from_bytes(self.to_bytes())
    }
}

pub(crate) mod seal {
    #[expect(
        unnameable_types,
        reason = "This is a sealed trait, so being unnameable is the entire point"
    )]
    pub trait Seal {}
}

impl Seal for f32 {}
impl Seal for f64 {}
impl Seal for u8 {}
impl Seal for i8 {}
impl Seal for u16 {}
impl Seal for i16 {}
impl Seal for u32 {}
impl Seal for i32 {}
impl Seal for u64 {}
impl Seal for i64 {}

/// Value conversion, adding a SIMD blessing.
///
/// Analogous to [`From`], but takes a SIMD token, which is used to bless
/// the new value. Most such conversions are safe transmutes, but this
/// trait also supports splats, and implementations can use the SIMD token
/// to use an efficient splat intrinsic.
///
/// The [`SimdInto`] trait is also provided for convenience.
pub trait SimdFrom<T, S: Simd> {
    fn simd_from(simd: S, value: T) -> Self;
}

/// Value conversion, adding a SIMD blessing.
///
/// This trait is syntactic sugar for [`SimdFrom`] and exists only to allow
/// `impl SimdInto` syntax in signatures, which would otherwise require
/// cumbersome `where` clauses in terms of `SimdFrom`.
///
/// Avoid implementing this trait directly, prefer implementing [`SimdFrom`].
pub trait SimdInto<T, S> {
    fn simd_into(self, simd: S) -> T;
}

impl<F, T: SimdFrom<F, S>, S: Simd> SimdInto<T, S> for F {
    fn simd_into(self, simd: S) -> T {
        SimdFrom::simd_from(simd, self)
    }
}

impl<T, S: Simd> SimdFrom<T, S> for T {
    fn simd_from(_simd: S, value: T) -> Self {
        value
    }
}

/// Types that can be used as elements in SIMD vectors.
pub trait SimdElement:
    Copy
    + Clone
    + Seal
    + Default
    + Debug
    + Display
    + FromStr
    + LowerExp
    + UpperExp
    + PartialOrd
    + PartialEq
    + From<bool>
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<Self, Output = Self>
    + DivAssign<Self>
    + Rem<Self, Output = Self>
    + RemAssign<Self>
    + Sum<Self>
    + Product<Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> DivAssign<&'a Self>
    + for<'a> Rem<&'a Self, Output = Self>
    + for<'a> RemAssign<&'a Self>
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
{
    /// The associated mask lane type. This will be a signed integer of the same size as this type.
    type Mask: SimdElement<Mask = Self::Mask>;

    /// The size of an element in bits.
    const BITS: usize = size_of::<Self>() * u8::BITS as usize;
}

impl SimdElement for f32 {
    type Mask = i32;
}

impl SimdElement for f64 {
    type Mask = i64;
}

impl SimdElement for u8 {
    type Mask = i8;
}

impl SimdElement for i8 {
    type Mask = Self;
}

impl SimdElement for u16 {
    type Mask = i16;
}

impl SimdElement for i16 {
    type Mask = Self;
}

impl SimdElement for u32 {
    type Mask = i32;
}

impl SimdElement for i32 {
    type Mask = Self;
}

impl SimdElement for u64 {
    type Mask = i64;
}

impl SimdElement for i64 {
    type Mask = Self;
}

/// Types that can be used as elements in integer SIMD vectors.
pub trait SimdIntElement:
    SimdElement
    + Eq
    + Ord
    + Binary
    + Not<Output = Self>
    + Shl<usize, Output = Self>
    + ShlAssign<usize>
    + Shr<usize, Output = Self>
    + ShrAssign<usize>
    + BitAnd<Self, Output = Self>
    + BitAndAssign<Self>
    + BitOr<Self, Output = Self>
    + BitOrAssign<Self>
    + BitXor<Self, Output = Self>
    + BitXorAssign<Self>
    + for<'a> Shl<&'a usize, Output = Self>
    + for<'a> ShlAssign<&'a usize>
    + for<'a> Shr<&'a usize, Output = Self>
    + for<'a> ShrAssign<&'a usize>
    + for<'a> BitAnd<&'a Self, Output = Self>
    + for<'a> BitAndAssign<&'a Self>
    + for<'a> BitOr<&'a Self, Output = Self>
    + for<'a> BitOrAssign<&'a Self>
    + for<'a> BitXor<&'a Self, Output = Self>
    + for<'a> BitXorAssign<&'a Self>
{
}

impl SimdIntElement for u8 {}
impl SimdIntElement for u16 {}
impl SimdIntElement for u32 {}
impl SimdIntElement for u64 {}
impl SimdIntElement for i8 {}
impl SimdIntElement for i16 {}
impl SimdIntElement for i32 {}
impl SimdIntElement for i64 {}

/// Types that can be used as elements in float SIMD vectors.
///
/// The scalar conversion bounds are limited to types that every floating-point
/// element can represent losslessly, including f16 for forward-compatibility.
pub trait SimdFloatElement: SimdElement + Neg<Output = Self> + From<i8> + From<u8> {}

impl SimdFloatElement for f32 {}
impl SimdFloatElement for f64 {}

/// Construction of integer vectors from floats by truncation
pub trait SimdCvtTruncate<T: Seal>: Seal {
    fn truncate_from(x: T) -> Self;
    fn truncate_from_precise(x: T) -> Self;
}

/// Construction of floating point vectors from integers
pub trait SimdCvtFloat<T: Seal>: Seal {
    fn float_from(x: T) -> Self;
}

/// Concatenation of two SIMD vectors.
///
/// This is implemented on all vectors 256 bits and lower, producing vectors of up to 512 bits.
pub trait SimdCombine<S: Simd>: SimdBase<S> + Seal {
    type Combined: SimdBase<S, Element = Self::Element, Block = Self::Block>
        + SimdSplit<S, Split = Self>;

    /// Concatenate two vectors into a new one that's twice as long.
    fn combine(self, rhs: impl SimdInto<Self, S>) -> Self::Combined;
}

/// Splitting of one SIMD vector into two.
///
/// This is implemented on all vectors 256 bits and higher, producing vectors of down to 128 bits.
pub trait SimdSplit<S: Simd>: SimdBase<S> + Seal {
    type Split: SimdBase<S, Element = Self::Element, Block = Self::Block>
        + SimdCombine<S, Combined = Self>;

    /// Split this vector into left and right halves.
    fn split(self) -> (Self::Split, Self::Split);
}
