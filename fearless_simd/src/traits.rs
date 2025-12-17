// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    missing_docs,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]
use crate::{Level, Simd, SimdBase};

/// Element-wise selection between two SIMD vectors using `self`.
pub trait Select<T> {
    /// For each element of this mask, select the first operand if the element is all ones, and select the second
    /// operand if the element is all zeroes.
    ///
    /// If a mask element is *not* all ones or all zeroes, the result is unspecified. It may vary depending on
    /// architecture, feature level, the mask elements' width, the mask vector's width, or library version.
    fn select(self, if_true: T, if_false: T) -> T;
}

// Same as pulp
pub trait WithSimd {
    type Output;

    fn with_simd<S: Simd>(self, simd: S) -> Self::Output;
}

impl<R, F: FnOnce(Level) -> R> WithSimd for F {
    type Output = R;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        self(simd.level())
    }
}

/// Conversion of SIMD types to and from raw bytes.
pub trait Bytes: Sized {
    type Bytes;

    /// Convert this type to an array of bytes.
    fn to_bytes(self) -> Self::Bytes;

    /// Create an instance of this type from an array of bytes.
    fn from_bytes(value: Self::Bytes) -> Self;

    /// Bitcast directly from this type to another one of the same size.
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

/// Value conversion, adding a SIMD blessing.
///
/// Analogous to [`From`], but takes a SIMD token, which is used to bless
/// the new value. Most such conversions are safe transmutes, but this
/// trait also supports splats, and implementations can use the SIMD token
/// to use an efficient splat intrinsic.
///
/// The [`SimdInto`] trait is also provided for convenience.
pub trait SimdFrom<T, S: Simd> {
    fn simd_from(value: T, simd: S) -> Self;
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
        SimdFrom::simd_from(self, simd)
    }
}

impl<T, S: Simd> SimdFrom<T, S> for T {
    fn simd_from(value: T, _simd: S) -> Self {
        value
    }
}

/// Types that can be used as elements in SIMD vectors.
pub trait SimdElement {
    /// The associated mask lane type. This will be a signed integer of the same size as this type.
    type Mask: SimdElement;
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

impl SimdElement for i64 {
    type Mask = Self;
}

/// Construction of integer vectors from floats by truncation
pub trait SimdCvtTruncate<T> {
    fn truncate_from(x: T) -> Self;
    fn truncate_from_precise(x: T) -> Self;
}

/// Construction of floating point vectors from integers
pub trait SimdCvtFloat<T> {
    fn float_from(x: T) -> Self;
}

/// Concatenation of two SIMD vectors.
///
/// This is implemented on all vectors 256 bits and lower, producing vectors of up to 512 bits.
pub trait SimdCombine<S: Simd>: SimdBase<S> {
    type Combined: SimdBase<S, Element = Self::Element, Block = Self::Block>;

    /// Concatenate two vectors into a new one that's twice as long.
    fn combine(self, rhs: impl SimdInto<Self, S>) -> Self::Combined;
}

/// Splitting of one SIMD vector into two.
///
/// This is implemented on all vectors 256 bits and higher, producing vectors of down to 128 bits.
pub trait SimdSplit<S: Simd>: SimdBase<S> {
    type Split: SimdBase<S, Element = Self::Element, Block = Self::Block>;

    /// Split this vector into left and right halves.
    fn split(self) -> (Self::Split, Self::Split);
}
