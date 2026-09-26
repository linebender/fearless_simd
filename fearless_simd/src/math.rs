// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SIMD math operators/kernels.

#![allow(
    dead_code,
    reason = "Generated backends use different subsets of these helpers"
)]

use crate::{f32x4, f64x2, i32x4, i64x2, prelude::*, u64x2};

#[cfg(all(feature = "libm", not(feature = "std")))]
pub(crate) trait FloatExt {
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round_ties_even(self) -> Self;
    fn fract(self) -> Self;
    fn sqrt(self) -> Self;
    fn trunc(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
}
#[cfg(all(feature = "libm", not(feature = "std")))]
impl FloatExt for f32 {
    #[inline(always)]
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        libm::ceilf(self)
    }
    #[inline(always)]
    fn round_ties_even(self) -> Self {
        libm::rintf(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline(always)]
    fn fract(self) -> Self {
        self - self.trunc()
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        libm::truncf(self)
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fmaf(self, a, b)
    }
}
#[cfg(all(feature = "libm", not(feature = "std")))]
impl FloatExt for f64 {
    #[inline(always)]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    #[inline(always)]
    fn round_ties_even(self) -> Self {
        libm::rint(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline(always)]
    fn fract(self) -> Self {
        self - self.trunc()
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        libm::trunc(self)
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fma(self, a, b)
    }
}

#[inline(always)]
pub(crate) fn mul_add_precise_f32x4<S: Simd>(
    simd: S,
    a: f32x4<S>,
    b: f32x4<S>,
    c: f32x4<S>,
) -> f32x4<S> {
    let (a_low, a_high) = a.widen();
    let (b_low, b_high) = b.widen();
    let (c_low, c_high) = c.widen();

    // Every finite f32 product is exactly representable as f64. Every exact finite
    // product-plus-add is a multiple of 2^-298 with magnitude below 2^256, so none of
    // these operations underflow or overflow in f64. Usually the f64 sum can be
    // narrowed directly. For a normal f32 result, the only possible double-rounding
    // error is when that sum is exactly halfway between two adjacent f32 values. This
    // also recognizes the f32 finite/infinity overflow threshold.
    let product_low = a_low * b_low;
    let product_high = a_high * b_high;
    let mut sum_low = product_low + c_low;
    let mut sum_high = product_high + c_high;

    const MIDPOINT_FRACTION_MASK: i64 = 0x1fff_ffff;
    const MIDPOINT_FRACTION: i64 = 0x1000_0000;
    let midpoint_low =
        (sum_low.bitcast::<i64x2<_>>() & MIDPOINT_FRACTION_MASK).simd_eq(MIDPOINT_FRACTION);
    let midpoint_high =
        (sum_high.bitcast::<i64x2<_>>() & MIDPOINT_FRACTION_MASK).simd_eq(MIDPOINT_FRACTION);
    // Subnormal f32 values have fewer than 24 significant bits, so their midpoint
    // position within the f64 significand varies with the result's exponent. Use a
    // provisional narrowing to identify every result whose rounding interval touches
    // the subnormal range, including zero and the smallest normal value. Those rare
    // lanes take the general round-to-odd path below.
    let mut provisional = sum_low.narrow(sum_high);
    let abs_provisional_bits = provisional.bitcast::<i32x4<_>>() & 0x7fff_ffff;
    let at_most_min_normal = abs_provisional_bits.simd_lt(0x0080_0001);
    let (subnormal_low, subnormal_high) = simd.widen_mask32x4(at_most_min_normal);
    let round_to_odd_low = midpoint_low | subnormal_low;
    let round_to_odd_high = midpoint_high | subnormal_high;
    let any_round_to_odd = (round_to_odd_low | round_to_odd_high).any_true();

    // Optimization notes:
    // On uniform numeric values over the entire range, this branch only fires once per 685k calls,
    // and on uniform numeric values in [-1, 1) it fires once in 17k calls (empirically).
    // So it is worth putting under an `if` despite the branch misprediction penalty.
    // Outlining this into a #[cold] function regresses performance on both fast and slow paths.
    // TODO: try using std::hint::cold_path() once MSRV is >= 1.95 and see if that does anything
    if any_round_to_odd {
        // Knuth's unconditional TwoSum establishes
        // `sum + error == product + c` exactly. If a candidate addition was
        // inexact and its rounded f64 significand is even, shift it by one ULP toward
        // the error. This produces the round-to-odd intermediate from Theorem 3.
        let virtual_low = sum_low - product_low;
        let error_low = (product_low - (sum_low - virtual_low)) + (c_low - virtual_low);
        let virtual_high = sum_high - product_high;
        let error_high = (product_high - (sum_high - virtual_high)) + (c_high - virtual_high);

        let sum_low_bits = sum_low.bitcast::<i64x2<_>>();
        let even_low = (sum_low_bits & 1).simd_eq(0);
        // Negate the sign bit and set the low bit: -1 if signs differ, +1 otherwise.
        let sign_low = ((sum_low_bits ^ error_low.bitcast::<i64x2<_>>()).bitcast::<u64x2<_>>()
            >> 63)
            .bitcast::<i64x2<_>>();
        let direction_low = (i64x2::<_>::splat(simd, 0) - sign_low) | 1;
        sum_low = (round_to_odd_low & even_low & !error_low.simd_eq(0.0)).select(
            (sum_low_bits + direction_low).bitcast::<f64x2<_>>(),
            sum_low,
        );

        let sum_high_bits = sum_high.bitcast::<i64x2<_>>();
        let even_high = (sum_high_bits & 1).simd_eq(0);
        let sign_high = ((sum_high_bits ^ error_high.bitcast::<i64x2<_>>()).bitcast::<u64x2<_>>()
            >> 63)
            .bitcast::<i64x2<_>>();
        let direction_high = (i64x2::<_>::splat(simd, 0) - sign_high) | 1;
        sum_high = (round_to_odd_high & even_high & !error_high.simd_eq(0.0)).select(
            (sum_high_bits + direction_high).bitcast::<f64x2<_>>(),
            sum_high,
        );

        provisional = sum_low.narrow(sum_high);
    }

    provisional
}

#[inline(always)]
pub(crate) fn mul_add_precise_f64x2<S: Simd>(
    simd: S,
    a: f64x2<S>,
    b: f64x2<S>,
    c: f64x2<S>,
) -> f64x2<S> {
    // Graillat and Muller's Algorithm 9 computes an FMA from a Dekker product
    // and their correctly rounded addition of a double-word and an FP number:
    // https://perso.lip6.fr/Stef.Graillat/papers/NM-2025.pdf
    // https://doi.org/10.1007/s00211-025-01487-2
    //
    // Their proof assumes an unbounded exponent range. Restrict the packed path
    // to a conservative range where no nonzero intermediate can underflow or
    // overflow, and use scalar FMA for complete IEEE binary64 range coverage.
    //
    // The vectorized path is taken for values between 2^-400 and 2^400,
    // values outside that range are routed to the scalar fallback.

    // 2^-400 and 2^400 have biased exponents 623 and 1423. Inclusive
    // comparisons are expressed using strict signed qword comparisons;
    // absolute-value binary64 bits are always nonnegative as i64 values.
    let lower = f64::from_bits(623_u64 << 52); // 2^-400
    let upper = f64::from_bits((1423_u64 << 52) + 1); // Next float above 2^400
    let a_abs = a.abs();
    let b_abs = b.abs();
    let c_abs = c.abs();
    let a_safe = a_abs.simd_ge(lower) & a_abs.simd_le(upper);
    let b_safe = b_abs.simd_ge(lower) & b_abs.simd_le(upper);
    let c_safe = (c_abs.simd_ge(lower) & c_abs.simd_le(upper)) | c_abs.simd_eq(0.0);
    // Scalarize both lanes if either one is outside the proven exponent-safe range.
    // This branch must precede the packed arithmetic so unsafe inactive lanes cannot
    // overflow or underflow inside the error-free transforms.
    if !(a_safe & b_safe & c_safe).all_true() {
        return [a[0].mul_add(b[0], c[0]), a[1].mul_add(b[1], c[1])].simd_into(simd);
    }

    // Split each normal multiplicand into nonoverlapping high and low parts.
    // Adding half of the discarded range before clearing 27 significand bits
    // gives the high part at most 26 significant bits and leaves the low part
    // at most 27. The addition operates on sign-magnitude binary64 encodings,
    // so it rounds the magnitude in the same direction for either sign. The
    // exponent guard above prevents the integer addition from wrapping.
    let split_rounding_bit = i64x2::<_>::splat(simd, 1 << 26);
    let split_high_mask = i64x2::<_>::splat(simd, !((1 << 27) - 1));
    let a_high =
        ((a.bitcast::<i64x2<_>>() + split_rounding_bit) & split_high_mask).bitcast::<f64x2<_>>();
    let a_low = a - a_high;
    let b_high =
        ((b.bitcast::<i64x2<_>>() + split_rounding_bit) & split_high_mask).bitcast::<f64x2<_>>();
    let b_low = b - b_high;

    // Dekker product: product_high + product_low is exactly a * b.
    let product_high = a * b;
    let product_error_1 = a_high * b_high - product_high;
    let product_error_2 = product_error_1 + a_high * b_low;
    let product_error_3 = product_error_2 + a_low * b_high;
    let product_low = product_error_3 + a_low * b_low;

    // Magnitude-sort each addition so Fast2Sum can replace TwoSum.
    // This trades packed compares and blends for a shorter floating-point
    // dependency chain, as suggested by Graillat and Muller.
    let sum_high = product_high + c;
    let product_high_larger = c_abs.simd_lt(product_high.abs());
    let sum_large = product_high_larger.select(product_high, c);
    let sum_small = product_high_larger.select(c, product_high);
    let sum_low = sum_small - (sum_high - sum_large);

    // Only the rounded sum of product_low and sum_low is needed on the
    // overwhelmingly common path. Its exact residual matters only when
    // v_high has one of the two significand shapes that can require the
    // Graillat-Muller correction, so defer that second Fast2Sum until then.
    let v_high = product_low + sum_low;
    let v_high_bits = v_high.bitcast::<u64x2<_>>();
    let special_fraction = (v_high_bits & 0x0007_ffff_ffff_ffff).simd_eq(0);

    // Zero has the same fraction shape and may enter this rare path harmlessly:
    // its exact residual is zero, so the correction mask stays false.
    if special_fraction.any_true() {
        // Fast2Sum(product_low, sum_low), after the same magnitude sort.
        let product_low_larger = sum_low.abs().simd_lt(product_low.abs());
        let v_large = product_low_larger.select(product_low, sum_low);
        let v_small = product_low_larger.select(sum_low, product_low);
        let v_low = v_small - (v_high - v_large);
        // The default sum is correctly rounded except when v_low is nonzero and
        // |v_high| is 2^k or 3 * 2^k. Under the exponent guard, every nonzero
        // product or residual is a multiple of at least 2^-904, so v_low != 0
        // implies that v_high is finite, normal, and nonzero. The two relevant
        // significand shapes therefore differ only in their top fraction bit.
        let special = special_fraction & !v_low.simd_eq(0.0);
        let different_sign = ((v_high_bits ^ v_low.bitcast::<u64x2<_>>()) >> 63).simd_eq(1);
        let factor = different_sign.select(
            f64x2::<_>::splat(simd, 7.0 / 8.0),
            f64x2::<_>::splat(simd, 9.0 / 8.0),
        );
        sum_high + special.select(factor * v_high, v_high)
    } else {
        sum_high + v_high
    }
}
