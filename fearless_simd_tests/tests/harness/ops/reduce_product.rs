// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// Floating-point products use a fixed balanced tree. The large and small
// values below make other plausible reduction orders produce infinity, zero,
// or NaN instead of the expected finite result.

#[simd_test]
fn reduce_product_f32x4<S: Simd>(simd: S) {
    assert_eq!(
        f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]).reduce_product(),
        24.0
    );

    let large = 1.0e30_f32;
    let small = 1.0e-30_f32;
    assert_eq!(
        f32x4::from_slice(simd, &[large, small, large, small]).reduce_product(),
        1.0
    );
    assert!(
        f32x4::from_slice(simd, &[large, large, small, small])
            .reduce_product()
            .is_nan()
    );

    assert_eq!(
        f32x4::from_slice(simd, &[-0.0, 1.0, 1.0, 1.0])
            .reduce_product()
            .to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        f32x4::from_slice(simd, &[-0.0, -1.0, 1.0, 1.0])
            .reduce_product()
            .to_bits(),
        0.0_f32.to_bits()
    );
    assert_eq!(
        f32x4::from_slice(simd, &[f32::INFINITY, 2.0, 1.0, 1.0]).reduce_product(),
        f32::INFINITY
    );
    assert_eq!(
        f32x4::from_slice(simd, &[f32::MIN_POSITIVE, 0.5, 1.0, 1.0]).reduce_product(),
        f32::MIN_POSITIVE * 0.5
    );

    let nan = f32::NAN;
    assert!(
        f32x4::from_slice(simd, &[1.0, 1.0, 1.0, nan])
            .reduce_product()
            .is_nan()
    );
}

#[simd_test]
fn reduce_product_f64x2<S: Simd>(simd: S) {
    assert_eq!(f64x2::from_slice(simd, &[2.0, 3.0]).reduce_product(), 6.0);

    let large = 1.0e300_f64;
    let small = 1.0e-300_f64;
    assert_eq!(
        f64x2::from_slice(simd, &[large, small]).reduce_product(),
        1.0
    );

    assert_eq!(
        f64x2::from_slice(simd, &[-0.0, 1.0])
            .reduce_product()
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        f64x2::from_slice(simd, &[-0.0, -1.0])
            .reduce_product()
            .to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        f64x2::from_slice(simd, &[f64::INFINITY, 2.0]).reduce_product(),
        f64::INFINITY
    );
    assert!(
        f64x2::from_slice(simd, &[f64::INFINITY, 0.0])
            .reduce_product()
            .is_nan()
    );
    assert_eq!(
        f64x2::from_slice(simd, &[f64::MIN_POSITIVE, 0.5]).reduce_product(),
        f64::MIN_POSITIVE * 0.5
    );

    let nan = f64::NAN;
    assert!(
        f64x2::from_slice(simd, &[1.0, nan])
            .reduce_product()
            .is_nan()
    );
}

#[simd_test]
fn reduce_product_f32x8<S: Simd>(simd: S) {
    assert_eq!(
        f32x8::from_slice(simd, &[2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reduce_product(),
        6.0
    );

    let large = 1.0e30_f32;
    let small = 1.0e-30_f32;
    assert_eq!(
        f32x8::from_slice(
            simd,
            &[large, large, large, large, small, small, small, small],
        )
        .reduce_product(),
        1.0
    );
    assert!(
        f32x8::from_slice(
            simd,
            &[large, large, small, small, large, large, small, small],
        )
        .reduce_product()
        .is_nan()
    );

    assert_eq!(
        f32x8::from_slice(simd, &[-0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .reduce_product()
            .to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        f32x8::from_slice(simd, &[-0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .reduce_product()
            .to_bits(),
        0.0_f32.to_bits()
    );
    assert_eq!(
        f32x8::from_slice(simd, &[f32::INFINITY, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],)
            .reduce_product(),
        f32::INFINITY
    );

    let nan = f32::NAN;
    assert!(
        f32x8::from_slice(simd, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan])
            .reduce_product()
            .is_nan()
    );
}

#[simd_test]
fn reduce_product_f64x4<S: Simd>(simd: S) {
    assert_eq!(
        f64x4::from_slice(simd, &[2.0, 3.0, 1.0, 1.0]).reduce_product(),
        6.0
    );

    let large = 1.0e300_f64;
    let small = 1.0e-300_f64;
    assert_eq!(
        f64x4::from_slice(simd, &[large, large, small, small]).reduce_product(),
        1.0
    );
    assert!(
        f64x4::from_slice(simd, &[large, small, large, small])
            .reduce_product()
            .is_nan()
    );

    assert_eq!(
        f64x4::from_slice(simd, &[-0.0, 1.0, 1.0, 1.0])
            .reduce_product()
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        f64x4::from_slice(simd, &[-0.0, -1.0, 1.0, 1.0])
            .reduce_product()
            .to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        f64x4::from_slice(simd, &[f64::INFINITY, 2.0, 1.0, 1.0]).reduce_product(),
        f64::INFINITY
    );

    let nan = f64::NAN;
    assert!(
        f64x4::from_slice(simd, &[1.0, 1.0, 1.0, nan])
            .reduce_product()
            .is_nan()
    );
}

#[simd_test]
fn reduce_product_f32x16<S: Simd>(simd: S) {
    assert_eq!(
        f32x16::from_slice(
            simd,
            &[
                2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )
        .reduce_product(),
        6.0
    );

    let large = 1.0e30_f32;
    let small = 1.0e-30_f32;
    assert_eq!(
        f32x16::from_slice(
            simd,
            &[
                large, large, large, large, large, large, large, large, small, small, small, small,
                small, small, small, small,
            ],
        )
        .reduce_product(),
        1.0
    );
    assert!(
        f32x16::from_slice(
            simd,
            &[
                large, large, small, small, large, large, small, small, large, large, small, small,
                large, large, small, small,
            ],
        )
        .reduce_product()
        .is_nan()
    );

    assert_eq!(
        f32x16::from_slice(
            simd,
            &[
                -0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )
        .reduce_product()
        .to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        f32x16::from_slice(
            simd,
            &[
                -0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )
        .reduce_product()
        .to_bits(),
        0.0_f32.to_bits()
    );

    let nan = f32::NAN;
    assert!(
        f32x16::from_slice(
            simd,
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan,
            ],
        )
        .reduce_product()
        .is_nan()
    );
}

#[simd_test]
fn reduce_product_f64x8<S: Simd>(simd: S) {
    assert_eq!(
        f64x8::from_slice(simd, &[2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reduce_product(),
        6.0
    );

    let large = 1.0e300_f64;
    let small = 1.0e-300_f64;
    assert_eq!(
        f64x8::from_slice(
            simd,
            &[large, large, large, large, small, small, small, small],
        )
        .reduce_product(),
        1.0
    );
    assert!(
        f64x8::from_slice(
            simd,
            &[large, large, small, small, large, large, small, small],
        )
        .reduce_product()
        .is_nan()
    );

    assert_eq!(
        f64x8::from_slice(simd, &[-0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .reduce_product()
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        f64x8::from_slice(simd, &[-0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .reduce_product()
            .to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        f64x8::from_slice(simd, &[f64::INFINITY, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],)
            .reduce_product(),
        f64::INFINITY
    );

    let nan = f64::NAN;
    assert!(
        f64x8::from_slice(simd, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan])
            .reduce_product()
            .is_nan()
    );
}

#[simd_test]
fn reduce_product_i8x16<S: Simd>(simd: S) {
    let mut ordinary = [1; 16];
    ordinary[0] = 2;
    ordinary[1] = 3;
    assert_eq!(i8x16::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut multiple_pairs = [1; 16];
    multiple_pairs[0] = 2;
    multiple_pairs[1] = 3;
    multiple_pairs[2] = 4;
    multiple_pairs[3] = 5;
    assert_eq!(
        i8x16::from_slice(simd, &multiple_pairs).reduce_product(),
        120
    );

    let mut both_halves = [1; 16];
    both_halves[0] = 2;
    both_halves[1] = 3;
    both_halves[8] = 4;
    both_halves[9] = 5;
    assert_eq!(i8x16::from_slice(simd, &both_halves).reduce_product(), 120);

    let mut reversed_high_bits = [1; 16];
    reversed_high_bits[8] = -1;
    reversed_high_bits[9] = i8::MIN;
    assert_eq!(
        i8x16::from_slice(simd, &reversed_high_bits).reduce_product(),
        i8::MIN
    );

    let mut negative = [1; 16];
    negative[0] = -2;
    negative[1] = 3;
    assert_eq!(i8x16::from_slice(simd, &negative).reduce_product(), -6);

    let mut even_negative = [1; 16];
    even_negative[0] = -2;
    even_negative[1] = -3;
    assert_eq!(i8x16::from_slice(simd, &even_negative).reduce_product(), 6);

    let mut zero = [1; 16];
    zero[7] = 0;
    assert_eq!(i8x16::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 16];
    overflow[0] = i8::MAX;
    overflow[1] = 2;
    assert_eq!(i8x16::from_slice(simd, &overflow).reduce_product(), -2);

    let mut underflow = [1; 16];
    underflow[0] = i8::MIN;
    underflow[1] = -1;
    assert_eq!(
        i8x16::from_slice(simd, &underflow).reduce_product(),
        i8::MIN
    );
}

#[simd_test]
fn reduce_product_u8x16<S: Simd>(simd: S) {
    let mut ordinary = [1; 16];
    ordinary[0] = 2;
    ordinary[1] = 3;
    assert_eq!(u8x16::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut multiple_pairs = [1; 16];
    multiple_pairs[0] = 2;
    multiple_pairs[1] = 3;
    multiple_pairs[2] = 4;
    multiple_pairs[3] = 5;
    assert_eq!(
        u8x16::from_slice(simd, &multiple_pairs).reduce_product(),
        120
    );

    let mut both_halves = [1; 16];
    both_halves[0] = 2;
    both_halves[1] = 3;
    both_halves[8] = 4;
    both_halves[9] = 5;
    assert_eq!(u8x16::from_slice(simd, &both_halves).reduce_product(), 120);

    let mut upper_high_bits = [1; 16];
    upper_high_bits[8] = u8::MAX;
    upper_high_bits[9] = 128;
    assert_eq!(
        u8x16::from_slice(simd, &upper_high_bits).reduce_product(),
        128
    );

    let mut zero = [1; 16];
    zero[7] = 0;
    assert_eq!(u8x16::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 16];
    overflow[0] = u8::MAX;
    overflow[1] = 2;
    assert_eq!(
        u8x16::from_slice(simd, &overflow).reduce_product(),
        u8::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i16x8<S: Simd>(simd: S) {
    let mut ordinary = [1; 8];
    ordinary[0] = 2;
    ordinary[1] = 3;
    assert_eq!(i16x8::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut both_halves = [1; 8];
    both_halves[0] = 2;
    both_halves[1] = 3;
    both_halves[4] = 4;
    both_halves[5] = 5;
    assert_eq!(i16x8::from_slice(simd, &both_halves).reduce_product(), 120);

    let mut negative = [1; 8];
    negative[0] = -2;
    negative[1] = 3;
    assert_eq!(i16x8::from_slice(simd, &negative).reduce_product(), -6);

    let mut even_negative = [1; 8];
    even_negative[0] = -2;
    even_negative[1] = -3;
    assert_eq!(i16x8::from_slice(simd, &even_negative).reduce_product(), 6);

    let mut zero = [1; 8];
    zero[3] = 0;
    assert_eq!(i16x8::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 8];
    overflow[0] = i16::MAX;
    overflow[1] = 2;
    assert_eq!(i16x8::from_slice(simd, &overflow).reduce_product(), -2);

    let mut underflow = [1; 8];
    underflow[0] = i16::MIN;
    underflow[1] = -1;
    assert_eq!(
        i16x8::from_slice(simd, &underflow).reduce_product(),
        i16::MIN
    );
}

#[simd_test]
fn reduce_product_u16x8<S: Simd>(simd: S) {
    let mut ordinary = [1; 8];
    ordinary[0] = 2;
    ordinary[1] = 3;
    assert_eq!(u16x8::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut both_halves = [1; 8];
    both_halves[0] = 2;
    both_halves[1] = 3;
    both_halves[4] = 4;
    both_halves[5] = 5;
    assert_eq!(u16x8::from_slice(simd, &both_halves).reduce_product(), 120);

    let mut zero = [1; 8];
    zero[3] = 0;
    assert_eq!(u16x8::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 8];
    overflow[0] = u16::MAX;
    overflow[1] = 2;
    assert_eq!(
        u16x8::from_slice(simd, &overflow).reduce_product(),
        u16::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i32x4<S: Simd>(simd: S) {
    let ordinary = [2, 3, 1, 1];
    assert_eq!(i32x4::from_slice(simd, &ordinary).reduce_product(), 6);

    let all_lanes = [2, 3, 5, 7];
    assert_eq!(i32x4::from_slice(simd, &all_lanes).reduce_product(), 210);

    let negative = [-2, 3, 1, 1];
    assert_eq!(i32x4::from_slice(simd, &negative).reduce_product(), -6);

    let even_negative = [-2, -3, 1, 1];
    assert_eq!(i32x4::from_slice(simd, &even_negative).reduce_product(), 6);

    let zero = [1, 1, 0, 1];
    assert_eq!(i32x4::from_slice(simd, &zero).reduce_product(), 0);

    let overflow = [i32::MAX, 2, 1, 1];
    assert_eq!(i32x4::from_slice(simd, &overflow).reduce_product(), -2);

    let underflow = [i32::MIN, -1, 1, 1];
    assert_eq!(
        i32x4::from_slice(simd, &underflow).reduce_product(),
        i32::MIN
    );
}

#[simd_test]
fn reduce_product_u32x4<S: Simd>(simd: S) {
    let ordinary = [2, 3, 1, 1];
    assert_eq!(u32x4::from_slice(simd, &ordinary).reduce_product(), 6);

    let all_lanes = [2, 3, 5, 7];
    assert_eq!(u32x4::from_slice(simd, &all_lanes).reduce_product(), 210);

    let zero = [1, 1, 0, 1];
    assert_eq!(u32x4::from_slice(simd, &zero).reduce_product(), 0);

    let overflow = [u32::MAX, 2, 1, 1];
    assert_eq!(
        u32x4::from_slice(simd, &overflow).reduce_product(),
        u32::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i64x2<S: Simd>(simd: S) {
    assert_eq!(i64x2::from_slice(simd, &[2, 3]).reduce_product(), 6);
    assert_eq!(i64x2::from_slice(simd, &[-2, 3]).reduce_product(), -6);
    assert_eq!(i64x2::from_slice(simd, &[-2, -3]).reduce_product(), 6);
    assert_eq!(i64x2::from_slice(simd, &[0, 3]).reduce_product(), 0);
    assert_eq!(i64x2::from_slice(simd, &[i64::MAX, 2]).reduce_product(), -2);
    assert_eq!(
        i64x2::from_slice(simd, &[i64::MIN, -1]).reduce_product(),
        i64::MIN
    );
}

#[simd_test]
fn reduce_product_u64x2<S: Simd>(simd: S) {
    assert_eq!(u64x2::from_slice(simd, &[2, 3]).reduce_product(), 6);
    assert_eq!(u64x2::from_slice(simd, &[0, 3]).reduce_product(), 0);
    assert_eq!(
        u64x2::from_slice(simd, &[u64::MAX, 2]).reduce_product(),
        u64::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i8x32<S: Simd>(simd: S) {
    let mut ordinary = [1; 32];
    ordinary[0] = 2;
    ordinary[16] = 3;
    assert_eq!(i8x32::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut negative = [1; 32];
    negative[0] = -2;
    negative[16] = 3;
    assert_eq!(i8x32::from_slice(simd, &negative).reduce_product(), -6);

    let mut even_negative = [1; 32];
    even_negative[0] = -2;
    even_negative[16] = -3;
    assert_eq!(i8x32::from_slice(simd, &even_negative).reduce_product(), 6);

    let mut zero = [1; 32];
    zero[15] = 0;
    assert_eq!(i8x32::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 32];
    overflow[0] = i8::MAX;
    overflow[16] = 2;
    assert_eq!(i8x32::from_slice(simd, &overflow).reduce_product(), -2);

    let mut underflow = [1; 32];
    underflow[0] = i8::MIN;
    underflow[16] = -1;
    assert_eq!(
        i8x32::from_slice(simd, &underflow).reduce_product(),
        i8::MIN
    );
}

#[simd_test]
fn reduce_product_u8x32<S: Simd>(simd: S) {
    let mut ordinary = [1; 32];
    ordinary[0] = 2;
    ordinary[16] = 3;
    assert_eq!(u8x32::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut zero = [1; 32];
    zero[15] = 0;
    assert_eq!(u8x32::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 32];
    overflow[0] = u8::MAX;
    overflow[16] = 2;
    assert_eq!(
        u8x32::from_slice(simd, &overflow).reduce_product(),
        u8::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i16x16<S: Simd>(simd: S) {
    let mut ordinary = [1; 16];
    ordinary[0] = 2;
    ordinary[8] = 3;
    assert_eq!(i16x16::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut negative = [1; 16];
    negative[0] = -2;
    negative[8] = 3;
    assert_eq!(i16x16::from_slice(simd, &negative).reduce_product(), -6);

    let mut even_negative = [1; 16];
    even_negative[0] = -2;
    even_negative[8] = -3;
    assert_eq!(i16x16::from_slice(simd, &even_negative).reduce_product(), 6);

    let mut zero = [1; 16];
    zero[7] = 0;
    assert_eq!(i16x16::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 16];
    overflow[0] = i16::MAX;
    overflow[8] = 2;
    assert_eq!(i16x16::from_slice(simd, &overflow).reduce_product(), -2);

    let mut underflow = [1; 16];
    underflow[0] = i16::MIN;
    underflow[8] = -1;
    assert_eq!(
        i16x16::from_slice(simd, &underflow).reduce_product(),
        i16::MIN
    );
}

#[simd_test]
fn reduce_product_u16x16<S: Simd>(simd: S) {
    let mut ordinary = [1; 16];
    ordinary[0] = 2;
    ordinary[8] = 3;
    assert_eq!(u16x16::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut zero = [1; 16];
    zero[7] = 0;
    assert_eq!(u16x16::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 16];
    overflow[0] = u16::MAX;
    overflow[8] = 2;
    assert_eq!(
        u16x16::from_slice(simd, &overflow).reduce_product(),
        u16::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i32x8<S: Simd>(simd: S) {
    let mut ordinary = [1; 8];
    ordinary[0] = 2;
    ordinary[4] = 3;
    assert_eq!(i32x8::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut negative = [1; 8];
    negative[0] = -2;
    negative[4] = 3;
    assert_eq!(i32x8::from_slice(simd, &negative).reduce_product(), -6);

    let mut even_negative = [1; 8];
    even_negative[0] = -2;
    even_negative[4] = -3;
    assert_eq!(i32x8::from_slice(simd, &even_negative).reduce_product(), 6);

    let mut zero = [1; 8];
    zero[3] = 0;
    assert_eq!(i32x8::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 8];
    overflow[0] = i32::MAX;
    overflow[4] = 2;
    assert_eq!(i32x8::from_slice(simd, &overflow).reduce_product(), -2);

    let mut underflow = [1; 8];
    underflow[0] = i32::MIN;
    underflow[4] = -1;
    assert_eq!(
        i32x8::from_slice(simd, &underflow).reduce_product(),
        i32::MIN
    );
}

#[simd_test]
fn reduce_product_u32x8<S: Simd>(simd: S) {
    let mut ordinary = [1; 8];
    ordinary[0] = 2;
    ordinary[4] = 3;
    assert_eq!(u32x8::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut zero = [1; 8];
    zero[3] = 0;
    assert_eq!(u32x8::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 8];
    overflow[0] = u32::MAX;
    overflow[4] = 2;
    assert_eq!(
        u32x8::from_slice(simd, &overflow).reduce_product(),
        u32::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i64x4<S: Simd>(simd: S) {
    let ordinary = [2, 1, 3, 1];
    assert_eq!(i64x4::from_slice(simd, &ordinary).reduce_product(), 6);

    let negative = [-2, 1, 3, 1];
    assert_eq!(i64x4::from_slice(simd, &negative).reduce_product(), -6);

    let even_negative = [-2, 1, -3, 1];
    assert_eq!(i64x4::from_slice(simd, &even_negative).reduce_product(), 6);

    let zero = [1, 0, 1, 1];
    assert_eq!(i64x4::from_slice(simd, &zero).reduce_product(), 0);

    let overflow = [i64::MAX, 1, 2, 1];
    assert_eq!(i64x4::from_slice(simd, &overflow).reduce_product(), -2);

    let underflow = [i64::MIN, 1, -1, 1];
    assert_eq!(
        i64x4::from_slice(simd, &underflow).reduce_product(),
        i64::MIN
    );
}

#[simd_test]
fn reduce_product_u64x4<S: Simd>(simd: S) {
    let ordinary = [2, 1, 3, 1];
    assert_eq!(u64x4::from_slice(simd, &ordinary).reduce_product(), 6);

    let zero = [1, 0, 1, 1];
    assert_eq!(u64x4::from_slice(simd, &zero).reduce_product(), 0);

    let overflow = [u64::MAX, 1, 2, 1];
    assert_eq!(
        u64x4::from_slice(simd, &overflow).reduce_product(),
        u64::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i8x64<S: Simd>(simd: S) {
    let mut ordinary = [1; 64];
    ordinary[0] = 2;
    ordinary[32] = 3;
    assert_eq!(i8x64::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut negative = [1; 64];
    negative[0] = -2;
    negative[32] = 3;
    assert_eq!(i8x64::from_slice(simd, &negative).reduce_product(), -6);

    let mut even_negative = [1; 64];
    even_negative[0] = -2;
    even_negative[32] = -3;
    assert_eq!(i8x64::from_slice(simd, &even_negative).reduce_product(), 6);

    let mut zero = [1; 64];
    zero[31] = 0;
    assert_eq!(i8x64::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 64];
    overflow[0] = i8::MAX;
    overflow[32] = 2;
    assert_eq!(i8x64::from_slice(simd, &overflow).reduce_product(), -2);

    let mut underflow = [1; 64];
    underflow[0] = i8::MIN;
    underflow[32] = -1;
    assert_eq!(
        i8x64::from_slice(simd, &underflow).reduce_product(),
        i8::MIN
    );
}

#[simd_test]
fn reduce_product_u8x64<S: Simd>(simd: S) {
    let mut ordinary = [1; 64];
    ordinary[0] = 2;
    ordinary[32] = 3;
    assert_eq!(u8x64::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut zero = [1; 64];
    zero[31] = 0;
    assert_eq!(u8x64::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 64];
    overflow[0] = u8::MAX;
    overflow[32] = 2;
    assert_eq!(
        u8x64::from_slice(simd, &overflow).reduce_product(),
        u8::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i16x32<S: Simd>(simd: S) {
    let mut ordinary = [1; 32];
    ordinary[0] = 2;
    ordinary[16] = 3;
    assert_eq!(i16x32::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut negative = [1; 32];
    negative[0] = -2;
    negative[16] = 3;
    assert_eq!(i16x32::from_slice(simd, &negative).reduce_product(), -6);

    let mut even_negative = [1; 32];
    even_negative[0] = -2;
    even_negative[16] = -3;
    assert_eq!(i16x32::from_slice(simd, &even_negative).reduce_product(), 6);

    let mut zero = [1; 32];
    zero[15] = 0;
    assert_eq!(i16x32::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 32];
    overflow[0] = i16::MAX;
    overflow[16] = 2;
    assert_eq!(i16x32::from_slice(simd, &overflow).reduce_product(), -2);

    let mut underflow = [1; 32];
    underflow[0] = i16::MIN;
    underflow[16] = -1;
    assert_eq!(
        i16x32::from_slice(simd, &underflow).reduce_product(),
        i16::MIN
    );
}

#[simd_test]
fn reduce_product_u16x32<S: Simd>(simd: S) {
    let mut ordinary = [1; 32];
    ordinary[0] = 2;
    ordinary[16] = 3;
    assert_eq!(u16x32::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut zero = [1; 32];
    zero[15] = 0;
    assert_eq!(u16x32::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 32];
    overflow[0] = u16::MAX;
    overflow[16] = 2;
    assert_eq!(
        u16x32::from_slice(simd, &overflow).reduce_product(),
        u16::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i32x16<S: Simd>(simd: S) {
    let mut ordinary = [1; 16];
    ordinary[0] = 2;
    ordinary[8] = 3;
    assert_eq!(i32x16::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut negative = [1; 16];
    negative[0] = -2;
    negative[8] = 3;
    assert_eq!(i32x16::from_slice(simd, &negative).reduce_product(), -6);

    let mut even_negative = [1; 16];
    even_negative[0] = -2;
    even_negative[8] = -3;
    assert_eq!(i32x16::from_slice(simd, &even_negative).reduce_product(), 6);

    let mut zero = [1; 16];
    zero[7] = 0;
    assert_eq!(i32x16::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 16];
    overflow[0] = i32::MAX;
    overflow[8] = 2;
    assert_eq!(i32x16::from_slice(simd, &overflow).reduce_product(), -2);

    let mut underflow = [1; 16];
    underflow[0] = i32::MIN;
    underflow[8] = -1;
    assert_eq!(
        i32x16::from_slice(simd, &underflow).reduce_product(),
        i32::MIN
    );
}

#[simd_test]
fn reduce_product_u32x16<S: Simd>(simd: S) {
    let mut ordinary = [1; 16];
    ordinary[0] = 2;
    ordinary[8] = 3;
    assert_eq!(u32x16::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut zero = [1; 16];
    zero[7] = 0;
    assert_eq!(u32x16::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 16];
    overflow[0] = u32::MAX;
    overflow[8] = 2;
    assert_eq!(
        u32x16::from_slice(simd, &overflow).reduce_product(),
        u32::MAX - 1
    );
}

#[simd_test]
fn reduce_product_i64x8<S: Simd>(simd: S) {
    let mut ordinary = [1; 8];
    ordinary[0] = 2;
    ordinary[4] = 3;
    assert_eq!(i64x8::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut negative = [1; 8];
    negative[0] = -2;
    negative[4] = 3;
    assert_eq!(i64x8::from_slice(simd, &negative).reduce_product(), -6);

    let mut even_negative = [1; 8];
    even_negative[0] = -2;
    even_negative[4] = -3;
    assert_eq!(i64x8::from_slice(simd, &even_negative).reduce_product(), 6);

    let mut zero = [1; 8];
    zero[3] = 0;
    assert_eq!(i64x8::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 8];
    overflow[0] = i64::MAX;
    overflow[4] = 2;
    assert_eq!(i64x8::from_slice(simd, &overflow).reduce_product(), -2);

    let mut underflow = [1; 8];
    underflow[0] = i64::MIN;
    underflow[4] = -1;
    assert_eq!(
        i64x8::from_slice(simd, &underflow).reduce_product(),
        i64::MIN
    );
}

#[simd_test]
fn reduce_product_u64x8<S: Simd>(simd: S) {
    let mut ordinary = [1; 8];
    ordinary[0] = 2;
    ordinary[4] = 3;
    assert_eq!(u64x8::from_slice(simd, &ordinary).reduce_product(), 6);

    let mut zero = [1; 8];
    zero[3] = 0;
    assert_eq!(u64x8::from_slice(simd, &zero).reduce_product(), 0);

    let mut overflow = [1; 8];
    overflow[0] = u64::MAX;
    overflow[4] = 2;
    assert_eq!(
        u64x8::from_slice(simd, &overflow).reduce_product(),
        u64::MAX - 1
    );
}
