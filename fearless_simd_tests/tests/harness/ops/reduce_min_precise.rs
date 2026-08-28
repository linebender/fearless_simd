// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn reduce_min_precise_f32x4<S: Simd>(simd: S) {
    let nan = f32::from_bits(0x7fc0_1234);
    assert_eq!(
        f32x4::from_slice(simd, &[nan, 4.0, -2.0, 7.0]).reduce_min_precise(),
        -2.0
    );
    assert_eq!(
        f32x4::from_slice(simd, &[4.0, -2.0, 7.0, nan]).reduce_min_precise(),
        -2.0
    );
    assert_eq!(
        f32x4::from_slice(simd, &[nan, 4.0, nan, -2.0]).reduce_min_precise(),
        -2.0
    );
    assert!(
        f32x4::from_slice(simd, &[nan; 4])
            .reduce_min_precise()
            .is_nan()
    );
    assert_eq!(
        f32x4::from_slice(simd, &[f32::INFINITY, nan, 3.0, f32::NEG_INFINITY]).reduce_min_precise(),
        f32::NEG_INFINITY
    );
    assert_eq!(
        f32x4::from_slice(simd, &[0.0, -0.0, 0.0, -0.0]).reduce_min_precise(),
        0.0
    );
}

#[simd_test]
fn reduce_min_precise_f64x2<S: Simd>(simd: S) {
    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    assert_eq!(
        f64x2::from_slice(simd, &[nan, -2.0]).reduce_min_precise(),
        -2.0
    );
    assert_eq!(
        f64x2::from_slice(simd, &[-2.0, nan]).reduce_min_precise(),
        -2.0
    );
    assert!(
        f64x2::from_slice(simd, &[nan; 2])
            .reduce_min_precise()
            .is_nan()
    );
    assert_eq!(
        f64x2::from_slice(simd, &[f64::INFINITY, f64::NEG_INFINITY]).reduce_min_precise(),
        f64::NEG_INFINITY
    );
    assert_eq!(
        f64x2::from_slice(simd, &[0.0, -0.0]).reduce_min_precise(),
        0.0
    );
}

#[simd_test]
fn reduce_min_precise_f32x8<S: Simd>(simd: S) {
    let nan = f32::from_bits(0x7fc0_1234);
    assert_eq!(
        f32x8::from_slice(simd, &[nan, 4.0, -2.0, 7.0, 9.0, 3.0, 6.0, 8.0]).reduce_min_precise(),
        -2.0
    );
    assert_eq!(
        f32x8::from_slice(simd, &[4.0, -2.0, 7.0, 9.0, 3.0, 6.0, 8.0, nan]).reduce_min_precise(),
        -2.0
    );
    assert_eq!(
        f32x8::from_slice(simd, &[nan, 4.0, nan, 7.0, -2.0, nan, 6.0, 8.0]).reduce_min_precise(),
        -2.0
    );
    assert!(
        f32x8::from_slice(simd, &[nan; 8])
            .reduce_min_precise()
            .is_nan()
    );
    assert_eq!(
        f32x8::from_slice(
            simd,
            &[
                f32::INFINITY,
                nan,
                3.0,
                8.0,
                2.0,
                f32::NEG_INFINITY,
                5.0,
                7.0
            ],
        )
        .reduce_min_precise(),
        f32::NEG_INFINITY
    );
    let zeros = f32x8::from_fn(simd, |i| if i % 2 == 0 { 0.0 } else { -0.0 });
    assert_eq!(zeros.reduce_min_precise(), 0.0);
}

#[simd_test]
fn reduce_min_precise_f64x4<S: Simd>(simd: S) {
    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    assert_eq!(
        f64x4::from_slice(simd, &[nan, 4.0, -2.0, 7.0]).reduce_min_precise(),
        -2.0
    );
    assert_eq!(
        f64x4::from_slice(simd, &[4.0, -2.0, 7.0, nan]).reduce_min_precise(),
        -2.0
    );
    assert_eq!(
        f64x4::from_slice(simd, &[nan, 4.0, nan, -2.0]).reduce_min_precise(),
        -2.0
    );
    assert!(
        f64x4::from_slice(simd, &[nan; 4])
            .reduce_min_precise()
            .is_nan()
    );
    assert_eq!(
        f64x4::from_slice(simd, &[f64::INFINITY, nan, 3.0, f64::NEG_INFINITY]).reduce_min_precise(),
        f64::NEG_INFINITY
    );
    assert_eq!(
        f64x4::from_slice(simd, &[0.0, -0.0, 0.0, -0.0]).reduce_min_precise(),
        0.0
    );
}

#[simd_test]
fn reduce_min_precise_f32x16<S: Simd>(simd: S) {
    let nan = f32::from_bits(0x7fc0_1234);
    let mut first = [4.0; 16];
    first[0] = nan;
    first[13] = -2.0;
    assert_eq!(f32x16::from_slice(simd, &first).reduce_min_precise(), -2.0);

    let mut last = [4.0; 16];
    last[15] = nan;
    last[2] = -2.0;
    assert_eq!(f32x16::from_slice(simd, &last).reduce_min_precise(), -2.0);

    let multiple = f32x16::from_fn(simd, |i| if i % 3 == 0 { nan } else { i as f32 });
    assert_eq!(multiple.reduce_min_precise(), 1.0);
    assert!(
        f32x16::from_slice(simd, &[nan; 16])
            .reduce_min_precise()
            .is_nan()
    );

    let infinity = f32x16::from_fn(simd, |i| {
        if i == 11 {
            f32::NEG_INFINITY
        } else if i == 3 {
            f32::INFINITY
        } else {
            i as f32
        }
    });
    assert_eq!(infinity.reduce_min_precise(), f32::NEG_INFINITY);
    let zeros = f32x16::from_fn(simd, |i| if i % 2 == 0 { 0.0 } else { -0.0 });
    assert_eq!(zeros.reduce_min_precise(), 0.0);
}

#[simd_test]
fn reduce_min_precise_f64x8<S: Simd>(simd: S) {
    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    assert_eq!(
        f64x8::from_slice(simd, &[nan, 4.0, -2.0, 7.0, 9.0, 3.0, 6.0, 8.0]).reduce_min_precise(),
        -2.0
    );
    assert_eq!(
        f64x8::from_slice(simd, &[4.0, -2.0, 7.0, 9.0, 3.0, 6.0, 8.0, nan]).reduce_min_precise(),
        -2.0
    );
    assert_eq!(
        f64x8::from_slice(simd, &[nan, 4.0, nan, 7.0, -2.0, nan, 6.0, 8.0]).reduce_min_precise(),
        -2.0
    );
    assert!(
        f64x8::from_slice(simd, &[nan; 8])
            .reduce_min_precise()
            .is_nan()
    );
    assert_eq!(
        f64x8::from_slice(
            simd,
            &[
                f64::INFINITY,
                nan,
                3.0,
                8.0,
                2.0,
                f64::NEG_INFINITY,
                5.0,
                7.0
            ],
        )
        .reduce_min_precise(),
        f64::NEG_INFINITY
    );
    let zeros = f64x8::from_fn(simd, |i| if i % 2 == 0 { 0.0 } else { -0.0 });
    assert_eq!(zeros.reduce_min_precise(), 0.0);
}

#[simd_test]
fn reduce_min_precise_i8x16_alias<S: Simd>(simd: S) {
    let value = i8x16::from_fn(simd, |i| if i == 13 { i8::MIN } else { i8::MAX });
    assert_eq!(value.reduce_min_precise(), i8::MIN);
}

#[simd_test]
fn reduce_min_precise_u64x8_alias<S: Simd>(simd: S) {
    let value = u64x8::from_fn(simd, |i| if i == 5 { u64::MIN } else { u64::MAX });
    assert_eq!(value.reduce_min_precise(), u64::MIN);
}
