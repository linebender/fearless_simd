// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn max_precise_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.0, -3.0, 0.0, 0.5]);
    let b = f32x4::from_slice(simd, &[1.0, -2.0, 7.0, 3.0]);
    assert_eq!(*a.max_precise(b), [2.0, -2.0, 7.0, 3.0]);
}

#[simd_test]
fn max_precise_f32x4_with_nan<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[f32::NAN, -3.0, f32::INFINITY, 0.5]);
    let b = f32x4::from_slice(simd, &[1.0, f32::NAN, 7.0, f32::NEG_INFINITY]);
    let result = a.max_precise(b);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -3.0);
    assert_eq!(result[2], f32::INFINITY);
    assert_eq!(result[3], 0.5);
}

#[simd_test]
fn max_precise_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[2.0, -3.0]);
    let b = f64x2::from_slice(simd, &[1.0, -2.0]);
    assert_eq!(*a.max_precise(b), [2.0, -2.0]);
}

#[simd_test]
fn max_precise_f64x2_with_nan<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[f64::NAN, -3.0]);
    let b = f64x2::from_slice(simd, &[1.0, f64::NAN]);
    let result = a.max_precise(b);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -3.0);
}

#[simd_test]
fn max_precise_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0]);
    let b = f32x8::from_slice(simd, &[1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0]);
    assert_eq!(*a.max_precise(b), [2.0, -2.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0]);
}

#[simd_test]
fn max_precise_f32x8_with_nan<S: Simd>(simd: S) {
    let a = f32x8::from_slice(
        simd,
        &[f32::NAN, -3.0, f32::INFINITY, 0.5, 1.0, f32::NAN, 3.0, 7.0],
    );
    let b = f32x8::from_slice(
        simd,
        &[
            1.0,
            f32::NAN,
            7.0,
            f32::NEG_INFINITY,
            f32::NAN,
            4.0,
            6.0,
            5.0,
        ],
    );
    let result = a.max_precise(b);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -3.0);
    assert_eq!(result[2], f32::INFINITY);
    assert_eq!(result[3], 0.5);
    assert_eq!(result[4], 1.0);
    assert_eq!(result[5], 4.0);
    assert_eq!(result[6], 6.0);
    assert_eq!(result[7], 7.0);
}

#[simd_test]
fn max_precise_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[2.0, -3.0, 0.0, 0.5]);
    let b = f64x4::from_slice(simd, &[1.0, -2.0, 7.0, 3.0]);
    assert_eq!(*a.max_precise(b), [2.0, -2.0, 7.0, 3.0]);
}

#[simd_test]
fn max_precise_f64x4_with_nan<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[f64::NAN, -3.0, f64::INFINITY, 0.5]);
    let b = f64x4::from_slice(simd, &[1.0, f64::NAN, 7.0, f64::NEG_INFINITY]);
    let result = a.max_precise(b);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -3.0);
    assert_eq!(result[2], f64::INFINITY);
    assert_eq!(result[3], 0.5);
}

#[simd_test]
fn max_precise_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0, 2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0, 1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0,
        ],
    );
    assert_eq!(
        *a.max_precise(b),
        [
            2.0, -2.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0, 2.0, -2.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0
        ]
    );
}

#[simd_test]
fn max_precise_f32x16_with_nan<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            f32::NAN,
            -3.0,
            f32::INFINITY,
            0.5,
            1.0,
            f32::NAN,
            3.0,
            7.0,
            f32::NAN,
            -3.0,
            f32::INFINITY,
            0.5,
            1.0,
            f32::NAN,
            3.0,
            7.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0,
            f32::NAN,
            7.0,
            f32::NEG_INFINITY,
            f32::NAN,
            4.0,
            6.0,
            5.0,
            1.0,
            f32::NAN,
            7.0,
            f32::NEG_INFINITY,
            f32::NAN,
            4.0,
            6.0,
            5.0,
        ],
    );
    let result = a.max_precise(b);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -3.0);
    assert_eq!(result[2], f32::INFINITY);
    assert_eq!(result[3], 0.5);
    assert_eq!(result[4], 1.0);
    assert_eq!(result[5], 4.0);
    assert_eq!(result[6], 6.0);
    assert_eq!(result[7], 7.0);
    assert_eq!(result[8], 1.0);
    assert_eq!(result[9], -3.0);
    assert_eq!(result[10], f32::INFINITY);
    assert_eq!(result[11], 0.5);
    assert_eq!(result[12], 1.0);
    assert_eq!(result[13], 4.0);
    assert_eq!(result[14], 6.0);
    assert_eq!(result[15], 7.0);
}

#[simd_test]
fn max_precise_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0]);
    let b = f64x8::from_slice(simd, &[1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0]);
    assert_eq!(*a.max_precise(b), [2.0, -2.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0]);
}

#[simd_test]
fn max_precise_f64x8_with_nan<S: Simd>(simd: S) {
    let a = f64x8::from_slice(
        simd,
        &[f64::NAN, -3.0, f64::INFINITY, 0.5, 1.0, f64::NAN, 3.0, 7.0],
    );
    let b = f64x8::from_slice(
        simd,
        &[
            1.0,
            f64::NAN,
            7.0,
            f64::NEG_INFINITY,
            f64::NAN,
            4.0,
            6.0,
            5.0,
        ],
    );
    let result = a.max_precise(b);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -3.0);
    assert_eq!(result[2], f64::INFINITY);
    assert_eq!(result[3], 0.5);
    assert_eq!(result[4], 1.0);
    assert_eq!(result[5], 4.0);
    assert_eq!(result[6], 6.0);
    assert_eq!(result[7], 7.0);
}
