// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn fract_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.7, -2.3, 3.9, -4.1]);
    assert_eq!(
        *simd.fract_f32x4(a),
        [0.70000005, -0.29999995, 0.9000001, -0.099999905]
    );
}

#[simd_test]
fn fract_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.7, -2.3]);
    assert_eq!(*a.fract(), [0.7, -0.2999999999999998]);
}

#[simd_test]
fn fract_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.7, -2.3, 3.9, -4.1, 5.5, -6.6, 7.2, -8.8]);
    let result = simd.fract_f32x8(a);
    assert_eq!(
        *result,
        [
            0.70000005,
            -0.29999995,
            0.9000001,
            -0.099999905,
            0.5,
            -0.5999999,
            0.19999981, // 7.2 - 7.0 has precision differences
            -0.8000002
        ]
    );
}

#[simd_test]
fn fract_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.7, -2.3, 3.9, -4.1, 5.5, -6.6, 7.2, -8.8, 1.25, -2.75, 0.0, -0.5, 10.125, -10.875,
            100.0, -100.0,
        ],
    );
    let result = simd.fract_f32x16(a);
    assert_eq!(
        *result,
        [
            0.70000005,
            -0.29999995,
            0.9000001,
            -0.099999905,
            0.5,
            -0.5999999,
            0.19999981,
            -0.8000002,
            0.25,
            -0.75,
            0.0,
            -0.5,
            0.125,
            -0.875,
            0.0,
            0.0
        ]
    );
}

#[simd_test]
fn fract_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.7, -2.3, 3.9, -4.1, 5.5, -6.6, 7.2, -8.8]);
    let result = simd.fract_f64x8(a);
    assert_eq!(
        *result,
        [
            0.7,
            -0.2999999999999998,
            0.8999999999999999,
            -0.09999999999999964,
            0.5,
            -0.5999999999999996,
            0.20000000000000018,
            -0.8000000000000007
        ]
    );
}
