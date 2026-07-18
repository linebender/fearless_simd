// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn sqrt_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 0.0, 1.0, 2.0]);
    assert_eq!(*f32x4::sqrt(a), [2.0, 0.0, 1.0, f32::sqrt(2.0)]);
}

#[simd_test]
fn sqrt_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[4.0, 9.0]);
    assert_eq!(*a.sqrt(), [2.0, 3.0]);
}

#[simd_test]
fn sqrt_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[4.0, 0.0, 1.0, 2.0, 9.0, 16.0, 25.0, 36.0]);
    assert_eq!(
        *f32x8::sqrt(a),
        [2.0, 0.0, 1.0, f32::sqrt(2.0), 3.0, 4.0, 5.0, 6.0]
    );
}

#[simd_test]
fn sqrt_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            4.0, 0.0, 1.0, 2.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0,
            169.0, 196.0,
        ],
    );
    assert_eq!(
        *f32x16::sqrt(a),
        [
            2.0,
            0.0,
            1.0,
            f32::sqrt(2.0),
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0
        ]
    );
}

#[simd_test]
fn sqrt_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[4.0, 0.0, 1.0, 2.0, 9.0, 16.0, 25.0, 36.0]);
    assert_eq!(
        *f64x8::sqrt(a),
        [2.0, 0.0, 1.0, f64::sqrt(2.0), 3.0, 4.0, 5.0, 6.0]
    );
}
