// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn div_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 2.0, 1.0, 0.0]);
    let b = f32x4::from_slice(simd, &[4.0, 1.0, 3.0, 0.1]);
    assert_eq!(*(a / b), [1.0, 2.0, 1.0 / 3.0, 0.0]);
}

#[simd_test]
fn div_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[4.0, 2.0, 1.0, 0.0, 10.0, 12.0, 15.0, 20.0]);
    let b = f32x8::from_slice(simd, &[4.0, 1.0, 3.0, 0.1, 2.0, 3.0, 5.0, 4.0]);
    assert_eq!(*(a / b), [1.0, 2.0, 1.0 / 3.0, 0.0, 5.0, 4.0, 3.0, 5.0]);
}

#[simd_test]
fn div_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            4.0, 2.0, 1.0, 0.0, 10.0, 12.0, 15.0, 20.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            4.0, 1.0, 3.0, 0.1, 2.0, 3.0, 5.0, 4.0, 2.0, 3.0, 2.0, 1.0, 4.0, 1.0, 2.0, 3.0,
        ],
    );
    assert_eq!(
        *(a / b),
        [
            1.0,
            2.0,
            1.0 / 3.0,
            0.0,
            5.0,
            4.0,
            3.0,
            5.0,
            4.0,
            3.0,
            5.0,
            11.0,
            3.0,
            13.0,
            7.0,
            5.0
        ]
    );
}

#[simd_test]
fn div_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[4.0, 2.0, 1.0, 0.0, 10.0, 12.0, 15.0, 20.0]);
    let b = f64x8::from_slice(simd, &[4.0, 1.0, 3.0, 0.1, 2.0, 3.0, 5.0, 4.0]);
    assert_eq!(*(a / b), [1.0, 2.0, 1.0 / 3.0, 0.0, 5.0, 4.0, 3.0, 5.0]);
}
