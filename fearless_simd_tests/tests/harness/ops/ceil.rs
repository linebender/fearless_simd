// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn ceil_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.1, -3.2, -2.7, 0.5]);
    assert_eq!(*a.ceil(), [3.0, -3.0, -2.0, 1.0]);
}

#[simd_test]
fn ceil_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[2.1, -3.2]);
    assert_eq!(*a.ceil(), [3.0, -3.0]);
    let b = f64x2::from_slice(simd, &[-2.7, 0.5]);
    assert_eq!(*b.ceil(), [-2.0, 1.0]);
}

#[simd_test]
fn ceil_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[2.1, -3.2, -2.7, 0.5, 1.1, -2.9, 3.0, -4.1]);
    assert_eq!(*a.ceil(), [3.0, -3.0, -2.0, 1.0, 2.0, -2.0, 3.0, -4.0]);
}

#[simd_test]
fn ceil_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            2.1, -3.2, -2.7, 0.5, 1.1, -2.9, 3.0, -4.1, 2.1, -3.2, -2.7, 0.5, 1.1, -2.9, 3.0, -4.1,
        ],
    );
    assert_eq!(
        *a.ceil(),
        [
            3.0, -3.0, -2.0, 1.0, 2.0, -2.0, 3.0, -4.0, 3.0, -3.0, -2.0, 1.0, 2.0, -2.0, 3.0, -4.0
        ]
    );
}

#[simd_test]
fn ceil_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[2.1, -3.2, -2.7, 0.5, 1.1, -2.9, 3.0, -4.1]);
    assert_eq!(*a.ceil(), [3.0, -3.0, -2.0, 1.0, 2.0, -2.0, 3.0, -4.0]);
}
