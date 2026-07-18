// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn mul_sub_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.0, 3.0, 4.0, 5.0]);
    let b = f32x4::from_slice(simd, &[10.0, 10.0, 10.0, 10.0]);
    let c = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(*a.mul_sub(b, c), [19.0, 28.0, 37.0, 46.0]);
}

#[simd_test]
fn mul_sub_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[2.0, 3.0]);
    let b = f64x2::from_slice(simd, &[4.0, 5.0]);
    let c = f64x2::from_slice(simd, &[1.0, 2.0]);
    assert_eq!(*a.mul_sub(b, c), [7.0, 13.0]);
}

#[simd_test]
fn mul_sub_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let b = f32x8::from_slice(simd, &[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
    let c = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(
        *a.mul_sub(b, c),
        [19.0, 28.0, 37.0, 46.0, 55.0, 64.0, 73.0, 82.0]
    );
}

#[simd_test]
fn mul_sub_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0, 10.0,
        ],
    );
    let c = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ],
    );
    assert_eq!(
        *a.mul_sub(b, c),
        [
            19.0, 28.0, 37.0, 46.0, 55.0, 64.0, 73.0, 82.0, 19.0, 28.0, 37.0, 46.0, 55.0, 64.0,
            73.0, 82.0
        ]
    );
}

#[simd_test]
fn mul_sub_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let b = f64x8::from_slice(simd, &[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
    let c = f64x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(
        *a.mul_sub(b, c),
        [19.0, 28.0, 37.0, 46.0, 55.0, 64.0, 73.0, 82.0]
    );
}
