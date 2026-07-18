// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn cvt_i32_f32x4_rounding<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[0.0, 0.49, 0.51, 0.99]);
    assert_eq!(*a.to_int::<i32x4<_>>(), [0, 0, 0, 0]);
    let a = f32x4::from_slice(simd, &[-0.0, -0.49, -0.51, -0.99]);
    assert_eq!(*a.to_int::<i32x4<_>>(), [0, 0, 0, 0]);
    let a = f32x4::from_slice(simd, &[1.01, 1.99, 2.5, 3.75]);
    assert_eq!(*a.to_int::<i32x4<_>>(), [1, 1, 2, 3]);
    let a = f32x4::from_slice(simd, &[-1.01, -1.99, -2.5, -3.75]);
    assert_eq!(*a.to_int::<i32x4<_>>(), [-1, -1, -2, -3]);
}

#[simd_test]
fn cvt_i32_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, -0.9, 13.34, 234234.8]);

    assert_eq!(*a.to_int::<i32x4<_>>(), [-10, 0, 13, 234234]);
}

#[simd_test]
fn cvt_i32_f32x8_rounding<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[0.0, 0.49, 0.51, 0.99, 1.01, 1.99, 2.5, 3.75]);
    assert_eq!(*a.to_int::<i32x8<_>>(), [0, 0, 0, 0, 1, 1, 2, 3]);
    let a = f32x8::from_slice(
        simd,
        &[-0.0, -0.49, -0.51, -0.99, -1.01, -1.99, -2.5, -3.75],
    );
    assert_eq!(*a.to_int::<i32x8<_>>(), [0, 0, 0, 0, -1, -1, -2, -3]);
}

#[simd_test]
fn cvt_i32_f32x16_rounding<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            0.0, 0.49, 0.51, 0.99, 1.01, 1.99, 2.5, 3.75, 0.0, 0.49, 0.51, 0.99, 1.01, 1.99, 2.5,
            3.75,
        ],
    );
    assert_eq!(
        *a.to_int::<i32x16<_>>(),
        [0, 0, 0, 0, 1, 1, 2, 3, 0, 0, 0, 0, 1, 1, 2, 3]
    );
    let a = f32x16::from_slice(
        simd,
        &[
            -0.0, -0.49, -0.51, -0.99, -1.01, -1.99, -2.5, -3.75, -0.0, -0.49, -0.51, -0.99, -1.01,
            -1.99, -2.5, -3.75,
        ],
    );
    assert_eq!(
        *a.to_int::<i32x16<_>>(),
        [0, 0, 0, 0, -1, -1, -2, -3, 0, 0, 0, 0, -1, -1, -2, -3]
    );
}

#[simd_test]
fn cvt_i32_f32x16<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtTruncate;
    let a = f32x16::from_slice(
        simd,
        &[
            1.7, -2.3, 3.9, -4.1, 5.5, -6.6, 7.2, -8.8, 10.0, -11.5, 12.9, -13.1, 14.0, -15.0, 0.0,
            100.5,
        ],
    );
    let result = i32x16::truncate_from(a);
    assert_eq!(
        *result,
        [
            1, -2, 3, -4, 5, -6, 7, -8, 10, -11, 12, -13, 14, -15, 0, 100
        ]
    );
}
