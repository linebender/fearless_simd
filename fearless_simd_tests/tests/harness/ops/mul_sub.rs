// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

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

// Generated gap-fill coverage rows.

#[simd_test]
fn mul_sub_f64x4<S: Simd>(simd: S) {
    let a_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 4] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let c_values: [f64; 4] = core::array::from_fn(|i| (i % 3) as f64 + 0.5_f64);
    let a = f64x4::from_slice(simd, &a_values);
    let b = f64x4::from_slice(simd, &b_values);
    let c = f64x4::from_slice(simd, &c_values);
    let expected: [f64; 4] = core::array::from_fn(|i| a_values[i] * b_values[i] - c_values[i]);
    let result = simd.mul_sub_f64x4(a, b, c);
    assert_eq!(result.as_slice(), expected.as_slice());
}
