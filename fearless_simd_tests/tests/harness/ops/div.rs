// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

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

// Generated gap-fill coverage rows.

#[simd_test]
fn div_f64x2<S: Simd>(simd: S) {
    let a_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.0_f64);
    let b_values: [f64; 2] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let expected: [f64; 2] = core::array::from_fn(|i| a_values[i] / b_values[i]);
    let result = simd.div_f64x2(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn div_f64x4<S: Simd>(simd: S) {
    let a_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.0_f64);
    let b_values: [f64; 4] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x4::from_slice(simd, &a_values);
    let b = f64x4::from_slice(simd, &b_values);
    let expected: [f64; 4] = core::array::from_fn(|i| a_values[i] / b_values[i]);
    let result = simd.div_f64x4(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}
