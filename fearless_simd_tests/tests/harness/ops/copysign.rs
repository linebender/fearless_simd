// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn copysign_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, -2.0, -3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[-1.0, 1.0, -1.0, 1.0]);
    assert_eq!(*a.copysign(b), [-1.0, 2.0, -3.0, 4.0]);
}

#[simd_test]
fn copysign_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.5, -2.5]);
    let b = f64x2::from_slice(simd, &[-1.0, 1.0]);
    assert_eq!(*a.copysign(b), [-1.5, 2.5]);
}

#[simd_test]
fn copysign_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, -2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0]);
    let b = f32x8::from_slice(simd, &[-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]);
    assert_eq!(*a.copysign(b), [-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, -7.0, 8.0]);
}

#[simd_test]
fn copysign_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, -2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0, 9.0, -10.0, -11.0, 12.0, -13.0, 14.0, 15.0,
            -16.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
        ],
    );
    assert_eq!(
        *a.copysign(b),
        [
            -1.0, 2.0, -3.0, 4.0, 5.0, -6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, 13.0, -14.0,
            -15.0, 16.0
        ]
    );
}

#[simd_test]
fn copysign_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, -2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0]);
    let b = f64x8::from_slice(simd, &[-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]);
    assert_eq!(*a.copysign(b), [-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, -7.0, 8.0]);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn copysign_f64x4<S: Simd>(simd: S) {
    let a_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.0_f64);
    let sign_values: [f64; 4] =
        core::array::from_fn(|i| if i % 2 == 0 { -1.0_f64 } else { 1.0_f64 });
    let a = f64x4::from_slice(simd, &a_values);
    let signs = f64x4::from_slice(simd, &sign_values);
    let expected: [f64; 4] = core::array::from_fn(|i| a_values[i].copysign(sign_values[i]));
    let result = simd.copysign_f64x4(a, signs);
    assert_eq!(result.as_slice(), expected.as_slice());
}
