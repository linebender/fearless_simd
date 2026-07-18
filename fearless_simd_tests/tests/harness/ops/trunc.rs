// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn trunc_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.9, -3.2, 0.0, 0.5]);
    assert_eq!(*a.trunc(), [2.0, -3.0, 0.0, 0.0]);
}

#[simd_test]
fn trunc_f32x4_special_values<S: Simd>(simd: S) {
    let a = f32x4::from_slice(
        simd,
        &[f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -f32::NAN],
    );
    let result = a.trunc();

    // Note: f32::NAN != f32::NAN hence we compare the bit pattern.
    assert_eq!(
        (*result).map(f32::to_bits),
        [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -f32::NAN].map(f32::to_bits)
    );
}

#[simd_test]
fn trunc_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.7, -2.3]);
    assert_eq!(*a.trunc(), [1.0, -2.0]);
}

#[simd_test]
fn trunc_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[2.9, -3.2, 0.0, 0.5, 1.8, -2.7, 3.1, -4.9]);
    assert_eq!(*a.trunc(), [2.0, -3.0, 0.0, 0.0, 1.0, -2.0, 3.0, -4.0]);
}

#[simd_test]
fn trunc_f32x8_special_values<S: Simd>(simd: S) {
    let a = f32x8::from_slice(
        simd,
        &[
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -f32::NAN,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -f32::NAN,
        ],
    );
    let result = a.trunc();

    // Note: f32::NAN != f32::NAN hence we compare the bit pattern.
    assert_eq!(
        (*result).map(f32::to_bits),
        [
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -f32::NAN,
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -f32::NAN
        ]
        .map(f32::to_bits)
    );
}

#[simd_test]
fn trunc_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            2.9, -3.2, 0.0, 0.5, 1.8, -2.7, 3.1, -4.9, 2.9, -3.2, 0.0, 0.5, 1.8, -2.7, 3.1, -4.9,
        ],
    );
    assert_eq!(
        *a.trunc(),
        [
            2.0, -3.0, 0.0, 0.0, 1.0, -2.0, 3.0, -4.0, 2.0, -3.0, 0.0, 0.0, 1.0, -2.0, 3.0, -4.0
        ]
    );
}

#[simd_test]
fn trunc_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[2.9, -3.2, 0.0, 0.5, 1.8, -2.7, 3.1, -4.9]);
    assert_eq!(*a.trunc(), [2.0, -3.0, 0.0, 0.0, 1.0, -2.0, 3.0, -4.0]);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn trunc_f64x4<S: Simd>(simd: S) {
    let values: [f64; 4] = core::array::from_fn(|i| i as f64 - 3.5_f64);
    let a = f64x4::from_slice(simd, &values);
    let expected: [f64; 4] = core::array::from_fn(|i| values[i].trunc());
    let result = simd.trunc_f64x4(a);
    assert_eq!(result.as_slice(), expected.as_slice());
}
