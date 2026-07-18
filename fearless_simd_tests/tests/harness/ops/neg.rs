// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn neg_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, -2.0, 3.0, -4.0]);
    assert_eq!(*(-a), [-1.0, 2.0, -3.0, 4.0]);
}

#[simd_test]
fn neg_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    assert_eq!(
        *(-a),
        [
            -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16
        ]
    );
}

#[simd_test]
fn neg_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    assert_eq!(*(-a), [-1, 2, -3, 4, -5, 6, -7, 8]);
}

#[simd_test]
fn neg_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.5, -2.5]);
    assert_eq!(*(-a), [-1.5, 2.5]);
}

#[simd_test]
fn neg_i32x4<S: Simd>(simd: S) {
    assert_eq!(*(-i32x4::from_slice(simd, &[1, -2, 3, -4])), [-1, 2, -3, 4]);
}

#[simd_test]
fn neg_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
    assert_eq!(*(-a), [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]);
}

#[simd_test]
fn neg_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    assert_eq!(
        *(-a),
        [
            -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21,
            22, -23, 24, -25, 26, -27, 28, -29, 30, -31, 32
        ]
    );
}

#[simd_test]
fn neg_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    assert_eq!(
        *(-a),
        [
            -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16
        ]
    );
}

#[simd_test]
fn neg_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    assert_eq!(*(-a), [-1, 2, -3, 4, -5, 6, -7, 8]);
}

#[simd_test]
fn neg_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0,
            -16.0,
        ],
    );
    assert_eq!(
        *(-a),
        [
            -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0,
            -15.0, 16.0
        ]
    );
}

#[simd_test]
fn neg_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10,
            11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29,
            -30, 31, -32,
        ],
    );
    assert_eq!(
        *(-a),
        [
            -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21,
            22, -23, 24, -25, 26, -27, 28, -29, 30, -31, 32, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10,
            -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23, 24, -25, 26, -27, 28, -29,
            30, -31, 32
        ]
    );
}

#[simd_test]
fn neg_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 1, -2, 3, -4, 5, -6, 7,
            -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    assert_eq!(
        *(-a),
        [
            -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -1, 2, -3, 4, -5, 6, -7,
            8, -9, 10, -11, 12, -13, 14, -15, 16
        ]
    );
}

#[simd_test]
fn neg_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    assert_eq!(
        *(-a),
        [
            -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16
        ]
    );
}

#[simd_test]
fn neg_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
    assert_eq!(*(-a), [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]);
}

#[simd_test]
fn neg_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[-1, 2, -3, 4, -5, 6, -7, 8]);
    assert_eq!(*(-a), [1, -2, 3, -4, 5, -6, 7, -8]);
}

#[simd_test]
fn neg_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[-1, 2]);
    assert_eq!(*(-a), [1, -2]);
}

#[simd_test]
fn neg_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[-1, 2, -3, 4]);
    assert_eq!(*(-a), [1, -2, 3, -4]);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn neg_f64x4<S: Simd>(simd: S) {
    let values: [f64; 4] = core::array::from_fn(|i| i as f64 - 3.5_f64);
    let a = f64x4::from_slice(simd, &values);
    let expected: [f64; 4] = core::array::from_fn(|i| -values[i]);
    let result = simd.neg_f64x4(a);
    assert_eq!(result.as_slice(), expected.as_slice());
}
