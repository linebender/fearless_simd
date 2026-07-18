// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn min_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.0, -3.0, 0.0, 0.5]);
    let b = f32x4::from_slice(simd, &[1.0, -2.0, 7.0, 3.0]);
    assert_eq!(*a.min(b), [1.0, -3.0, 0.0, 0.5]);
}

#[simd_test]
fn min_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16
        ]
    );
}

#[simd_test]
fn min_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            15, 15, 35, 35, 45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            10, 15, 30, 35, 45, 60, 65, 80, 85, 100, 105, 120, 125, 140, 145, 160
        ]
    );
}

#[simd_test]
fn min_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i16x8::from_slice(simd, &[2, -1, 4, -3, 6, -5, 8, -7]);
    assert_eq!(*a.min(b), [1, -2, 3, -4, 5, -6, 7, -8]);
}

#[simd_test]
fn min_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    let b = u16x8::from_slice(simd, &[15, 15, 35, 35, 45, 65, 65, 85]);
    assert_eq!(*a.min(b), [10, 15, 30, 35, 45, 60, 65, 80]);
}

#[simd_test]
fn min_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, -2, 3, -4]);
    let b = i32x4::from_slice(simd, &[2, -1, 4, -3]);
    assert_eq!(*a.min(b), [1, -2, 3, -4]);
}

#[simd_test]
fn min_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[10, 20, 30, 40]);
    let b = u32x4::from_slice(simd, &[15, 15, 35, 35]);
    assert_eq!(*a.min(b), [10, 15, 30, 35]);
}

#[simd_test]
fn min_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0]);
    let b = f32x8::from_slice(simd, &[1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0]);
    assert_eq!(*a.min(b), [1.0, -3.0, 0.0, 0.5, 1.0, 4.0, 3.0, 5.0]);
}

#[simd_test]
fn min_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 1, -2, 3, -4, 5, -6, 7,
            -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15, 2, -1, 4, -3, 6, -5, 8,
            -7, 10, -9, 12, -11, 14, -13, 16, -15,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 1, -2, 3, -4, 5, -6, 7,
            -8, 9, -10, 11, -12, 13, -14, 15, -16
        ]
    );
}

#[simd_test]
fn min_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40,
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            15, 15, 35, 35, 45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165, 15, 15, 35, 35,
            45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            10, 15, 30, 35, 45, 60, 65, 80, 85, 100, 105, 120, 125, 140, 145, 160, 10, 15, 30, 35,
            45, 60, 65, 80, 85, 100, 105, 120, 125, 140, 145, 160
        ]
    );
}

#[simd_test]
fn min_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16
        ]
    );
}

#[simd_test]
fn min_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            15, 15, 35, 35, 45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            10, 15, 30, 35, 45, 60, 65, 80, 85, 100, 105, 120, 125, 140, 145, 160
        ]
    );
}

#[simd_test]
fn min_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i32x8::from_slice(simd, &[2, -1, 4, -3, 6, -5, 8, -7]);
    assert_eq!(*a.min(b), [1, -2, 3, -4, 5, -6, 7, -8]);
}

#[simd_test]
fn min_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    let b = u32x8::from_slice(simd, &[15, 15, 35, 35, 45, 65, 65, 85]);
    assert_eq!(*a.min(b), [10, 15, 30, 35, 45, 60, 65, 80]);
}

#[simd_test]
fn min_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0, 2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0, 1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            1.0, -3.0, 0.0, 0.5, 1.0, 4.0, 3.0, 5.0, 1.0, -3.0, 0.0, 0.5, 1.0, 4.0, 3.0, 5.0
        ]
    );
}

#[simd_test]
fn min_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 1, -2, 3, -4, 5, -6, 7,
            -8, 9, -10, 11, -12, 13, -14, 15, -16, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13,
            -14, 15, -16, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15, 2, -1, 4, -3, 6, -5, 8,
            -7, 10, -9, 12, -11, 14, -13, 16, -15, 2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14,
            -13, 16, -15, 2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 1, -2, 3, -4, 5, -6, 7,
            -8, 9, -10, 11, -12, 13, -14, 15, -16, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13,
            -14, 15, -16, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16
        ]
    );
}

#[simd_test]
fn min_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40,
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40, 50, 60, 70, 80,
            90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
            120, 130, 140, 150, 160,
        ],
    );
    let b = u8x64::from_slice(
        simd,
        &[
            15, 15, 35, 35, 45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165, 15, 15, 35, 35,
            45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165, 15, 15, 35, 35, 45, 65, 65, 85,
            85, 105, 105, 125, 125, 145, 145, 165, 15, 15, 35, 35, 45, 65, 65, 85, 85, 105, 105,
            125, 125, 145, 145, 165,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            10, 15, 30, 35, 45, 60, 65, 80, 85, 100, 105, 120, 125, 140, 145, 160, 10, 15, 30, 35,
            45, 60, 65, 80, 85, 100, 105, 120, 125, 140, 145, 160, 10, 15, 30, 35, 45, 60, 65, 80,
            85, 100, 105, 120, 125, 140, 145, 160, 10, 15, 30, 35, 45, 60, 65, 80, 85, 100, 105,
            120, 125, 140, 145, 160
        ]
    );
}

#[simd_test]
fn min_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 1, -2, 3, -4, 5, -6, 7,
            -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i16x32::from_slice(
        simd,
        &[
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15, 2, -1, 4, -3, 6, -5, 8,
            -7, 10, -9, 12, -11, 14, -13, 16, -15,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 1, -2, 3, -4, 5, -6, 7,
            -8, 9, -10, 11, -12, 13, -14, 15, -16
        ]
    );
}

#[simd_test]
fn min_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40,
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    let b = u16x32::from_slice(
        simd,
        &[
            15, 15, 35, 35, 45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165, 15, 15, 35, 35,
            45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            10, 15, 30, 35, 45, 60, 65, 80, 85, 100, 105, 120, 125, 140, 145, 160, 10, 15, 30, 35,
            45, 60, 65, 80, 85, 100, 105, 120, 125, 140, 145, 160
        ]
    );
}

#[simd_test]
fn min_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16
        ]
    );
}

#[simd_test]
fn min_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    let b = u32x16::from_slice(
        simd,
        &[
            15, 15, 35, 35, 45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            10, 15, 30, 35, 45, 60, 65, 80, 85, 100, 105, 120, 125, 140, 145, 160
        ]
    );
}

#[simd_test]
fn min_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0]);
    let b = f64x8::from_slice(simd, &[1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0]);
    assert_eq!(*a.min(b), [1.0, -3.0, 0.0, 0.5, 1.0, 4.0, 3.0, 5.0]);
}

// Additional concrete rows for this operation.

#[simd_test]
fn min_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[0_i64, 0_i64]);
    assert_eq!(*simd.min_i64x2(a, b), [0_i64, -2_i64]);
}

#[simd_test]
fn min_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[0_i64, 0_i64, 0_i64, 0_i64]);
    assert_eq!(*simd.min_i64x4(a, b), [0_i64, -2_i64, 0_i64, -4_i64]);
}

#[simd_test]
fn min_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64],
    );
    assert_eq!(
        *simd.min_i64x8(a, b),
        [0_i64, -2_i64, 0_i64, -4_i64, 0_i64, -6_i64, 0_i64, -8_i64]
    );
}

#[simd_test]
fn min_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[0_u64, 0_u64]);
    assert_eq!(*simd.min_u64x2(a, b), [0_u64, 0_u64]);
}

#[simd_test]
fn min_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[0_u64, 0_u64, 0_u64, 0_u64]);
    assert_eq!(*simd.min_u64x4(a, b), [0_u64, 0_u64, 0_u64, 0_u64]);
}

#[simd_test]
fn min_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64],
    );
    assert_eq!(
        *simd.min_u64x8(a, b),
        [0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn min_f64x2<S: Simd>(simd: S) {
    let a_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 2] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let expected: [f64; 2] = core::array::from_fn(|i| a_values[i].min(b_values[i]));
    let result = simd.min_f64x2(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn min_f64x4<S: Simd>(simd: S) {
    let a_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 4] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x4::from_slice(simd, &a_values);
    let b = f64x4::from_slice(simd, &b_values);
    let expected: [f64; 4] = core::array::from_fn(|i| a_values[i].min(b_values[i]));
    let result = simd.min_f64x4(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}
