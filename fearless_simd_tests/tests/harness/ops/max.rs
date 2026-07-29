// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn max_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.0, -3.0, 0.0, 0.5]);
    let b = f32x4::from_slice(simd, &[1.0, -2.0, 7.0, 3.0]);
    assert_eq!(*a.max(b), [2.0, -2.0, 7.0, 3.0]);
}

#[simd_test]
fn max_i8x16<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15
        ]
    );
}

#[simd_test]
fn max_u8x16<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            15, 20, 35, 40, 50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 165
        ]
    );
}

#[simd_test]
fn max_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i16x8::from_slice(simd, &[2, -1, 4, -3, 6, -5, 8, -7]);
    assert_eq!(*a.max(b), [2, -1, 4, -3, 6, -5, 8, -7]);
}

#[simd_test]
fn max_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    let b = u16x8::from_slice(simd, &[15, 15, 35, 35, 45, 65, 65, 85]);
    assert_eq!(*a.max(b), [15, 20, 35, 40, 50, 65, 70, 85]);
}

#[simd_test]
fn max_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, -2, 3, -4]);
    let b = i32x4::from_slice(simd, &[2, -1, 4, -3]);
    assert_eq!(*a.max(b), [2, -1, 4, -3]);
}

#[simd_test]
fn max_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[10, 20, 30, 40]);
    let b = u32x4::from_slice(simd, &[15, 15, 35, 35]);
    assert_eq!(*a.max(b), [15, 20, 35, 40]);
}

#[simd_test]
fn max_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0]);
    let b = f32x8::from_slice(simd, &[1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0]);
    assert_eq!(*a.max(b), [2.0, -2.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0]);
}

#[simd_test]
fn max_i8x32<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15, 2, -1, 4, -3, 6, -5, 8,
            -7, 10, -9, 12, -11, 14, -13, 16, -15
        ]
    );
}

#[simd_test]
fn max_u8x32<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            15, 20, 35, 40, 50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 165, 15, 20, 35, 40,
            50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 165
        ]
    );
}

#[simd_test]
fn max_i16x16<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15
        ]
    );
}

#[simd_test]
fn max_u16x16<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            15, 20, 35, 40, 50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 165
        ]
    );
}

#[simd_test]
fn max_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i32x8::from_slice(simd, &[2, -1, 4, -3, 6, -5, 8, -7]);
    assert_eq!(*a.max(b), [2, -1, 4, -3, 6, -5, 8, -7]);
}

#[simd_test]
fn max_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    let b = u32x8::from_slice(simd, &[15, 15, 35, 35, 45, 65, 65, 85]);
    assert_eq!(*a.max(b), [15, 20, 35, 40, 50, 65, 70, 85]);
}

#[simd_test]
fn max_f32x16<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            2.0, -2.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0, 2.0, -2.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0
        ]
    );
}

#[simd_test]
fn max_i8x64<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15, 2, -1, 4, -3, 6, -5, 8,
            -7, 10, -9, 12, -11, 14, -13, 16, -15, 2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14,
            -13, 16, -15, 2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15
        ]
    );
}

#[simd_test]
fn max_u8x64<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            15, 20, 35, 40, 50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 165, 15, 20, 35, 40,
            50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 165, 15, 20, 35, 40, 50, 65, 70, 85,
            90, 105, 110, 125, 130, 145, 150, 165, 15, 20, 35, 40, 50, 65, 70, 85, 90, 105, 110,
            125, 130, 145, 150, 165
        ]
    );
}

#[simd_test]
fn max_i16x32<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15, 2, -1, 4, -3, 6, -5, 8,
            -7, 10, -9, 12, -11, 14, -13, 16, -15
        ]
    );
}

#[simd_test]
fn max_u16x32<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            15, 20, 35, 40, 50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 165, 15, 20, 35, 40,
            50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 165
        ]
    );
}

#[simd_test]
fn max_i32x16<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15
        ]
    );
}

#[simd_test]
fn max_u32x16<S: Simd>(simd: S) {
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
        *a.max(b),
        [
            15, 20, 35, 40, 50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 165
        ]
    );
}

#[simd_test]
fn max_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0]);
    let b = f64x8::from_slice(simd, &[1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0]);
    assert_eq!(*a.max(b), [2.0, -2.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0]);
}

// Additional concrete rows for this operation.

#[simd_test]
fn max_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[0_i64, 0_i64]);
    assert_eq!(*simd.max_i64x2(a, b), [1_i64, 0_i64]);
}

#[simd_test]
fn max_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[0_i64, 0_i64, 0_i64, 0_i64]);
    assert_eq!(*simd.max_i64x4(a, b), [1_i64, 0_i64, 3_i64, 0_i64]);
}

#[simd_test]
fn max_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64],
    );
    assert_eq!(
        *simd.max_i64x8(a, b),
        [1_i64, 0_i64, 3_i64, 0_i64, 5_i64, 0_i64, 7_i64, 0_i64]
    );
}

#[simd_test]
fn max_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[0_u64, 0_u64]);
    assert_eq!(*simd.max_u64x2(a, b), [1_u64, 2_u64]);
}

#[simd_test]
fn max_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[0_u64, 0_u64, 0_u64, 0_u64]);
    assert_eq!(*simd.max_u64x4(a, b), [1_u64, 2_u64, 3_u64, 4_u64]);
}

#[simd_test]
fn max_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64],
    );
    assert_eq!(
        *simd.max_u64x8(a, b),
        [1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn max_f64x2<S: Simd>(simd: S) {
    let a_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 2] =
        core::array::from_fn(|i| i as f64 + if i % 2 == 0 { 2.5_f64 } else { 0.5_f64 });
    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let expected: [f64; 2] = core::array::from_fn(|i| a_values[i].max(b_values[i]));
    let result = simd.max_f64x2(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn max_f64x4<S: Simd>(simd: S) {
    let a_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 4] =
        core::array::from_fn(|i| i as f64 + if i % 2 == 0 { 2.5_f64 } else { 0.5_f64 });
    let a = f64x4::from_slice(simd, &a_values);
    let b = f64x4::from_slice(simd, &b_values);
    let expected: [f64; 4] = core::array::from_fn(|i| a_values[i].max(b_values[i]));
    let result = simd.max_f64x4(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn max_integer_sign_boundaries<S: Simd>(simd: S) {
    let i8_a = [
        i8::MIN,
        i8::MAX,
        -1,
        0,
        1,
        -64,
        64,
        -127,
        127,
        -2,
        2,
        -65,
        65,
        -126,
        126,
        42,
    ];
    let i8_b = [
        i8::MAX,
        i8::MIN,
        0,
        -1,
        -1,
        64,
        -64,
        127,
        -127,
        2,
        -2,
        65,
        -65,
        126,
        -126,
        42,
    ];
    let expected_i8: [i8; 16] = core::array::from_fn(|i| i8_a[i].max(i8_b[i]));
    assert_eq!(
        simd.max_i8x16(
            i8x16::from_slice(simd, &i8_a),
            i8x16::from_slice(simd, &i8_b),
        )
        .as_slice(),
        expected_i8.as_slice(),
    );

    let u16_a = [0, 1, 0x7fff, 0x8000, 0x8001, 0xfffe, u16::MAX, 42];
    let u16_b = [u16::MAX, 0x8000, 0x8000, 0x7fff, 1, 0xffff, 0, 42];
    let expected_u16: [u16; 8] = core::array::from_fn(|i| u16_a[i].max(u16_b[i]));
    assert_eq!(
        simd.max_u16x8(
            u16x8::from_slice(simd, &u16_a),
            u16x8::from_slice(simd, &u16_b),
        )
        .as_slice(),
        expected_u16.as_slice(),
    );

    let i32_a = [i32::MIN, i32::MAX, -1, 0, 1, -0x4000_0000, 0x4000_0000, 42];
    let i32_b = [i32::MAX, i32::MIN, 0, -1, -1, 0x4000_0000, -0x4000_0000, 42];
    let expected_i32: [i32; 8] = core::array::from_fn(|i| i32_a[i].max(i32_b[i]));
    assert_eq!(
        simd.max_i32x8(
            i32x8::from_slice(simd, &i32_a),
            i32x8::from_slice(simd, &i32_b),
        )
        .as_slice(),
        expected_i32.as_slice(),
    );

    let u32_a = [
        0,
        1,
        0x7fff_ffff,
        0x8000_0000,
        0x8000_0001,
        0xffff_fffe,
        u32::MAX,
        42,
    ];
    let u32_b = [
        u32::MAX,
        0x8000_0000,
        0x8000_0000,
        0x7fff_ffff,
        1,
        u32::MAX,
        0,
        42,
    ];
    let expected_u32: [u32; 8] = core::array::from_fn(|i| u32_a[i].max(u32_b[i]));
    assert_eq!(
        simd.max_u32x8(
            u32x8::from_slice(simd, &u32_a),
            u32x8::from_slice(simd, &u32_b),
        )
        .as_slice(),
        expected_u32.as_slice(),
    );
}
