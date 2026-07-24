// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn add_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*(a + b), [6.0, 8.0, 10.0, 12.0]);
}

#[simd_test]
fn add_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8],
    );
    let b = i8x16::from_slice(
        simd,
        &[10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8],
    );
    assert_eq!(
        *(a + b),
        [11, 22, 33, 44, 55, 66, 77, 88, 0, 0, 0, 0, 0, 0, 0, 0]
    );
}

#[simd_test]
fn add_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176
        ]
    );
}

#[simd_test]
fn add_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    assert_eq!(*(a + b), [11, 22, 33, 44, 55, 66, 77, 88]);
}

#[simd_test]
fn add_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u16x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    assert_eq!(*(a + b), [11, 22, 33, 44, 55, 66, 77, 88]);
}

#[simd_test]
fn add_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i32x4::from_slice(simd, &[10, 20, 30, 40]);
    assert_eq!(*(a + b), [11, 22, 33, 44]);
}

#[simd_test]
fn add_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u32x4::from_slice(simd, &[10, 20, 30, 40]);
    assert_eq!(*(a + b), [11, 22, 33, 44]);
}

#[simd_test]
fn wrapping_add_u32<S: Simd>(simd: S) {
    assert_eq!(
        (S::u32s::splat(simd, u32::MAX) + 1).as_slice(),
        &vec![0; S::u32s::N]
    );
}

#[simd_test]
fn add_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::from_slice(simd, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    assert_eq!(*(a + b), [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]);
}

#[simd_test]
fn add_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2,
            -3, -4, -5, -6, -7, -8,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80,
            1, 2, 3, 4, 5, 6, 7, 8,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 0, 0, 0, 0, 0, 0, 0, 0, 11, 22, 33, 44, 55, 66, 77, 88,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    );
}

#[simd_test]
fn add_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40,
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 11, 22, 33, 44,
            55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176
        ]
    );
}

#[simd_test]
fn add_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176
        ]
    );
}

#[simd_test]
fn add_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176
        ]
    );
}

#[simd_test]
fn add_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    assert_eq!(*(a + b), [11, 22, 33, 44, 55, 66, 77, 88]);
}

#[simd_test]
fn add_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u32x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    assert_eq!(*(a + b), [11, 22, 33, 44, 55, 66, 77, 88]);
}

#[simd_test]
fn add_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0,
            140.0, 150.0, 160.0,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0, 110.0, 121.0, 132.0, 143.0,
            154.0, 165.0, 176.0
        ]
    );
}

#[simd_test]
fn add_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2,
            -3, -4, -5, -6, -7, -8, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8, 1, 2,
            3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80,
            1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20,
            30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 0, 0, 0, 0, 0, 0, 0, 0, 11, 22, 33, 44, 55, 66, 77, 88,
            0, 0, 0, 0, 0, 0, 0, 0, 11, 22, 33, 44, 55, 66, 77, 88, 0, 0, 0, 0, 0, 0, 0, 0, 11, 22,
            33, 44, 55, 66, 77, 88, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    );
}

#[simd_test]
fn add_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ],
    );
    let b = u8x64::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40,
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40, 50, 60, 70, 80,
            90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
            120, 130, 140, 150, 160,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 11, 22, 33, 44,
            55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 11, 22, 33, 44, 55, 66, 77, 88,
            99, 110, 121, 132, 143, 154, 165, 176, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121,
            132, 143, 154, 165, 176
        ]
    );
}

#[simd_test]
fn add_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16,
        ],
    );
    let b = i16x32::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40,
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 11, 22, 33, 44,
            55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176
        ]
    );
}

#[simd_test]
fn add_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16,
        ],
    );
    let b = u16x32::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40,
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 11, 22, 33, 44,
            55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176
        ]
    );
}

#[simd_test]
fn add_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176
        ]
    );
}

#[simd_test]
fn add_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u32x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176
        ]
    );
}

#[simd_test]
fn add_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f64x8::from_slice(simd, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    assert_eq!(*(a + b), [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]);
}

// Additional concrete rows for this operation.

#[simd_test]
fn add_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[3_i64, 3_i64]);
    assert_eq!(*simd.add_i64x2(a, b), [4_i64, 1_i64]);
}

#[simd_test]
fn add_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[3_i64, 3_i64, 3_i64, 3_i64]);
    assert_eq!(*simd.add_i64x4(a, b), [4_i64, 1_i64, 6_i64, -1_i64]);
}

#[simd_test]
fn add_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64],
    );
    assert_eq!(
        *simd.add_i64x8(a, b),
        [4_i64, 1_i64, 6_i64, -1_i64, 8_i64, -3_i64, 10_i64, -5_i64]
    );
}

#[simd_test]
fn add_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[2_u64, 2_u64]);
    assert_eq!(*simd.add_u64x2(a, b), [3_u64, 4_u64]);
}

#[simd_test]
fn add_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[2_u64, 2_u64, 2_u64, 2_u64]);
    assert_eq!(*simd.add_u64x4(a, b), [3_u64, 4_u64, 5_u64, 6_u64]);
}

#[simd_test]
fn add_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[2_u64, 2_u64, 2_u64, 2_u64, 2_u64, 2_u64, 2_u64, 2_u64],
    );
    assert_eq!(
        *simd.add_u64x8(a, b),
        [3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64, 9_u64, 10_u64]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn add_f64x2<S: Simd>(simd: S) {
    let a_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 2] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let expected: [f64; 2] = core::array::from_fn(|i| a_values[i] + b_values[i]);
    let result = simd.add_f64x2(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn add_f64x4<S: Simd>(simd: S) {
    let a_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 4] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x4::from_slice(simd, &a_values);
    let b = f64x4::from_slice(simd, &b_values);
    let expected: [f64; 4] = core::array::from_fn(|i| a_values[i] + b_values[i]);
    let result = simd.add_f64x4(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}
