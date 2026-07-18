// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn mul_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 100],
    );
    let b = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2],
    );

    assert_eq!(
        *(a * b),
        [
            0, 2, 6, 12, 20, 30, 70, 120, 180, 250, 74, 164, 8, 188, 132, 200
        ]
    );
}

#[simd_test]
fn mul_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            0, 1, -2, 3, -4, 5, 10, -15, 20, -25, 30, 35, -40, 50, -60, 100,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[1, 2, 3, -4, 5, -6, 7, 8, 9, 10, -11, 12, 13, -14, 15, 2],
    );

    assert_eq!(
        *(a * b),
        [
            0, 2, -6, -12, -20, -30, 70, -120, -76, 6, -74, -92, -8, 68, 124, -56
        ]
    );
}

#[simd_test]
fn mul_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[0, 5, 10, 30, 500, 0, 0, 0]);
    let b = u16x8::from_slice(simd, &[5, 8, 13, 21, 230, 0, 0, 0]);

    assert_eq!(*(a * b), [0, 40, 130, 630, 49464, 0, 0, 0]);
}

#[simd_test]
fn mul_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 5464, 23234, 456456]);
    let b = u32x4::from_slice(simd, &[23, 34, 565, 34234]);

    assert_eq!(*(a * b), [23, 185776, 13127210, 2741412816]);
}

#[simd_test]
fn mul_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, 0.0, 13.34, 234234.0]);
    let b = f32x4::from_slice(simd, &[-8.1, 7.9, -9.8, 3243.6]);

    assert_eq!(*(a * b), [83.43001, 0.0, -130.73201, 759761400.0]);
}

#[simd_test]
fn mul_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::from_slice(simd, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(*(a * b), [2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0]);
}

#[simd_test]
fn mul_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::from_slice(simd, &[2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(*(a * b), [2, 6, 12, 20, 30, 42, 56, 72]);
}

#[simd_test]
fn mul_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u32x8::from_slice(simd, &[2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(*(a * b), [2, 6, 12, 20, 30, 42, 56, 72]);
}

#[simd_test]
fn mul_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ],
    );
    assert_eq!(
        *(a * b),
        [
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
            32.0
        ]
    );
}

#[simd_test]
fn mul_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i32x16::from_slice(simd, &[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
    assert_eq!(
        *(a * b),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    );
}

#[simd_test]
fn mul_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u32x16::from_slice(simd, &[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
    assert_eq!(
        *(a * b),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    );
}

#[simd_test]
fn mul_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f64x8::from_slice(simd, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(*(a * b), [2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0]);
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn mul_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[3_i64, 3_i64]);
    assert_eq!(*simd.mul_i64x2(a, b), [3_i64, -6_i64]);
}

#[simd_test]
fn mul_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[3_i64, 3_i64, 3_i64, 3_i64]);
    assert_eq!(*simd.mul_i64x4(a, b), [3_i64, -6_i64, 9_i64, -12_i64]);
}

#[simd_test]
fn mul_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64],
    );
    assert_eq!(
        *simd.mul_i64x8(a, b),
        [
            3_i64, -6_i64, 9_i64, -12_i64, 15_i64, -18_i64, 21_i64, -24_i64
        ]
    );
}

#[simd_test]
fn mul_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[2_u64, 2_u64]);
    assert_eq!(*simd.mul_u64x2(a, b), [2_u64, 4_u64]);
}

#[simd_test]
fn mul_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[2_u64, 2_u64, 2_u64, 2_u64]);
    assert_eq!(*simd.mul_u64x4(a, b), [2_u64, 4_u64, 6_u64, 8_u64]);
}

#[simd_test]
fn mul_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[2_u64, 2_u64, 2_u64, 2_u64, 2_u64, 2_u64, 2_u64, 2_u64],
    );
    assert_eq!(
        *simd.mul_u64x8(a, b),
        [2_u64, 4_u64, 6_u64, 8_u64, 10_u64, 12_u64, 14_u64, 16_u64]
    );
}
