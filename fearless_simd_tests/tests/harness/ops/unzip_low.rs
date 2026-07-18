// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn unzip_low_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*simd.unzip_low_f32x4(a, b), [1.0, 3.0, 5.0, 7.0]);
}

#[simd_test]
fn unzip_low_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b = f32x8::from_slice(simd, &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    assert_eq!(
        *simd.unzip_low_f32x8(a, b),
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    );
}

#[simd_test]
fn unzip_low_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_low_i8x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_low_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ],
    );
    assert_eq!(
        *simd.unzip_low_i8x32(a, b),
        [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45,
            47, 49, 51, 53, 55, 57, 59, 61, 63
        ]
    );
}

#[simd_test]
fn unzip_low_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_low_u8x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_low_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ],
    );
    assert_eq!(
        *simd.unzip_low_u8x32(a, b),
        [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45,
            47, 49, 51, 53, 55, 57, 59, 61, 63
        ]
    );
}

#[simd_test]
fn unzip_low_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_low_i16x8(a, b), [1, 3, 5, 7, 9, 11, 13, 15]);
}

#[simd_test]
fn unzip_low_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u16x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_low_u16x8(a, b), [1, 3, 5, 7, 9, 11, 13, 15]);
}

#[simd_test]
fn unzip_low_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_low_i16x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_low_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_low_u16x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_low_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*simd.unzip_low_i32x4(a, b), [1, 3, 5, 7]);
}

#[simd_test]
fn unzip_low_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*simd.unzip_low_u32x4(a, b), [1, 3, 5, 7]);
}

#[simd_test]
fn unzip_low_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_low_i32x8(a, b), [1, 3, 5, 7, 9, 11, 13, 15]);
}

#[simd_test]
fn unzip_low_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_low_u32x8(a, b), [1, 3, 5, 7, 9, 11, 13, 15]);
}

#[simd_test]
fn unzip_low_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.0, 2.0]);
    let b = f64x2::from_slice(simd, &[3.0, 4.0]);
    assert_eq!(*simd.unzip_low_f64x2(a, b), [1.0, 3.0]);
}

#[simd_test]
fn unzip_low_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*simd.unzip_low_f64x4(a, b), [1.0, 3.0, 5.0, 7.0]);
}

#[simd_test]
fn unzip_low_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ],
    );
    // unzip_low takes all even-indexed elements
    assert_eq!(
        *simd.unzip_low_f32x16(a, b),
        [
            1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
            31.0
        ]
    );
}

#[simd_test]
fn unzip_low_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    // unzip_low takes even-indexed elements from each vector
    assert_eq!(
        *simd.unzip_low_i32x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_low_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u32x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    // unzip_low takes even-indexed elements from each vector
    assert_eq!(
        *simd.unzip_low_u32x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn unzip_low_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1_i16, 2_i16, 3_i16, 4_i16, 5_i16, 6_i16, 7_i16, 8_i16, 9_i16, 10_i16, 11_i16, 12_i16,
            13_i16, 14_i16, 15_i16, 16_i16, 17_i16, 18_i16, 19_i16, 20_i16, 21_i16, 22_i16, 23_i16,
            24_i16, 25_i16, 26_i16, 27_i16, 28_i16, 29_i16, 30_i16, 31_i16, 32_i16,
        ],
    );
    let b = i16x32::from_slice(
        simd,
        &[
            33_i16, 34_i16, 35_i16, 36_i16, 37_i16, 38_i16, 39_i16, 40_i16, 41_i16, 42_i16, 43_i16,
            44_i16, 45_i16, 46_i16, 47_i16, 48_i16, 49_i16, 50_i16, 51_i16, 52_i16, 53_i16, 54_i16,
            55_i16, 56_i16, 57_i16, 58_i16, 59_i16, 60_i16, 61_i16, 62_i16, 63_i16, 64_i16,
        ],
    );
    assert_eq!(
        *simd.unzip_low_i16x32(a, b),
        [
            1_i16, 3_i16, 5_i16, 7_i16, 9_i16, 11_i16, 13_i16, 15_i16, 17_i16, 19_i16, 21_i16,
            23_i16, 25_i16, 27_i16, 29_i16, 31_i16, 33_i16, 35_i16, 37_i16, 39_i16, 41_i16, 43_i16,
            45_i16, 47_i16, 49_i16, 51_i16, 53_i16, 55_i16, 57_i16, 59_i16, 61_i16, 63_i16
        ]
    );
}

#[simd_test]
fn unzip_low_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            1_u16, 2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16, 9_u16, 10_u16, 11_u16, 12_u16,
            13_u16, 14_u16, 15_u16, 16_u16, 17_u16, 18_u16, 19_u16, 20_u16, 21_u16, 22_u16, 23_u16,
            24_u16, 25_u16, 26_u16, 27_u16, 28_u16, 29_u16, 30_u16, 31_u16, 32_u16,
        ],
    );
    let b = u16x32::from_slice(
        simd,
        &[
            33_u16, 34_u16, 35_u16, 36_u16, 37_u16, 38_u16, 39_u16, 40_u16, 41_u16, 42_u16, 43_u16,
            44_u16, 45_u16, 46_u16, 47_u16, 48_u16, 49_u16, 50_u16, 51_u16, 52_u16, 53_u16, 54_u16,
            55_u16, 56_u16, 57_u16, 58_u16, 59_u16, 60_u16, 61_u16, 62_u16, 63_u16, 64_u16,
        ],
    );
    assert_eq!(
        *simd.unzip_low_u16x32(a, b),
        [
            1_u16, 3_u16, 5_u16, 7_u16, 9_u16, 11_u16, 13_u16, 15_u16, 17_u16, 19_u16, 21_u16,
            23_u16, 25_u16, 27_u16, 29_u16, 31_u16, 33_u16, 35_u16, 37_u16, 39_u16, 41_u16, 43_u16,
            45_u16, 47_u16, 49_u16, 51_u16, 53_u16, 55_u16, 57_u16, 59_u16, 61_u16, 63_u16
        ]
    );
}

#[simd_test]
fn unzip_low_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, 2_i64]);
    let b = i64x2::from_slice(simd, &[3_i64, 4_i64]);
    assert_eq!(*simd.unzip_low_i64x2(a, b), [1_i64, 3_i64]);
}

#[simd_test]
fn unzip_low_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, 2_i64, 3_i64, 4_i64]);
    let b = i64x4::from_slice(simd, &[5_i64, 6_i64, 7_i64, 8_i64]);
    assert_eq!(*simd.unzip_low_i64x4(a, b), [1_i64, 3_i64, 5_i64, 7_i64]);
}

#[simd_test]
fn unzip_low_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[
            9_i64, 10_i64, 11_i64, 12_i64, 13_i64, 14_i64, 15_i64, 16_i64,
        ],
    );
    assert_eq!(
        *simd.unzip_low_i64x8(a, b),
        [1_i64, 3_i64, 5_i64, 7_i64, 9_i64, 11_i64, 13_i64, 15_i64]
    );
}

#[simd_test]
fn unzip_low_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[3_u64, 4_u64]);
    assert_eq!(*simd.unzip_low_u64x2(a, b), [1_u64, 3_u64]);
}

#[simd_test]
fn unzip_low_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[5_u64, 6_u64, 7_u64, 8_u64]);
    assert_eq!(*simd.unzip_low_u64x4(a, b), [1_u64, 3_u64, 5_u64, 7_u64]);
}

#[simd_test]
fn unzip_low_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[
            9_u64, 10_u64, 11_u64, 12_u64, 13_u64, 14_u64, 15_u64, 16_u64,
        ],
    );
    assert_eq!(
        *simd.unzip_low_u64x8(a, b),
        [1_u64, 3_u64, 5_u64, 7_u64, 9_u64, 11_u64, 13_u64, 15_u64]
    );
}
