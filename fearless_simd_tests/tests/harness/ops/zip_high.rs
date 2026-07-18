// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn zip_high_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[0.0, 1.0, 2.0, 3.0]);
    let b = f32x4::from_slice(simd, &[4.0, 5.0, 6.0, 7.0]);
    assert_eq!(*simd.zip_high_f32x4(a, b), [2.0, 6.0, 3.0, 7.0]);
}

#[simd_test]
fn zip_high_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b = f32x8::from_slice(simd, &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    assert_eq!(
        *simd.zip_high_f32x8(a, b),
        [4.0, 12.0, 5.0, 13.0, 6.0, 14.0, 7.0, 15.0]
    );
}

#[simd_test]
fn zip_high_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    assert_eq!(
        *simd.zip_high_i8x16(a, b),
        [
            9, 25, -10, -26, 11, 27, -12, -28, 13, 29, -14, -30, 15, 31, -16, -32
        ]
    );
}

#[simd_test]
fn zip_high_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            33, -34, 35, -36, 37, -38, 39, -40, 41, -42, 43, -44, 45, -46, 47, -48, 49, -50, 51,
            -52, 53, -54, 55, -56, 57, -58, 59, -60, 61, -62, 63, -64,
        ],
    );
    assert_eq!(
        *simd.zip_high_i8x32(a, b),
        [
            17, 49, -18, -50, 19, 51, -20, -52, 21, 53, -22, -54, 23, 55, -24, -56, 25, 57, -26,
            -58, 27, 59, -28, -60, 29, 61, -30, -62, 31, 63, -32, -64
        ]
    );
}

#[simd_test]
fn zip_high_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    assert_eq!(
        *simd.zip_high_u8x16(a, b),
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    );
}

#[simd_test]
fn zip_high_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    );
    assert_eq!(
        *simd.zip_high_u8x32(a, b),
        [
            16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58,
            27, 59, 28, 60, 29, 61, 30, 62, 31, 63
        ]
    );
}

#[simd_test]
fn zip_high_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i16x8::from_slice(simd, &[9, -10, 11, -12, 13, -14, 15, -16]);
    assert_eq!(*simd.zip_high_i16x8(a, b), [5, 13, -6, -14, 7, 15, -8, -16]);
}

#[simd_test]
fn zip_high_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    assert_eq!(
        *simd.zip_high_i16x16(a, b),
        [
            9, 25, -10, -26, 11, 27, -12, -28, 13, 29, -14, -30, 15, 31, -16, -32
        ]
    );
}

#[simd_test]
fn zip_high_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let b = u16x8::from_slice(simd, &[8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(*simd.zip_high_u16x8(a, b), [4, 12, 5, 13, 6, 14, 7, 15]);
}

#[simd_test]
fn zip_high_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    assert_eq!(
        *simd.zip_high_u16x16(a, b),
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    );
}

#[simd_test]
fn zip_high_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, -2, 3, -4]);
    let b = i32x4::from_slice(simd, &[5, -6, 7, -8]);
    assert_eq!(*simd.zip_high_i32x4(a, b), [3, 7, -4, -8]);
}

#[simd_test]
fn zip_high_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i32x8::from_slice(simd, &[9, -10, 11, -12, 13, -14, 15, -16]);
    assert_eq!(*simd.zip_high_i32x8(a, b), [5, 13, -6, -14, 7, 15, -8, -16]);
}

#[simd_test]
fn zip_high_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0, 1, 2, 3]);
    let b = u32x4::from_slice(simd, &[4, 5, 6, 7]);
    assert_eq!(*simd.zip_high_u32x4(a, b), [2, 6, 3, 7]);
}

#[simd_test]
fn zip_high_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let b = u32x8::from_slice(simd, &[8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(*simd.zip_high_u32x8(a, b), [4, 12, 5, 13, 6, 14, 7, 15]);
}

#[simd_test]
fn zip_high_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*simd.zip_high_f64x4(a, b), [3.0, 7.0, 4.0, 8.0]);
}

#[simd_test]
fn zip_high_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0,
        ],
    );
    // zip_high interleaves the second half of each 256-bit lane
    assert_eq!(
        *simd.zip_high_f32x16(a, b),
        [
            8.0, 24.0, 9.0, 25.0, 10.0, 26.0, 11.0, 27.0, 12.0, 28.0, 13.0, 29.0, 14.0, 30.0, 15.0,
            31.0
        ]
    );
}

#[simd_test]
fn zip_high_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
            106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
            123, 124, 125, 126, 127,
        ],
    );
    // zip_high takes the second half of the 512-bit vectors and interleaves them
    assert_eq!(
        *simd.zip_high_i8x64(a, b),
        [
            32, 96, 33, 97, 34, 98, 35, 99, 36, 100, 37, 101, 38, 102, 39, 103, 40, 104, 41, 105,
            42, 106, 43, 107, 44, 108, 45, 109, 46, 110, 47, 111, 48, 112, 49, 113, 50, 114, 51,
            115, 52, 116, 53, 117, 54, 118, 55, 119, 56, 120, 57, 121, 58, 122, 59, 123, 60, 124,
            61, 125, 62, 126, 63, 127
        ]
    );
}

#[simd_test]
fn zip_high_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    // zip_high takes the second half of each vector and interleaves them
    assert_eq!(
        *simd.zip_high_i32x16(a, b),
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    );
}

#[simd_test]
fn zip_high_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u32x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    // zip_high takes the second half of each vector and interleaves them
    assert_eq!(
        *simd.zip_high_u32x16(a, b),
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    );
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn zip_high_i16x32<S: Simd>(simd: S) {
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
        *simd.zip_high_i16x32(a, b),
        [
            17_i16, 49_i16, 18_i16, 50_i16, 19_i16, 51_i16, 20_i16, 52_i16, 21_i16, 53_i16, 22_i16,
            54_i16, 23_i16, 55_i16, 24_i16, 56_i16, 25_i16, 57_i16, 26_i16, 58_i16, 27_i16, 59_i16,
            28_i16, 60_i16, 29_i16, 61_i16, 30_i16, 62_i16, 31_i16, 63_i16, 32_i16, 64_i16
        ]
    );
}

#[simd_test]
fn zip_high_u16x32<S: Simd>(simd: S) {
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
        *simd.zip_high_u16x32(a, b),
        [
            17_u16, 49_u16, 18_u16, 50_u16, 19_u16, 51_u16, 20_u16, 52_u16, 21_u16, 53_u16, 22_u16,
            54_u16, 23_u16, 55_u16, 24_u16, 56_u16, 25_u16, 57_u16, 26_u16, 58_u16, 27_u16, 59_u16,
            28_u16, 60_u16, 29_u16, 61_u16, 30_u16, 62_u16, 31_u16, 63_u16, 32_u16, 64_u16
        ]
    );
}

#[simd_test]
fn zip_high_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, 2_i64]);
    let b = i64x2::from_slice(simd, &[3_i64, 4_i64]);
    assert_eq!(*simd.zip_high_i64x2(a, b), [2_i64, 4_i64]);
}

#[simd_test]
fn zip_high_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, 2_i64, 3_i64, 4_i64]);
    let b = i64x4::from_slice(simd, &[5_i64, 6_i64, 7_i64, 8_i64]);
    assert_eq!(*simd.zip_high_i64x4(a, b), [3_i64, 7_i64, 4_i64, 8_i64]);
}

#[simd_test]
fn zip_high_i64x8<S: Simd>(simd: S) {
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
        *simd.zip_high_i64x8(a, b),
        [5_i64, 13_i64, 6_i64, 14_i64, 7_i64, 15_i64, 8_i64, 16_i64]
    );
}

#[simd_test]
fn zip_high_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[3_u64, 4_u64]);
    assert_eq!(*simd.zip_high_u64x2(a, b), [2_u64, 4_u64]);
}

#[simd_test]
fn zip_high_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[5_u64, 6_u64, 7_u64, 8_u64]);
    assert_eq!(*simd.zip_high_u64x4(a, b), [3_u64, 7_u64, 4_u64, 8_u64]);
}

#[simd_test]
fn zip_high_u64x8<S: Simd>(simd: S) {
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
        *simd.zip_high_u64x8(a, b),
        [5_u64, 13_u64, 6_u64, 14_u64, 7_u64, 15_u64, 8_u64, 16_u64]
    );
}
