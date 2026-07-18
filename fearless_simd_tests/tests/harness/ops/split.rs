// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn split_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let (lo, hi) = simd.split_f32x8(a);
    assert_eq!(*lo, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(*hi, [5.0, 6.0, 7.0, 8.0]);
}

#[simd_test]
fn split_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let (lo, hi) = simd.split_i8x32(a);
    assert_eq!(*lo, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(
        *hi,
        [
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ]
    );
}

#[simd_test]
fn split_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    let (lo, hi) = simd.split_u8x32(a);
    assert_eq!(*lo, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(
        *hi,
        [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ]
    );
}

#[simd_test]
fn split_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let (lo, hi) = simd.split_i16x16(a);
    assert_eq!(*lo, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*hi, [9, 10, 11, 12, 13, 14, 15, 16]);
}

#[simd_test]
fn split_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let (lo, hi) = simd.split_u16x16(a);
    assert_eq!(*lo, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(*hi, [8, 9, 10, 11, 12, 13, 14, 15]);
}

#[simd_test]
fn split_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let (lo, hi) = simd.split_i32x8(a);
    assert_eq!(*lo, [1, 2, 3, 4]);
    assert_eq!(*hi, [5, 6, 7, 8]);
}

#[simd_test]
fn split_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let (lo, hi) = simd.split_u32x8(a);
    assert_eq!(*lo, [0, 1, 2, 3]);
    assert_eq!(*hi, [4, 5, 6, 7]);
}

#[simd_test]
fn split_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let (lo, hi) = simd.split_f64x4(a);
    assert_eq!(*lo, [1.0, 2.0]);
    assert_eq!(*hi, [3.0, 4.0]);
}

#[simd_test]
fn split_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let (lo, hi) = a.split();
    assert_eq!(*lo, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*hi, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
}

#[simd_test]
fn split_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ],
    );
    let (lo, hi) = a.split();
    assert_eq!(
        *lo,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        ]
    );
    assert_eq!(
        *hi,
        [
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64
        ]
    );
}

#[simd_test]
fn split_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    );
    let (lo, hi) = a.split();
    assert_eq!(
        *lo,
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
        ]
    );
    assert_eq!(
        *hi,
        [
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63
        ]
    );
}

#[simd_test]
fn split_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let (lo, hi) = a.split();
    assert_eq!(*lo, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(
        *hi,
        [
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ]
    );
}

#[simd_test]
fn split_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    let (lo, hi) = a.split();
    assert_eq!(*lo, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(
        *hi,
        [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ]
    );
}

#[simd_test]
fn split_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let (lo, hi) = a.split();
    assert_eq!(*lo, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*hi, [9, 10, 11, 12, 13, 14, 15, 16]);
}

#[simd_test]
fn split_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let (lo, hi) = a.split();
    assert_eq!(*lo, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(*hi, [8, 9, 10, 11, 12, 13, 14, 15]);
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn split_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, 2_i64, 3_i64, 4_i64]);
    let (lo, hi) = simd.split_i64x4(a);
    assert_eq!(*lo, [1_i64, 2_i64]);
    assert_eq!(*hi, [3_i64, 4_i64]);
}

#[simd_test]
fn split_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64],
    );
    let (lo, hi) = simd.split_i64x8(a);
    assert_eq!(*lo, [1_i64, 2_i64, 3_i64, 4_i64]);
    assert_eq!(*hi, [5_i64, 6_i64, 7_i64, 8_i64]);
}

#[simd_test]
fn split_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let (lo, hi) = simd.split_u64x4(a);
    assert_eq!(*lo, [1_u64, 2_u64]);
    assert_eq!(*hi, [3_u64, 4_u64]);
}

#[simd_test]
fn split_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let (lo, hi) = simd.split_u64x8(a);
    assert_eq!(*lo, [1_u64, 2_u64, 3_u64, 4_u64]);
    assert_eq!(*hi, [5_u64, 6_u64, 7_u64, 8_u64]);
}

#[simd_test]
fn split_mask64x4<S: Simd>(simd: S) {
    let a = mask64x4::from_slice(simd, &[-1_i64, 0_i64, -1_i64, 0_i64]);
    let (lo, hi) = simd.split_mask64x4(a);
    assert_eq!(<[i64; 2]>::from(lo), [-1_i64, 0_i64]);
    assert_eq!(<[i64; 2]>::from(hi), [-1_i64, 0_i64]);
}

#[simd_test]
fn split_mask64x8<S: Simd>(simd: S) {
    let a = mask64x8::from_slice(
        simd,
        &[-1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64],
    );
    let (lo, hi) = simd.split_mask64x8(a);
    assert_eq!(<[i64; 4]>::from(lo), [-1_i64, 0_i64, -1_i64, 0_i64]);
    assert_eq!(<[i64; 4]>::from(hi), [-1_i64, 0_i64, -1_i64, 0_i64]);
}
