// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn reverse_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    assert_eq!(
        *a.reverse(),
        [
            -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1
        ]
    );
}

#[simd_test]
fn reverse_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    assert_eq!(
        *a.reverse(),
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    );
}

#[simd_test]
fn reverse_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    assert_eq!(*a.reverse(), [-8, 7, -6, 5, -4, 3, -2, 1]);
}

#[simd_test]
fn reverse_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(*a.reverse(), [7, 6, 5, 4, 3, 2, 1, 0]);
}

#[simd_test]
fn reverse_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, -2, 3, -4]);
    assert_eq!(*a.reverse(), [-4, 3, -2, 1]);
}

#[simd_test]
fn reverse_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0, 1, 2, 3]);
    assert_eq!(*a.reverse(), [3, 2, 1, 0]);
}

#[simd_test]
fn reverse_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1, -2]);
    assert_eq!(*a.reverse(), [-2, 1]);
}

#[simd_test]
fn reverse_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[0, 1]);
    assert_eq!(*a.reverse(), [1, 0]);
}

#[simd_test]
fn reverse_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, -2.0, 3.0, -4.0]);
    assert_eq!(*a.reverse(), [-4.0, 3.0, -2.0, 1.0]);
}

#[simd_test]
fn reverse_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.0, -2.0]);
    assert_eq!(*a.reverse(), [-2.0, 1.0]);
}

#[simd_test]
fn reverse_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    assert_eq!(
        *a.reverse(),
        [
            -32, 31, -30, 29, -28, 27, -26, 25, -24, 23, -22, 21, -20, 19, -18, 17, -16, 15, -14,
            13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1
        ]
    );
}

#[simd_test]
fn reverse_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    assert_eq!(
        *a.reverse(),
        [
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
            9, 8, 7, 6, 5, 4, 3, 2, 1, 0
        ]
    );
}

#[simd_test]
fn reverse_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    assert_eq!(
        *a.reverse(),
        [
            -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1
        ]
    );
}

#[simd_test]
fn reverse_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    assert_eq!(
        *a.reverse(),
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    );
}

#[simd_test]
fn reverse_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    assert_eq!(*a.reverse(), [-8, 7, -6, 5, -4, 3, -2, 1]);
}

#[simd_test]
fn reverse_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(*a.reverse(), [7, 6, 5, 4, 3, 2, 1, 0]);
}

#[simd_test]
fn reverse_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1, -2, 3, -4]);
    assert_eq!(*a.reverse(), [-4, 3, -2, 1]);
}

#[simd_test]
fn reverse_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[0, 1, 2, 3]);
    assert_eq!(*a.reverse(), [3, 2, 1, 0]);
}

#[simd_test]
fn reverse_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
    assert_eq!(*a.reverse(), [-8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0]);
}

#[simd_test]
fn reverse_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, -2.0, 3.0, -4.0]);
    assert_eq!(*a.reverse(), [-4.0, 3.0, -2.0, 1.0]);
}

#[simd_test]
fn reverse_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32, 33, -34, 35, -36, 37, -38, 39, -40,
            41, -42, 43, -44, 45, -46, 47, -48, 49, -50, 51, -52, 53, -54, 55, -56, 57, -58, 59,
            -60, 61, -62, 63, -64,
        ],
    );
    assert_eq!(
        *a.reverse(),
        [
            -64, 63, -62, 61, -60, 59, -58, 57, -56, 55, -54, 53, -52, 51, -50, 49, -48, 47, -46,
            45, -44, 43, -42, 41, -40, 39, -38, 37, -36, 35, -34, 33, -32, 31, -30, 29, -28, 27,
            -26, 25, -24, 23, -22, 21, -20, 19, -18, 17, -16, 15, -14, 13, -12, 11, -10, 9, -8, 7,
            -6, 5, -4, 3, -2, 1
        ]
    );
}

#[simd_test]
fn reverse_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    );
    assert_eq!(
        *a.reverse(),
        [
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
            41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
            19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
        ]
    );
}

#[simd_test]
fn reverse_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    assert_eq!(
        *a.reverse(),
        [
            -32, 31, -30, 29, -28, 27, -26, 25, -24, 23, -22, 21, -20, 19, -18, 17, -16, 15, -14,
            13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1
        ]
    );
}

#[simd_test]
fn reverse_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    assert_eq!(
        *a.reverse(),
        [
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
            9, 8, 7, 6, 5, 4, 3, 2, 1, 0
        ]
    );
}

#[simd_test]
fn reverse_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    assert_eq!(
        *a.reverse(),
        [
            -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1
        ]
    );
}

#[simd_test]
fn reverse_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    assert_eq!(
        *a.reverse(),
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    );
}

#[simd_test]
fn reverse_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    assert_eq!(*a.reverse(), [-8, 7, -6, 5, -4, 3, -2, 1]);
}

#[simd_test]
fn reverse_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(*a.reverse(), [7, 6, 5, 4, 3, 2, 1, 0]);
}

#[simd_test]
fn reverse_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0,
            -16.0,
        ],
    );
    assert_eq!(
        *a.reverse(),
        [
            -16.0, 15.0, -14.0, 13.0, -12.0, 11.0, -10.0, 9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0,
            -2.0, 1.0
        ]
    );
}

#[simd_test]
fn reverse_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
    assert_eq!(*a.reverse(), [-8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0]);
}

#[simd_test]
fn reverse_mask8x16<S: Simd>(simd: S) {
    let lanes: [i8; 16] = [-1, 0, -1, 0, -1, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0];
    let a = mask8x16::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i8; 16]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask16x8<S: Simd>(simd: S) {
    let lanes: [i16; 8] = [-1, 0, -1, 0, -1, -1, 0, 0];
    let a = mask16x8::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i16; 8]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask32x4<S: Simd>(simd: S) {
    let lanes: [i32; 4] = [-1, 0, -1, 0];
    let a = mask32x4::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i32; 4]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask64x2<S: Simd>(simd: S) {
    let lanes: [i64; 2] = [-1, 0];
    let a = mask64x2::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i64; 2]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask8x32<S: Simd>(simd: S) {
    let lanes: [i8; 32] = [
        -1, -1, -1, 0, 0, -1, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1, -1, 0, 0, 0, -1, 0, 0, -1, 0,
        0, 0, 0, 0, 0, 0,
    ];
    let a = mask8x32::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i8; 32]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask16x16<S: Simd>(simd: S) {
    let lanes: [i16; 16] = [-1, 0, -1, 0, -1, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0];
    let a = mask16x16::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i16; 16]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask32x8<S: Simd>(simd: S) {
    let lanes: [i32; 8] = [-1, 0, -1, 0, -1, -1, 0, 0];
    let a = mask32x8::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i32; 8]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask64x4<S: Simd>(simd: S) {
    let lanes: [i64; 4] = [-1, 0, -1, 0];
    let a = mask64x4::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i64; 4]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask8x64<S: Simd>(simd: S) {
    let lanes: [i8; 64] = [
        -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1,
        -1, 0, 0, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0, -1, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1, -1,
        0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
    ];
    let a = mask8x64::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i8; 64]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask16x32<S: Simd>(simd: S) {
    let lanes: [i16; 32] = [
        -1, -1, -1, 0, 0, -1, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1, -1, 0, 0, 0, -1, 0, 0, -1, 0,
        0, 0, 0, 0, 0, 0,
    ];
    let a = mask16x32::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i16; 32]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask32x16<S: Simd>(simd: S) {
    let lanes: [i32; 16] = [-1, 0, -1, 0, -1, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0];
    let a = mask32x16::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i32; 16]>::from(a.reverse()), expected);
}

#[simd_test]
fn reverse_mask64x8<S: Simd>(simd: S) {
    let lanes: [i64; 8] = [-1, 0, -1, 0, -1, -1, 0, 0];
    let a = mask64x8::from_slice(simd, &lanes);
    let mut expected = lanes;
    expected.reverse();
    assert_eq!(<[i64; 8]>::from(a.reverse()), expected);
}
