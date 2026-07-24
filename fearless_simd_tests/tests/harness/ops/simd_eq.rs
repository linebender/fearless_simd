// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn simd_eq_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 2.0, 1.0, 0.0]);
    let b = f32x4::from_slice(simd, &[4.0, 3.1, 1.0, 0.0]);
    assert_eq!(<[i32; 4]>::from(a.simd_eq(b)), [-1, 0, -1, -1]);
}

#[simd_test]
fn simd_eq_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i8x16::from_slice(simd, &[1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0]);
    assert_eq!(
        <[i8; 16]>::from(a.simd_eq(b)),
        [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]
    );
}

#[simd_test]
fn simd_eq_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x8::from_slice(simd, &[1, 0, 3, 0, 5, 0, 7, 0]);
    assert_eq!(<[i16; 8]>::from(a.simd_eq(b)), [-1, 0, -1, 0, -1, 0, -1, 0]);
}

#[simd_test]
fn simd_eq_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 32768, 40000, 65535, 6, 7, 8]);
    let b = u16x8::from_slice(simd, &[1, 0, 32768, 0, 65535, 0, 7, 0]);
    assert_eq!(<[i16; 8]>::from(a.simd_eq(b)), [-1, 0, -1, 0, -1, 0, -1, 0]);
}

#[simd_test]
fn simd_eq_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i32x4::from_slice(simd, &[1, 0, 3, 0]);
    assert_eq!(<[i32; 4]>::from(a.simd_eq(b)), [-1, 0, -1, 0]);
}

#[simd_test]
fn simd_eq_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 2147483648, 4294967295]);
    let b = u32x4::from_slice(simd, &[1, 0, 2147483648, 0]);
    assert_eq!(<[i32; 4]>::from(a.simd_eq(b)), [-1, 0, -1, 0]);
}

#[simd_test]
fn simd_eq_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[1, 2, 128, 200, 255, 6, 7, 8, 1, 2, 128, 200, 255, 6, 7, 8],
    );
    let b = u8x16::from_slice(
        simd,
        &[1, 0, 128, 0, 255, 0, 7, 0, 1, 0, 128, 0, 255, 0, 7, 0],
    );
    assert_eq!(
        <[i8; 16]>::from(a.simd_eq(b)),
        [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]
    );
}

#[simd_test]
fn simd_eq_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[4.0, 2.0, 1.0, 0.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::from_slice(simd, &[4.0, 3.1, 1.0, 0.0, 5.0, 7.0, 7.0, 9.0]);
    assert_eq!(
        <[i32; 8]>::from(a.simd_eq(b)),
        [-1, 0, -1, -1, -1, 0, -1, 0]
    );
}

#[simd_test]
fn simd_eq_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5,
            0, 7, 0,
        ],
    );
    assert_eq!(
        <[i8; 32]>::from(a.simd_eq(b)),
        [
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0
        ]
    );
}

#[simd_test]
fn simd_eq_u8x32<S: Simd>(simd: S) {
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
            1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0,
            13, 0, 15, 0,
        ],
    );
    assert_eq!(
        <[i8; 32]>::from(a.simd_eq(b)),
        [
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0
        ]
    );
}

#[simd_test]
fn simd_eq_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x16::from_slice(simd, &[1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0]);
    assert_eq!(
        <[i16; 16]>::from(a.simd_eq(b)),
        [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]
    );
}

#[simd_test]
fn simd_eq_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u16x16::from_slice(simd, &[1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0]);
    assert_eq!(
        <[i16; 16]>::from(a.simd_eq(b)),
        [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]
    );
}

#[simd_test]
fn simd_eq_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::from_slice(simd, &[1, 0, 3, 0, 5, 0, 7, 0]);
    assert_eq!(<[i32; 8]>::from(a.simd_eq(b)), [-1, 0, -1, 0, -1, 0, -1, 0]);
}

#[simd_test]
fn simd_eq_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u32x8::from_slice(simd, &[1, 0, 3, 0, 5, 0, 7, 0]);
    assert_eq!(<[i32; 8]>::from(a.simd_eq(b)), [-1, 0, -1, 0, -1, 0, -1, 0]);
}

#[simd_test]
fn simd_eq_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            4.0, 2.0, 1.0, 0.0, 5.0, 6.0, 7.0, 8.0, 4.0, 2.0, 1.0, 0.0, 5.0, 6.0, 7.0, 8.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            4.0, 3.1, 1.0, 0.0, 5.0, 7.0, 7.0, 9.0, 4.0, 3.1, 1.0, 0.0, 5.0, 7.0, 7.0, 9.0,
        ],
    );
    assert_eq!(
        <[i32; 16]>::from(a.simd_eq(b)),
        [-1, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, 0, -1, 0]
    );
}

#[simd_test]
fn simd_eq_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2,
            3, 4, 5, 6, 7, 8,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5,
            0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0,
            3, 0, 5, 0, 7, 0,
        ],
    );
    assert_eq!(
        <[i8; 64]>::from(a.simd_eq(b)),
        [
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0
        ]
    );
}

#[simd_test]
fn simd_eq_u8x64<S: Simd>(simd: S) {
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
            1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0,
            13, 0, 15, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 1, 0, 3, 0, 5, 0, 7,
            0, 9, 0, 11, 0, 13, 0, 15, 0,
        ],
    );
    assert_eq!(
        <[i8; 64]>::from(a.simd_eq(b)),
        [
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0
        ]
    );
}

#[simd_test]
fn simd_eq_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8,
        ],
    );
    let b = i16x32::from_slice(
        simd,
        &[
            1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5,
            0, 7, 0,
        ],
    );
    assert_eq!(
        <[i16; 32]>::from(a.simd_eq(b)),
        [
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0
        ]
    );
}

#[simd_test]
fn simd_eq_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8,
        ],
    );
    let b = u16x32::from_slice(
        simd,
        &[
            1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5,
            0, 7, 0,
        ],
    );
    assert_eq!(
        <[i16; 32]>::from(a.simd_eq(b)),
        [
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0
        ]
    );
}

#[simd_test]
fn simd_eq_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x16::from_slice(simd, &[1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0]);
    assert_eq!(
        <[i32; 16]>::from(a.simd_eq(b)),
        [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]
    );
}

#[simd_test]
fn simd_eq_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u32x16::from_slice(simd, &[1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0]);
    assert_eq!(
        <[i32; 16]>::from(a.simd_eq(b)),
        [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]
    );
}

#[simd_test]
fn simd_eq_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[4.0, 2.0, 1.0, 0.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f64x8::from_slice(simd, &[4.0, 3.1, 1.0, 0.0, 5.0, 7.0, 7.0, 9.0]);
    assert_eq!(
        <[i64; 8]>::from(a.simd_eq(b)),
        [-1, 0, -1, -1, -1, 0, -1, 0]
    );
}

// Additional concrete rows for this operation.

#[simd_test]
fn simd_eq_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[0_i64, 0_i64]);
    assert_eq!(<[i64; 2]>::from(simd.simd_eq_i64x2(a, b)), [0_i64, 0_i64]);
}

#[simd_test]
fn simd_eq_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[0_i64, 0_i64, 0_i64, 0_i64]);
    assert_eq!(
        <[i64; 4]>::from(simd.simd_eq_i64x4(a, b)),
        [0_i64, 0_i64, 0_i64, 0_i64]
    );
}

#[simd_test]
fn simd_eq_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64],
    );
    assert_eq!(
        <[i64; 8]>::from(simd.simd_eq_i64x8(a, b)),
        [0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64]
    );
}

#[simd_test]
fn simd_eq_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[0_u64, 0_u64]);
    assert_eq!(<[i64; 2]>::from(simd.simd_eq_u64x2(a, b)), [0_i64, 0_i64]);
}

#[simd_test]
fn simd_eq_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[0_u64, 0_u64, 0_u64, 0_u64]);
    assert_eq!(
        <[i64; 4]>::from(simd.simd_eq_u64x4(a, b)),
        [0_i64, 0_i64, 0_i64, 0_i64]
    );
}

#[simd_test]
fn simd_eq_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64],
    );
    assert_eq!(
        <[i64; 8]>::from(simd.simd_eq_u64x8(a, b)),
        [0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64]
    );
}

#[simd_test]
fn simd_eq_mask64x2<S: Simd>(simd: S) {
    let a = mask64x2::from_slice(simd, &[-1_i64, 0_i64]);
    let b = mask64x2::from_slice(simd, &[0_i64, -1_i64]);
    assert_eq!(<[i64; 2]>::from(simd.simd_eq_mask64x2(a, a)), [-1; 2]);
    assert_eq!(
        <[i64; 2]>::from(simd.simd_eq_mask64x2(a, b)),
        [0_i64, 0_i64]
    );
}

#[simd_test]
fn simd_eq_mask64x4<S: Simd>(simd: S) {
    let a = mask64x4::from_slice(simd, &[-1_i64, 0_i64, -1_i64, 0_i64]);
    let b = mask64x4::from_slice(simd, &[0_i64, -1_i64, 0_i64, -1_i64]);
    assert_eq!(<[i64; 4]>::from(simd.simd_eq_mask64x4(a, a)), [-1; 4]);
    assert_eq!(
        <[i64; 4]>::from(simd.simd_eq_mask64x4(a, b)),
        [0_i64, 0_i64, 0_i64, 0_i64]
    );
}

#[simd_test]
fn simd_eq_mask64x8<S: Simd>(simd: S) {
    let a = mask64x8::from_slice(
        simd,
        &[-1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64],
    );
    let b = mask64x8::from_slice(
        simd,
        &[0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64],
    );
    assert_eq!(<[i64; 8]>::from(simd.simd_eq_mask64x8(a, a)), [-1; 8]);
    assert_eq!(
        <[i64; 8]>::from(simd.simd_eq_mask64x8(a, b)),
        [0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn simd_eq_f64x2<S: Simd>(simd: S) {
    let a_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 2] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let expected: [i64; 2] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i64
        } else {
            0_i64
        }
    });
    let result = simd.simd_eq_f64x2(a, b);
    assert_eq!(<[i64; 2]>::from(result), expected);
}

#[simd_test]
fn simd_eq_f64x4<S: Simd>(simd: S) {
    let a_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 4] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x4::from_slice(simd, &a_values);
    let b = f64x4::from_slice(simd, &b_values);
    let expected: [i64; 4] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i64
        } else {
            0_i64
        }
    });
    let result = simd.simd_eq_f64x4(a, b);
    assert_eq!(<[i64; 4]>::from(result), expected);
}

#[simd_test]
fn simd_eq_mask8x16<S: Simd>(simd: S) {
    let a_values: [i8; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let b_values: [i8; 16] = core::array::from_fn(|i| if i % 3 == 0 { -1_i8 } else { 0_i8 });
    let a = mask8x16::from_slice(simd, &a_values);
    let b = mask8x16::from_slice(simd, &b_values);
    let expected: [i8; 16] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i8
        } else {
            0_i8
        }
    });
    let result = simd.simd_eq_mask8x16(a, b);
    assert_eq!(<[i8; 16]>::from(result), expected);
}

#[simd_test]
fn simd_eq_mask8x32<S: Simd>(simd: S) {
    let a_values: [i8; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let b_values: [i8; 32] = core::array::from_fn(|i| if i % 3 == 0 { -1_i8 } else { 0_i8 });
    let a = mask8x32::from_slice(simd, &a_values);
    let b = mask8x32::from_slice(simd, &b_values);
    let expected: [i8; 32] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i8
        } else {
            0_i8
        }
    });
    let result = simd.simd_eq_mask8x32(a, b);
    assert_eq!(<[i8; 32]>::from(result), expected);
}

#[simd_test]
fn simd_eq_mask8x64<S: Simd>(simd: S) {
    let a_values: [i8; 64] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let b_values: [i8; 64] = core::array::from_fn(|i| if i % 3 == 0 { -1_i8 } else { 0_i8 });
    let a = mask8x64::from_slice(simd, &a_values);
    let b = mask8x64::from_slice(simd, &b_values);
    let expected: [i8; 64] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i8
        } else {
            0_i8
        }
    });
    let result = simd.simd_eq_mask8x64(a, b);
    assert_eq!(<[i8; 64]>::from(result), expected);
}

#[simd_test]
fn simd_eq_mask16x8<S: Simd>(simd: S) {
    let a_values: [i16; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let b_values: [i16; 8] = core::array::from_fn(|i| if i % 3 == 0 { -1_i16 } else { 0_i16 });
    let a = mask16x8::from_slice(simd, &a_values);
    let b = mask16x8::from_slice(simd, &b_values);
    let expected: [i16; 8] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i16
        } else {
            0_i16
        }
    });
    let result = simd.simd_eq_mask16x8(a, b);
    assert_eq!(<[i16; 8]>::from(result), expected);
}

#[simd_test]
fn simd_eq_mask16x16<S: Simd>(simd: S) {
    let a_values: [i16; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let b_values: [i16; 16] = core::array::from_fn(|i| if i % 3 == 0 { -1_i16 } else { 0_i16 });
    let a = mask16x16::from_slice(simd, &a_values);
    let b = mask16x16::from_slice(simd, &b_values);
    let expected: [i16; 16] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i16
        } else {
            0_i16
        }
    });
    let result = simd.simd_eq_mask16x16(a, b);
    assert_eq!(<[i16; 16]>::from(result), expected);
}

#[simd_test]
fn simd_eq_mask16x32<S: Simd>(simd: S) {
    let a_values: [i16; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let b_values: [i16; 32] = core::array::from_fn(|i| if i % 3 == 0 { -1_i16 } else { 0_i16 });
    let a = mask16x32::from_slice(simd, &a_values);
    let b = mask16x32::from_slice(simd, &b_values);
    let expected: [i16; 32] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i16
        } else {
            0_i16
        }
    });
    let result = simd.simd_eq_mask16x32(a, b);
    assert_eq!(<[i16; 32]>::from(result), expected);
}

#[simd_test]
fn simd_eq_mask32x4<S: Simd>(simd: S) {
    let a_values: [i32; 4] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let b_values: [i32; 4] = core::array::from_fn(|i| if i % 3 == 0 { -1_i32 } else { 0_i32 });
    let a = mask32x4::from_slice(simd, &a_values);
    let b = mask32x4::from_slice(simd, &b_values);
    let expected: [i32; 4] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i32
        } else {
            0_i32
        }
    });
    let result = simd.simd_eq_mask32x4(a, b);
    assert_eq!(<[i32; 4]>::from(result), expected);
}

#[simd_test]
fn simd_eq_mask32x8<S: Simd>(simd: S) {
    let a_values: [i32; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let b_values: [i32; 8] = core::array::from_fn(|i| if i % 3 == 0 { -1_i32 } else { 0_i32 });
    let a = mask32x8::from_slice(simd, &a_values);
    let b = mask32x8::from_slice(simd, &b_values);
    let expected: [i32; 8] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i32
        } else {
            0_i32
        }
    });
    let result = simd.simd_eq_mask32x8(a, b);
    assert_eq!(<[i32; 8]>::from(result), expected);
}

#[simd_test]
fn simd_eq_mask32x16<S: Simd>(simd: S) {
    let a_values: [i32; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let b_values: [i32; 16] = core::array::from_fn(|i| if i % 3 == 0 { -1_i32 } else { 0_i32 });
    let a = mask32x16::from_slice(simd, &a_values);
    let b = mask32x16::from_slice(simd, &b_values);
    let expected: [i32; 16] = core::array::from_fn(|i| {
        if a_values[i] == b_values[i] {
            -1_i32
        } else {
            0_i32
        }
    });
    let result = simd.simd_eq_mask32x16(a, b);
    assert_eq!(<[i32; 16]>::from(result), expected);
}
