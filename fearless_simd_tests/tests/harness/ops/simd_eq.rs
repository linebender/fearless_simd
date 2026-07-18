// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

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

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

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
