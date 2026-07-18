// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn shl_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0xFFFFFFFF, 0xFFFF, 0xFF, 0]);
    assert_eq!(*(a << 4), [0xFFFFFFF0, 0xFFFF0, 0xFF0, 0]);
}

#[simd_test]
fn shl_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ],
    );
    assert_eq!(
        *(a << 2),
        [
            4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 4, 8, 12, 16, 20, 24, 28,
            32, 36, 40, 44, 48, 52, 56, 60, 64, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52,
            56, 60, 64, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64
        ]
    );
}

#[simd_test]
fn shl_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ],
    );
    assert_eq!(
        *(a << 2),
        [
            4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 4, 8, 12, 16, 20, 24, 28,
            32, 36, 40, 44, 48, 52, 56, 60, 64, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52,
            56, 60, 64, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64
        ]
    );
}

#[simd_test]
fn shl_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8,
        ],
    );
    assert_eq!(
        *(a << 4),
        [
            16, 32, 48, 64, 80, 96, 112, 128, 16, 32, 48, 64, 80, 96, 112, 128, 16, 32, 48, 64, 80,
            96, 112, 128, 16, 32, 48, 64, 80, 96, 112, 128
        ]
    );
}

#[simd_test]
fn shl_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8,
        ],
    );
    assert_eq!(
        *(a << 4),
        [
            16, 32, 48, 64, 80, 96, 112, 128, 16, 32, 48, 64, 80, 96, 112, 128, 16, 32, 48, 64, 80,
            96, 112, 128, 16, 32, 48, 64, 80, 96, 112, 128
        ]
    );
}

#[simd_test]
fn shl_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        *(a << 4),
        [
            16, 32, 48, 64, 80, 96, 112, 128, 16, 32, 48, 64, 80, 96, 112, 128
        ]
    );
}

#[simd_test]
fn shl_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            0xFFFFFFFF, 0xFFFF, 0xFF, 0, 0xFFFFFFFF, 0xFFFF, 0xFF, 0, 0xFFFFFFFF, 0xFFFF, 0xFF, 0,
            0xFFFFFFFF, 0xFFFF, 0xFF, 0,
        ],
    );
    assert_eq!(
        *(a << 4),
        [
            0xFFFFFFF0, 0xFFFF0, 0xFF0, 0, 0xFFFFFFF0, 0xFFFF0, 0xFF0, 0, 0xFFFFFFF0, 0xFFFF0,
            0xFF0, 0, 0xFFFFFFF0, 0xFFFF0, 0xFF0, 0
        ]
    );
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn shl_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    assert_eq!(*simd.shl_i64x2(a, 1), [2_i64, -4_i64]);
}

#[simd_test]
fn shl_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    assert_eq!(*simd.shl_i64x4(a, 1), [2_i64, -4_i64, 6_i64, -8_i64]);
}

#[simd_test]
fn shl_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    assert_eq!(
        *simd.shl_i64x8(a, 1),
        [
            2_i64, -4_i64, 6_i64, -8_i64, 10_i64, -12_i64, 14_i64, -16_i64
        ]
    );
}

#[simd_test]
fn shl_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    assert_eq!(*simd.shl_u64x2(a, 1), [2_u64, 4_u64]);
}

#[simd_test]
fn shl_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    assert_eq!(*simd.shl_u64x4(a, 1), [2_u64, 4_u64, 6_u64, 8_u64]);
}

#[simd_test]
fn shl_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    assert_eq!(
        *simd.shl_u64x8(a, 1),
        [2_u64, 4_u64, 6_u64, 8_u64, 10_u64, 12_u64, 14_u64, 16_u64]
    );
}
