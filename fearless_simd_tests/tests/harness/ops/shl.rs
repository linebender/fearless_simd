// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

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

// Additional concrete rows for this operation.

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

// Generated gap-fill coverage rows.

#[simd_test]
fn shl_i8x16<S: Simd>(simd: S) {
    let values: [i8; 16] = core::array::from_fn(|i| (i % 31) as i8 + 1_i8);
    let a = i8x16::from_slice(simd, &values);
    let expected: [i8; 16] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_i8x16(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shl_u8x16<S: Simd>(simd: S) {
    let values: [u8; 16] = core::array::from_fn(|i| (i % 31) as u8 + 1_u8);
    let a = u8x16::from_slice(simd, &values);
    let expected: [u8; 16] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_u8x16(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shl_i8x32<S: Simd>(simd: S) {
    let values: [i8; 32] = core::array::from_fn(|i| (i % 31) as i8 + 1_i8);
    let a = i8x32::from_slice(simd, &values);
    let expected: [i8; 32] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_i8x32(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shl_u8x32<S: Simd>(simd: S) {
    let values: [u8; 32] = core::array::from_fn(|i| (i % 31) as u8 + 1_u8);
    let a = u8x32::from_slice(simd, &values);
    let expected: [u8; 32] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_u8x32(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shl_i16x8<S: Simd>(simd: S) {
    let values: [i16; 8] = core::array::from_fn(|i| (i % 31) as i16 + 1_i16);
    let a = i16x8::from_slice(simd, &values);
    let expected: [i16; 8] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_i16x8(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shl_u16x8<S: Simd>(simd: S) {
    let values: [u16; 8] = core::array::from_fn(|i| (i % 31) as u16 + 1_u16);
    let a = u16x8::from_slice(simd, &values);
    let expected: [u16; 8] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_u16x8(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shl_i16x16<S: Simd>(simd: S) {
    let values: [i16; 16] = core::array::from_fn(|i| (i % 31) as i16 + 1_i16);
    let a = i16x16::from_slice(simd, &values);
    let expected: [i16; 16] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_i16x16(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shl_u16x16<S: Simd>(simd: S) {
    let values: [u16; 16] = core::array::from_fn(|i| (i % 31) as u16 + 1_u16);
    let a = u16x16::from_slice(simd, &values);
    let expected: [u16; 16] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_u16x16(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shl_i32x4<S: Simd>(simd: S) {
    let values: [i32; 4] = core::array::from_fn(|i| (i % 31) as i32 + 1_i32);
    let a = i32x4::from_slice(simd, &values);
    let expected: [i32; 4] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_i32x4(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shl_i32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = core::array::from_fn(|i| (i % 31) as i32 + 1_i32);
    let a = i32x8::from_slice(simd, &values);
    let expected: [i32; 8] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_i32x8(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shl_u32x8<S: Simd>(simd: S) {
    let values: [u32; 8] = core::array::from_fn(|i| (i % 31) as u32 + 1_u32);
    let a = u32x8::from_slice(simd, &values);
    let expected: [u32; 8] = core::array::from_fn(|i| values[i] << 1);
    let result = simd.shl_u32x8(a, 1);
    assert_eq!(result.as_slice(), expected.as_slice());
}
