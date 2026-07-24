// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// Additional concrete rows for this operation.

#[simd_test]
fn from_bitmask_mask64x2<S: Simd>(simd: S) {
    let a = simd.from_bitmask_mask64x2(3);
    assert_eq!(<[i64; 2]>::from(a), [-1_i64, -1_i64]);
}

#[simd_test]
fn from_bitmask_mask64x4<S: Simd>(simd: S) {
    let a = simd.from_bitmask_mask64x4(9);
    assert_eq!(<[i64; 4]>::from(a), [-1_i64, 0_i64, 0_i64, -1_i64]);
}

#[simd_test]
fn from_bitmask_mask64x8<S: Simd>(simd: S) {
    let a = simd.from_bitmask_mask64x8(129);
    assert_eq!(
        <[i64; 8]>::from(a),
        [-1_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, -1_i64]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn from_bitmask_mask8x16<S: Simd>(simd: S) {
    let result = simd.from_bitmask_mask8x16((1_u64 << 0) | (1_u64 << 15));
    let expected: [i8; 16] = core::array::from_fn(|i| if i == 0 || i == 15 { -1_i8 } else { 0_i8 });
    assert_eq!(<[i8; 16]>::from(result), expected);
}

#[simd_test]
fn from_bitmask_mask8x32<S: Simd>(simd: S) {
    let result = simd.from_bitmask_mask8x32((1_u64 << 0) | (1_u64 << 31));
    let expected: [i8; 32] = core::array::from_fn(|i| if i == 0 || i == 31 { -1_i8 } else { 0_i8 });
    assert_eq!(<[i8; 32]>::from(result), expected);
}

#[simd_test]
fn from_bitmask_mask8x64<S: Simd>(simd: S) {
    let result = simd.from_bitmask_mask8x64((1_u64 << 0) | (1_u64 << 63));
    let expected: [i8; 64] = core::array::from_fn(|i| if i == 0 || i == 63 { -1_i8 } else { 0_i8 });
    assert_eq!(<[i8; 64]>::from(result), expected);
}

#[simd_test]
fn from_bitmask_mask16x8<S: Simd>(simd: S) {
    let result = simd.from_bitmask_mask16x8((1_u64 << 0) | (1_u64 << 7));
    let expected: [i16; 8] =
        core::array::from_fn(|i| if i == 0 || i == 7 { -1_i16 } else { 0_i16 });
    assert_eq!(<[i16; 8]>::from(result), expected);
}

#[simd_test]
fn from_bitmask_mask16x16<S: Simd>(simd: S) {
    let result = simd.from_bitmask_mask16x16((1_u64 << 0) | (1_u64 << 15));
    let expected: [i16; 16] =
        core::array::from_fn(|i| if i == 0 || i == 15 { -1_i16 } else { 0_i16 });
    assert_eq!(<[i16; 16]>::from(result), expected);
}

#[simd_test]
fn from_bitmask_mask16x32<S: Simd>(simd: S) {
    let result = simd.from_bitmask_mask16x32((1_u64 << 0) | (1_u64 << 31));
    let expected: [i16; 32] =
        core::array::from_fn(|i| if i == 0 || i == 31 { -1_i16 } else { 0_i16 });
    assert_eq!(<[i16; 32]>::from(result), expected);
}

#[simd_test]
fn from_bitmask_mask32x4<S: Simd>(simd: S) {
    let result = simd.from_bitmask_mask32x4((1_u64 << 0) | (1_u64 << 3));
    let expected: [i32; 4] =
        core::array::from_fn(|i| if i == 0 || i == 3 { -1_i32 } else { 0_i32 });
    assert_eq!(<[i32; 4]>::from(result), expected);
}

#[simd_test]
fn from_bitmask_mask32x8<S: Simd>(simd: S) {
    let result = simd.from_bitmask_mask32x8((1_u64 << 0) | (1_u64 << 7));
    let expected: [i32; 8] =
        core::array::from_fn(|i| if i == 0 || i == 7 { -1_i32 } else { 0_i32 });
    assert_eq!(<[i32; 8]>::from(result), expected);
}

#[simd_test]
fn from_bitmask_mask32x16<S: Simd>(simd: S) {
    let result = simd.from_bitmask_mask32x16((1_u64 << 0) | (1_u64 << 15));
    let expected: [i32; 16] =
        core::array::from_fn(|i| if i == 0 || i == 15 { -1_i32 } else { 0_i32 });
    assert_eq!(<[i32; 16]>::from(result), expected);
}
