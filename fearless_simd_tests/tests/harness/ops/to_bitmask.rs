// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// Additional concrete rows for this operation.

#[simd_test]
fn to_bitmask_mask64x2<S: Simd>(simd: S) {
    let a = mask64x2::from_slice(simd, &[-1_i64, -1_i64]);
    assert_eq!(simd.to_bitmask_mask64x2(a), 3);
}

#[simd_test]
fn to_bitmask_mask64x4<S: Simd>(simd: S) {
    let a = mask64x4::from_slice(simd, &[-1_i64, 0_i64, 0_i64, -1_i64]);
    assert_eq!(simd.to_bitmask_mask64x4(a), 9);
}

#[simd_test]
fn to_bitmask_mask64x8<S: Simd>(simd: S) {
    let a = mask64x8::from_slice(
        simd,
        &[-1_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, -1_i64],
    );
    assert_eq!(simd.to_bitmask_mask64x8(a), 129);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn to_bitmask_mask8x16<S: Simd>(simd: S) {
    let values: [i8; 16] = core::array::from_fn(|i| if i == 0 || i == 15 { -1_i8 } else { 0_i8 });
    let mask = mask8x16::from_slice(simd, &values);
    assert_eq!(simd.to_bitmask_mask8x16(mask), (1_u64 << 0) | (1_u64 << 15));
}

#[simd_test]
fn to_bitmask_mask8x32<S: Simd>(simd: S) {
    let values: [i8; 32] = core::array::from_fn(|i| if i == 0 || i == 31 { -1_i8 } else { 0_i8 });
    let mask = mask8x32::from_slice(simd, &values);
    assert_eq!(simd.to_bitmask_mask8x32(mask), (1_u64 << 0) | (1_u64 << 31));
}

#[simd_test]
fn to_bitmask_mask8x64<S: Simd>(simd: S) {
    let values: [i8; 64] = core::array::from_fn(|i| if i == 0 || i == 63 { -1_i8 } else { 0_i8 });
    let mask = mask8x64::from_slice(simd, &values);
    assert_eq!(simd.to_bitmask_mask8x64(mask), (1_u64 << 0) | (1_u64 << 63));
}

#[simd_test]
fn to_bitmask_mask16x8<S: Simd>(simd: S) {
    let values: [i16; 8] = core::array::from_fn(|i| if i == 0 || i == 7 { -1_i16 } else { 0_i16 });
    let mask = mask16x8::from_slice(simd, &values);
    assert_eq!(simd.to_bitmask_mask16x8(mask), (1_u64 << 0) | (1_u64 << 7));
}

#[simd_test]
fn to_bitmask_mask16x16<S: Simd>(simd: S) {
    let values: [i16; 16] =
        core::array::from_fn(|i| if i == 0 || i == 15 { -1_i16 } else { 0_i16 });
    let mask = mask16x16::from_slice(simd, &values);
    assert_eq!(
        simd.to_bitmask_mask16x16(mask),
        (1_u64 << 0) | (1_u64 << 15)
    );
}

#[simd_test]
fn to_bitmask_mask16x32<S: Simd>(simd: S) {
    let values: [i16; 32] =
        core::array::from_fn(|i| if i == 0 || i == 31 { -1_i16 } else { 0_i16 });
    let mask = mask16x32::from_slice(simd, &values);
    assert_eq!(
        simd.to_bitmask_mask16x32(mask),
        (1_u64 << 0) | (1_u64 << 31)
    );
}

#[simd_test]
fn to_bitmask_mask32x4<S: Simd>(simd: S) {
    let values: [i32; 4] = core::array::from_fn(|i| if i == 0 || i == 3 { -1_i32 } else { 0_i32 });
    let mask = mask32x4::from_slice(simd, &values);
    assert_eq!(simd.to_bitmask_mask32x4(mask), (1_u64 << 0) | (1_u64 << 3));
}

#[simd_test]
fn to_bitmask_mask32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = core::array::from_fn(|i| if i == 0 || i == 7 { -1_i32 } else { 0_i32 });
    let mask = mask32x8::from_slice(simd, &values);
    assert_eq!(simd.to_bitmask_mask32x8(mask), (1_u64 << 0) | (1_u64 << 7));
}

#[simd_test]
fn to_bitmask_mask32x16<S: Simd>(simd: S) {
    let values: [i32; 16] =
        core::array::from_fn(|i| if i == 0 || i == 15 { -1_i32 } else { 0_i32 });
    let mask = mask32x16::from_slice(simd, &values);
    assert_eq!(
        simd.to_bitmask_mask32x16(mask),
        (1_u64 << 0) | (1_u64 << 15)
    );
}
