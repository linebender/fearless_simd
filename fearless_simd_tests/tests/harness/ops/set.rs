// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// Additional concrete rows for this operation.

#[simd_test]
fn set_mask64x2<S: Simd>(simd: S) {
    let mut a = simd.splat_mask64x2(false);
    simd.set_mask64x2(&mut a, 0, true);
    simd.set_mask64x2(&mut a, 1, true);
    assert_eq!(simd.to_bitmask_mask64x2(a), 3);
}

#[simd_test]
fn set_mask64x4<S: Simd>(simd: S) {
    let mut a = simd.splat_mask64x4(false);
    simd.set_mask64x4(&mut a, 0, true);
    simd.set_mask64x4(&mut a, 3, true);
    assert_eq!(simd.to_bitmask_mask64x4(a), 9);
}

#[simd_test]
fn set_mask64x8<S: Simd>(simd: S) {
    let mut a = simd.splat_mask64x8(false);
    simd.set_mask64x8(&mut a, 0, true);
    simd.set_mask64x8(&mut a, 7, true);
    assert_eq!(simd.to_bitmask_mask64x8(a), 129);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn set_mask8x16<S: Simd>(simd: S) {
    let mut mask = mask8x16::splat(simd, false);
    simd.set_mask8x16(&mut mask, 0, true);
    simd.set_mask8x16(&mut mask, 15, true);
    let expected: [i8; 16] = core::array::from_fn(|i| if i == 0 || i == 15 { -1_i8 } else { 0_i8 });
    assert_eq!(<[i8; 16]>::from(mask), expected);
}

#[simd_test]
fn set_mask8x32<S: Simd>(simd: S) {
    let mut mask = mask8x32::splat(simd, false);
    simd.set_mask8x32(&mut mask, 0, true);
    simd.set_mask8x32(&mut mask, 31, true);
    let expected: [i8; 32] = core::array::from_fn(|i| if i == 0 || i == 31 { -1_i8 } else { 0_i8 });
    assert_eq!(<[i8; 32]>::from(mask), expected);
}

#[simd_test]
fn set_mask8x64<S: Simd>(simd: S) {
    let mut mask = mask8x64::splat(simd, false);
    simd.set_mask8x64(&mut mask, 0, true);
    simd.set_mask8x64(&mut mask, 63, true);
    let expected: [i8; 64] = core::array::from_fn(|i| if i == 0 || i == 63 { -1_i8 } else { 0_i8 });
    assert_eq!(<[i8; 64]>::from(mask), expected);
}

#[simd_test]
fn set_mask16x8<S: Simd>(simd: S) {
    let mut mask = mask16x8::splat(simd, false);
    simd.set_mask16x8(&mut mask, 0, true);
    simd.set_mask16x8(&mut mask, 7, true);
    let expected: [i16; 8] =
        core::array::from_fn(|i| if i == 0 || i == 7 { -1_i16 } else { 0_i16 });
    assert_eq!(<[i16; 8]>::from(mask), expected);
}

#[simd_test]
fn set_mask16x16<S: Simd>(simd: S) {
    let mut mask = mask16x16::splat(simd, false);
    simd.set_mask16x16(&mut mask, 0, true);
    simd.set_mask16x16(&mut mask, 15, true);
    let expected: [i16; 16] =
        core::array::from_fn(|i| if i == 0 || i == 15 { -1_i16 } else { 0_i16 });
    assert_eq!(<[i16; 16]>::from(mask), expected);
}

#[simd_test]
fn set_mask16x32<S: Simd>(simd: S) {
    let mut mask = mask16x32::splat(simd, false);
    simd.set_mask16x32(&mut mask, 0, true);
    simd.set_mask16x32(&mut mask, 31, true);
    let expected: [i16; 32] =
        core::array::from_fn(|i| if i == 0 || i == 31 { -1_i16 } else { 0_i16 });
    assert_eq!(<[i16; 32]>::from(mask), expected);
}

#[simd_test]
fn set_mask32x4<S: Simd>(simd: S) {
    let mut mask = mask32x4::splat(simd, false);
    simd.set_mask32x4(&mut mask, 0, true);
    simd.set_mask32x4(&mut mask, 3, true);
    let expected: [i32; 4] =
        core::array::from_fn(|i| if i == 0 || i == 3 { -1_i32 } else { 0_i32 });
    assert_eq!(<[i32; 4]>::from(mask), expected);
}

#[simd_test]
fn set_mask32x8<S: Simd>(simd: S) {
    let mut mask = mask32x8::splat(simd, false);
    simd.set_mask32x8(&mut mask, 0, true);
    simd.set_mask32x8(&mut mask, 7, true);
    let expected: [i32; 8] =
        core::array::from_fn(|i| if i == 0 || i == 7 { -1_i32 } else { 0_i32 });
    assert_eq!(<[i32; 8]>::from(mask), expected);
}

#[simd_test]
fn set_mask32x16<S: Simd>(simd: S) {
    let mut mask = mask32x16::splat(simd, false);
    simd.set_mask32x16(&mut mask, 0, true);
    simd.set_mask32x16(&mut mask, 15, true);
    let expected: [i32; 16] =
        core::array::from_fn(|i| if i == 0 || i == 15 { -1_i32 } else { 0_i32 });
    assert_eq!(<[i32; 16]>::from(mask), expected);
}
