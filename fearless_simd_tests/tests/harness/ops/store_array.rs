// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn store_array_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let mut dest = [0.0_f32; 4];
    simd.store_array_f32x4(a, &mut dest);
    assert_eq!(dest, [1.0, 2.0, 3.0, 4.0]);
}

#[simd_test]
fn store_array_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let mut dest = [0.0_f32; 8];
    simd.store_array_f32x8(a, &mut dest);
    assert_eq!(dest, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[simd_test]
fn store_array_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.5, 2.5]);
    let mut dest = [0.0_f64; 2];
    simd.store_array_f64x2(a, &mut dest);
    assert_eq!(dest, [1.5, 2.5]);
}

#[simd_test]
fn store_array_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.5, 2.5, 3.5, 4.5]);
    let mut dest = [0.0_f64; 4];
    simd.store_array_f64x4(a, &mut dest);
    assert_eq!(dest, [1.5, 2.5, 3.5, 4.5]);
}

#[simd_test]
fn store_array_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let mut dest = [0_i8; 16];
    simd.store_array_i8x16(a, &mut dest);
    assert_eq!(
        dest,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
}

#[simd_test]
fn store_array_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut dest = [0_i16; 8];
    simd.store_array_i16x8(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn store_array_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let mut dest = [0_i32; 4];
    simd.store_array_i32x4(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4]);
}

#[simd_test]
fn store_array_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let mut dest = [0_u8; 16];
    simd.store_array_u8x16(a, &mut dest);
    assert_eq!(
        dest,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
}

#[simd_test]
fn store_array_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut dest = [0_u16; 8];
    simd.store_array_u16x8(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn store_array_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    let mut dest = [0_u32; 4];
    simd.store_array_u32x4(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4]);
}

#[simd_test]
fn store_array_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut dest = [0_i32; 8];
    simd.store_array_i32x8(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn store_array_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut dest = [0_u32; 8];
    simd.store_array_u32x8(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn store_array_f32x16<S: Simd>(simd: S) {
    let data: [f32; 16] = core::array::from_fn(|i| i as f32);
    let a = f32x16::from_slice(simd, &data);
    let mut dest = [0.0_f32; 16];
    simd.store_array_f32x16(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_f64x8<S: Simd>(simd: S) {
    let data: [f64; 8] = core::array::from_fn(|i| i as f64);
    let a = f64x8::from_slice(simd, &data);
    let mut dest = [0.0_f64; 8];
    simd.store_array_f64x8(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_i8x32<S: Simd>(simd: S) {
    let data: [i8; 32] = core::array::from_fn(|i| i8::try_from(i).unwrap());
    let a = i8x32::from_slice(simd, &data);
    let mut dest = [0_i8; 32];
    simd.store_array_i8x32(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_i8x64<S: Simd>(simd: S) {
    let data: [i8; 64] = core::array::from_fn(|i| i8::try_from(i).unwrap());
    let a = i8x64::from_slice(simd, &data);
    let mut dest = [0_i8; 64];
    simd.store_array_i8x64(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_i16x16<S: Simd>(simd: S) {
    let data: [i16; 16] = core::array::from_fn(|i| i16::try_from(i).unwrap());
    let a = i16x16::from_slice(simd, &data);
    let mut dest = [0_i16; 16];
    simd.store_array_i16x16(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_i16x32<S: Simd>(simd: S) {
    let data: [i16; 32] = core::array::from_fn(|i| i16::try_from(i).unwrap());
    let a = i16x32::from_slice(simd, &data);
    let mut dest = [0_i16; 32];
    simd.store_array_i16x32(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_i32x16<S: Simd>(simd: S) {
    let data: [i32; 16] = core::array::from_fn(|i| i32::try_from(i).unwrap());
    let a = i32x16::from_slice(simd, &data);
    let mut dest = [0_i32; 16];
    simd.store_array_i32x16(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_u8x32<S: Simd>(simd: S) {
    let data: [u8; 32] = core::array::from_fn(|i| u8::try_from(i).unwrap());
    let a = u8x32::from_slice(simd, &data);
    let mut dest = [0_u8; 32];
    simd.store_array_u8x32(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_u8x64<S: Simd>(simd: S) {
    let data: [u8; 64] = core::array::from_fn(|i| u8::try_from(i).unwrap());
    let a = u8x64::from_slice(simd, &data);
    let mut dest = [0_u8; 64];
    simd.store_array_u8x64(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_u16x16<S: Simd>(simd: S) {
    let data: [u16; 16] = core::array::from_fn(|i| u16::try_from(i).unwrap());
    let a = u16x16::from_slice(simd, &data);
    let mut dest = [0_u16; 16];
    simd.store_array_u16x16(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_u16x32<S: Simd>(simd: S) {
    let data: [u16; 32] = core::array::from_fn(|i| u16::try_from(i).unwrap());
    let a = u16x32::from_slice(simd, &data);
    let mut dest = [0_u16; 32];
    simd.store_array_u16x32(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_u32x16<S: Simd>(simd: S) {
    let data: [u32; 16] = core::array::from_fn(|i| u32::try_from(i).unwrap());
    let a = u32x16::from_slice(simd, &data);
    let mut dest = [0_u32; 16];
    simd.store_array_u32x16(a, &mut dest);
    assert_eq!(dest, data);
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn store_array_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let mut out = [0_i64; 2];
    simd.store_array_i64x2(a, &mut out);
    assert_eq!(out, [1_i64, -2_i64]);
}

#[simd_test]
fn store_array_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let mut out = [0_i64; 4];
    simd.store_array_i64x4(a, &mut out);
    assert_eq!(out, [1_i64, -2_i64, 3_i64, -4_i64]);
}

#[simd_test]
fn store_array_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let mut out = [0_i64; 8];
    simd.store_array_i64x8(a, &mut out);
    assert_eq!(
        out,
        [1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64]
    );
}

#[simd_test]
fn store_array_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let mut out = [0_u64; 2];
    simd.store_array_u64x2(a, &mut out);
    assert_eq!(out, [1_u64, 2_u64]);
}

#[simd_test]
fn store_array_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let mut out = [0_u64; 4];
    simd.store_array_u64x4(a, &mut out);
    assert_eq!(out, [1_u64, 2_u64, 3_u64, 4_u64]);
}

#[simd_test]
fn store_array_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let mut out = [0_u64; 8];
    simd.store_array_u64x8(a, &mut out);
    assert_eq!(
        out,
        [1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64]
    );
}
