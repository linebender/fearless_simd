// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// Additional concrete rows for this operation.

#[simd_test]
fn as_array_ref_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    assert_eq!(simd.as_array_ref_i64x2(&a), &[1_i64, -2_i64]);
}

#[simd_test]
fn as_array_ref_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    assert_eq!(simd.as_array_ref_i64x4(&a), &[1_i64, -2_i64, 3_i64, -4_i64]);
}

#[simd_test]
fn as_array_ref_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    assert_eq!(
        simd.as_array_ref_i64x8(&a),
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64]
    );
}

#[simd_test]
fn as_array_ref_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    assert_eq!(simd.as_array_ref_u64x2(&a), &[1_u64, 2_u64]);
}

#[simd_test]
fn as_array_ref_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    assert_eq!(simd.as_array_ref_u64x4(&a), &[1_u64, 2_u64, 3_u64, 4_u64]);
}

#[simd_test]
fn as_array_ref_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    assert_eq!(
        simd.as_array_ref_u64x8(&a),
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn as_array_ref_i8x16<S: Simd>(simd: S) {
    let values: [i8; 16] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let a = i8x16::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_i8x16(&a), &values);
}

#[simd_test]
fn as_array_ref_u8x16<S: Simd>(simd: S) {
    let values: [u8; 16] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let a = u8x16::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_u8x16(&a), &values);
}

#[simd_test]
fn as_array_ref_i8x32<S: Simd>(simd: S) {
    let values: [i8; 32] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let a = i8x32::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_i8x32(&a), &values);
}

#[simd_test]
fn as_array_ref_u8x32<S: Simd>(simd: S) {
    let values: [u8; 32] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let a = u8x32::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_u8x32(&a), &values);
}

#[simd_test]
fn as_array_ref_i8x64<S: Simd>(simd: S) {
    let values: [i8; 64] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let a = i8x64::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_i8x64(&a), &values);
}

#[simd_test]
fn as_array_ref_u8x64<S: Simd>(simd: S) {
    let values: [u8; 64] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let a = u8x64::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_u8x64(&a), &values);
}

#[simd_test]
fn as_array_ref_i16x8<S: Simd>(simd: S) {
    let values: [i16; 8] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let a = i16x8::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_i16x8(&a), &values);
}

#[simd_test]
fn as_array_ref_u16x8<S: Simd>(simd: S) {
    let values: [u16; 8] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let a = u16x8::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_u16x8(&a), &values);
}

#[simd_test]
fn as_array_ref_i16x16<S: Simd>(simd: S) {
    let values: [i16; 16] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let a = i16x16::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_i16x16(&a), &values);
}

#[simd_test]
fn as_array_ref_u16x16<S: Simd>(simd: S) {
    let values: [u16; 16] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let a = u16x16::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_u16x16(&a), &values);
}

#[simd_test]
fn as_array_ref_i16x32<S: Simd>(simd: S) {
    let values: [i16; 32] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let a = i16x32::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_i16x32(&a), &values);
}

#[simd_test]
fn as_array_ref_u16x32<S: Simd>(simd: S) {
    let values: [u16; 32] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let a = u16x32::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_u16x32(&a), &values);
}

#[simd_test]
fn as_array_ref_f32x4<S: Simd>(simd: S) {
    let values: [f32; 4] = core::array::from_fn(|i| i as f32 + 1.25_f32);
    let a = f32x4::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_f32x4(&a), &values);
}

#[simd_test]
fn as_array_ref_i32x4<S: Simd>(simd: S) {
    let values: [i32; 4] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let a = i32x4::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_i32x4(&a), &values);
}

#[simd_test]
fn as_array_ref_u32x4<S: Simd>(simd: S) {
    let values: [u32; 4] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let a = u32x4::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_u32x4(&a), &values);
}

#[simd_test]
fn as_array_ref_f32x8<S: Simd>(simd: S) {
    let values: [f32; 8] = core::array::from_fn(|i| i as f32 + 1.25_f32);
    let a = f32x8::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_f32x8(&a), &values);
}

#[simd_test]
fn as_array_ref_i32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let a = i32x8::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_i32x8(&a), &values);
}

#[simd_test]
fn as_array_ref_u32x8<S: Simd>(simd: S) {
    let values: [u32; 8] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let a = u32x8::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_u32x8(&a), &values);
}

#[simd_test]
fn as_array_ref_f32x16<S: Simd>(simd: S) {
    let values: [f32; 16] = core::array::from_fn(|i| i as f32 + 1.25_f32);
    let a = f32x16::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_f32x16(&a), &values);
}

#[simd_test]
fn as_array_ref_i32x16<S: Simd>(simd: S) {
    let values: [i32; 16] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let a = i32x16::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_i32x16(&a), &values);
}

#[simd_test]
fn as_array_ref_u32x16<S: Simd>(simd: S) {
    let values: [u32; 16] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let a = u32x16::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_u32x16(&a), &values);
}

#[simd_test]
fn as_array_ref_f64x2<S: Simd>(simd: S) {
    let values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let a = f64x2::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_f64x2(&a), &values);
}

#[simd_test]
fn as_array_ref_f64x4<S: Simd>(simd: S) {
    let values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let a = f64x4::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_f64x4(&a), &values);
}

#[simd_test]
fn as_array_ref_f64x8<S: Simd>(simd: S) {
    let values: [f64; 8] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let a = f64x8::from_slice(simd, &values);
    assert_eq!(simd.as_array_ref_f64x8(&a), &values);
}
