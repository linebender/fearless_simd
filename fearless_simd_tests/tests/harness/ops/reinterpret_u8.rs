// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn reinterpret_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1, -2]);
    let bytes: u8x16<S> = a.bitcast();
    let words: u32x4<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_i64x2(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_i64x2(a).as_slice(), words.as_slice());
}

#[simd_test]
fn reinterpret_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1, -2, 3, -4]);
    let bytes: u8x32<S> = a.bitcast();
    let words: u32x8<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_i64x4(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_i64x4(a).as_slice(), words.as_slice());
}

#[simd_test]
fn reinterpret_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let bytes: u8x64<S> = a.bitcast();
    let words: u32x16<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_i64x8(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_i64x8(a).as_slice(), words.as_slice());
}

#[simd_test]
fn reinterpret_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1, u64::MAX - 1]);
    let bytes: u8x16<S> = a.bitcast();
    let words: u32x4<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_u64x2(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_u64x2(a).as_slice(), words.as_slice());
}

#[simd_test]
fn reinterpret_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1, u64::MAX - 1, 3, u64::MAX - 3]);
    let bytes: u8x32<S> = a.bitcast();
    let words: u32x8<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_u64x4(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_u64x4(a).as_slice(), words.as_slice());
}

#[simd_test]
fn reinterpret_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[1, u64::MAX - 1, 3, u64::MAX - 3, 5, 6, 7, 8]);
    let bytes: u8x64<S> = a.bitcast();
    let words: u32x16<S> = a.bitcast();
    assert_eq!(simd.reinterpret_u8_u64x8(a).as_slice(), bytes.as_slice());
    assert_eq!(simd.reinterpret_u32_u64x8(a).as_slice(), words.as_slice());
}

// Generated gap-fill coverage rows.

#[simd_test]
fn reinterpret_u8_i8x16<S: Simd>(simd: S) {
    let values: [i8; 16] = core::array::from_fn(|i| (i % 31) as i8 + 1_i8);
    let a = i8x16::from_slice(simd, &values);
    let result = simd.reinterpret_u8_i8x16(a);
    let roundtrip = i8x16::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_i8x32<S: Simd>(simd: S) {
    let values: [i8; 32] = core::array::from_fn(|i| (i % 31) as i8 + 1_i8);
    let a = i8x32::from_slice(simd, &values);
    let result = simd.reinterpret_u8_i8x32(a);
    let roundtrip = i8x32::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_i8x64<S: Simd>(simd: S) {
    let values: [i8; 64] = core::array::from_fn(|i| (i % 31) as i8 + 1_i8);
    let a = i8x64::from_slice(simd, &values);
    let result = simd.reinterpret_u8_i8x64(a);
    let roundtrip = i8x64::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_i16x8<S: Simd>(simd: S) {
    let values: [i16; 8] = core::array::from_fn(|i| (i % 31) as i16 + 1_i16);
    let a = i16x8::from_slice(simd, &values);
    let result = simd.reinterpret_u8_i16x8(a);
    let roundtrip = i16x8::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_u16x8<S: Simd>(simd: S) {
    let values: [u16; 8] = core::array::from_fn(|i| (i % 31) as u16 + 1_u16);
    let a = u16x8::from_slice(simd, &values);
    let result = simd.reinterpret_u8_u16x8(a);
    let roundtrip = u16x8::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_i16x16<S: Simd>(simd: S) {
    let values: [i16; 16] = core::array::from_fn(|i| (i % 31) as i16 + 1_i16);
    let a = i16x16::from_slice(simd, &values);
    let result = simd.reinterpret_u8_i16x16(a);
    let roundtrip = i16x16::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_u16x16<S: Simd>(simd: S) {
    let values: [u16; 16] = core::array::from_fn(|i| (i % 31) as u16 + 1_u16);
    let a = u16x16::from_slice(simd, &values);
    let result = simd.reinterpret_u8_u16x16(a);
    let roundtrip = u16x16::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_i16x32<S: Simd>(simd: S) {
    let values: [i16; 32] = core::array::from_fn(|i| (i % 31) as i16 + 1_i16);
    let a = i16x32::from_slice(simd, &values);
    let result = simd.reinterpret_u8_i16x32(a);
    let roundtrip = i16x32::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_u16x32<S: Simd>(simd: S) {
    let values: [u16; 32] = core::array::from_fn(|i| (i % 31) as u16 + 1_u16);
    let a = u16x32::from_slice(simd, &values);
    let result = simd.reinterpret_u8_u16x32(a);
    let roundtrip = u16x32::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_f32x4<S: Simd>(simd: S) {
    let values: [f32; 4] = core::array::from_fn(|i| i as f32 + 1.0_f32);
    let a = f32x4::from_slice(simd, &values);
    let result = simd.reinterpret_u8_f32x4(a);
    let roundtrip = f32x4::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_i32x4<S: Simd>(simd: S) {
    let values: [i32; 4] = core::array::from_fn(|i| (i % 31) as i32 + 1_i32);
    let a = i32x4::from_slice(simd, &values);
    let result = simd.reinterpret_u8_i32x4(a);
    let roundtrip = i32x4::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_u32x4<S: Simd>(simd: S) {
    let values: [u32; 4] = core::array::from_fn(|i| (i % 31) as u32 + 1_u32);
    let a = u32x4::from_slice(simd, &values);
    let result = simd.reinterpret_u8_u32x4(a);
    let roundtrip = u32x4::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_f32x8<S: Simd>(simd: S) {
    let values: [f32; 8] = core::array::from_fn(|i| i as f32 + 1.0_f32);
    let a = f32x8::from_slice(simd, &values);
    let result = simd.reinterpret_u8_f32x8(a);
    let roundtrip = f32x8::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_i32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = core::array::from_fn(|i| (i % 31) as i32 + 1_i32);
    let a = i32x8::from_slice(simd, &values);
    let result = simd.reinterpret_u8_i32x8(a);
    let roundtrip = i32x8::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_u32x8<S: Simd>(simd: S) {
    let values: [u32; 8] = core::array::from_fn(|i| (i % 31) as u32 + 1_u32);
    let a = u32x8::from_slice(simd, &values);
    let result = simd.reinterpret_u8_u32x8(a);
    let roundtrip = u32x8::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_f32x16<S: Simd>(simd: S) {
    let values: [f32; 16] = core::array::from_fn(|i| i as f32 + 1.0_f32);
    let a = f32x16::from_slice(simd, &values);
    let result = simd.reinterpret_u8_f32x16(a);
    let roundtrip = f32x16::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_i32x16<S: Simd>(simd: S) {
    let values: [i32; 16] = core::array::from_fn(|i| (i % 31) as i32 + 1_i32);
    let a = i32x16::from_slice(simd, &values);
    let result = simd.reinterpret_u8_i32x16(a);
    let roundtrip = i32x16::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_u8_u32x16<S: Simd>(simd: S) {
    let values: [u32; 16] = core::array::from_fn(|i| (i % 31) as u32 + 1_u32);
    let a = u32x16::from_slice(simd, &values);
    let result = simd.reinterpret_u8_u32x16(a);
    let roundtrip = u32x16::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}
