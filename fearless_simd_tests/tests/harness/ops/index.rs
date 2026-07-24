// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn index_consistency<S: Simd>(simd: S) {
    // We'll call index methods by name to avoid confusing clippy.
    use std::ops::{Index, IndexMut};

    let mut v = u32x4::from_slice(simd, &[0, 1, 2, 3]);
    for i in 0..4 {
        assert_eq!(i, *v.index(i) as usize);
        assert_eq!(i, *v.index_mut(i) as usize);
    }
}

// Generated gap-fill coverage rows.

#[simd_test]
fn index_i8x16<S: Simd>(simd: S) {
    let values: [i8; 16] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let mut result = i8x16::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i8;
    assert_eq!(result[0], 42_i8);
}

#[simd_test]
fn index_u8x16<S: Simd>(simd: S) {
    let values: [u8; 16] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let mut result = u8x16::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u8;
    assert_eq!(result[0], 42_u8);
}

#[simd_test]
fn index_i8x32<S: Simd>(simd: S) {
    let values: [i8; 32] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let mut result = i8x32::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i8;
    assert_eq!(result[0], 42_i8);
}

#[simd_test]
fn index_u8x32<S: Simd>(simd: S) {
    let values: [u8; 32] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let mut result = u8x32::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u8;
    assert_eq!(result[0], 42_u8);
}

#[simd_test]
fn index_i8x64<S: Simd>(simd: S) {
    let values: [i8; 64] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let mut result = i8x64::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i8;
    assert_eq!(result[0], 42_i8);
}

#[simd_test]
fn index_u8x64<S: Simd>(simd: S) {
    let values: [u8; 64] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let mut result = u8x64::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u8;
    assert_eq!(result[0], 42_u8);
}

#[simd_test]
fn index_i16x8<S: Simd>(simd: S) {
    let values: [i16; 8] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let mut result = i16x8::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i16;
    assert_eq!(result[0], 42_i16);
}

#[simd_test]
fn index_u16x8<S: Simd>(simd: S) {
    let values: [u16; 8] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let mut result = u16x8::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u16;
    assert_eq!(result[0], 42_u16);
}

#[simd_test]
fn index_i16x16<S: Simd>(simd: S) {
    let values: [i16; 16] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let mut result = i16x16::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i16;
    assert_eq!(result[0], 42_i16);
}

#[simd_test]
fn index_u16x16<S: Simd>(simd: S) {
    let values: [u16; 16] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let mut result = u16x16::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u16;
    assert_eq!(result[0], 42_u16);
}

#[simd_test]
fn index_i16x32<S: Simd>(simd: S) {
    let values: [i16; 32] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let mut result = i16x32::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i16;
    assert_eq!(result[0], 42_i16);
}

#[simd_test]
fn index_u16x32<S: Simd>(simd: S) {
    let values: [u16; 32] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let mut result = u16x32::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u16;
    assert_eq!(result[0], 42_u16);
}

#[simd_test]
fn index_f32x4<S: Simd>(simd: S) {
    let values: [f32; 4] = core::array::from_fn(|i| i as f32 + 1.25_f32);
    let mut result = f32x4::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42.5_f32;
    assert_eq!(result[0], 42.5_f32);
}

#[simd_test]
fn index_i32x4<S: Simd>(simd: S) {
    let values: [i32; 4] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let mut result = i32x4::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i32;
    assert_eq!(result[0], 42_i32);
}

#[simd_test]
fn index_u32x4<S: Simd>(simd: S) {
    let values: [u32; 4] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let mut result = u32x4::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u32;
    assert_eq!(result[0], 42_u32);
}

#[simd_test]
fn index_f32x8<S: Simd>(simd: S) {
    let values: [f32; 8] = core::array::from_fn(|i| i as f32 + 1.25_f32);
    let mut result = f32x8::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42.5_f32;
    assert_eq!(result[0], 42.5_f32);
}

#[simd_test]
fn index_i32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let mut result = i32x8::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i32;
    assert_eq!(result[0], 42_i32);
}

#[simd_test]
fn index_u32x8<S: Simd>(simd: S) {
    let values: [u32; 8] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let mut result = u32x8::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u32;
    assert_eq!(result[0], 42_u32);
}

#[simd_test]
fn index_f32x16<S: Simd>(simd: S) {
    let values: [f32; 16] = core::array::from_fn(|i| i as f32 + 1.25_f32);
    let mut result = f32x16::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42.5_f32;
    assert_eq!(result[0], 42.5_f32);
}

#[simd_test]
fn index_i32x16<S: Simd>(simd: S) {
    let values: [i32; 16] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let mut result = i32x16::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i32;
    assert_eq!(result[0], 42_i32);
}

#[simd_test]
fn index_u32x16<S: Simd>(simd: S) {
    let values: [u32; 16] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let mut result = u32x16::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u32;
    assert_eq!(result[0], 42_u32);
}

#[simd_test]
fn index_f64x2<S: Simd>(simd: S) {
    let values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let mut result = f64x2::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42.5_f64;
    assert_eq!(result[0], 42.5_f64);
}

#[simd_test]
fn index_i64x2<S: Simd>(simd: S) {
    let values: [i64; 2] = core::array::from_fn(|i| (i % 23) as i64 + 10_i64);
    let mut result = i64x2::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i64;
    assert_eq!(result[0], 42_i64);
}

#[simd_test]
fn index_u64x2<S: Simd>(simd: S) {
    let values: [u64; 2] = core::array::from_fn(|i| (i % 23) as u64 + 10_u64);
    let mut result = u64x2::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u64;
    assert_eq!(result[0], 42_u64);
}

#[simd_test]
fn index_f64x4<S: Simd>(simd: S) {
    let values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let mut result = f64x4::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42.5_f64;
    assert_eq!(result[0], 42.5_f64);
}

#[simd_test]
fn index_i64x4<S: Simd>(simd: S) {
    let values: [i64; 4] = core::array::from_fn(|i| (i % 23) as i64 + 10_i64);
    let mut result = i64x4::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i64;
    assert_eq!(result[0], 42_i64);
}

#[simd_test]
fn index_u64x4<S: Simd>(simd: S) {
    let values: [u64; 4] = core::array::from_fn(|i| (i % 23) as u64 + 10_u64);
    let mut result = u64x4::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u64;
    assert_eq!(result[0], 42_u64);
}

#[simd_test]
fn index_f64x8<S: Simd>(simd: S) {
    let values: [f64; 8] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let mut result = f64x8::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42.5_f64;
    assert_eq!(result[0], 42.5_f64);
}

#[simd_test]
fn index_i64x8<S: Simd>(simd: S) {
    let values: [i64; 8] = core::array::from_fn(|i| (i % 23) as i64 + 10_i64);
    let mut result = i64x8::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_i64;
    assert_eq!(result[0], 42_i64);
}

#[simd_test]
fn index_u64x8<S: Simd>(simd: S) {
    let values: [u64; 8] = core::array::from_fn(|i| (i % 23) as u64 + 10_u64);
    let mut result = u64x8::from_slice(simd, &values);
    assert_eq!(result[0], values[0]);
    result[0] = 42_u64;
    assert_eq!(result[0], 42_u64);
}
