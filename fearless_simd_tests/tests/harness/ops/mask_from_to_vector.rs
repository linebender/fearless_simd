// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn from_to_vector_mask32x4<S: Simd>(simd: S) {
    let values: [i32; 4] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let a = i32x4::from_slice(simd, &values);
    let mask = mask32x4::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i32; 4];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask8x16<S: Simd>(simd: S) {
    let values: [i8; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let a = i8x16::from_slice(simd, &values);
    let mask = mask8x16::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i8; 16];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask8x32<S: Simd>(simd: S) {
    let values: [i8; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let a = i8x32::from_slice(simd, &values);
    let mask = mask8x32::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i8; 32];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask8x64<S: Simd>(simd: S) {
    let values: [i8; 64] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let a = i8x64::from_slice(simd, &values);
    let mask = mask8x64::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i8; 64];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask16x8<S: Simd>(simd: S) {
    let values: [i16; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let a = i16x8::from_slice(simd, &values);
    let mask = mask16x8::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i16; 8];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask16x16<S: Simd>(simd: S) {
    let values: [i16; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let a = i16x16::from_slice(simd, &values);
    let mask = mask16x16::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i16; 16];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask16x32<S: Simd>(simd: S) {
    let values: [i16; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let a = i16x32::from_slice(simd, &values);
    let mask = mask16x32::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i16; 32];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let a = i32x8::from_slice(simd, &values);
    let mask = mask32x8::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i32; 8];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask32x16<S: Simd>(simd: S) {
    let values: [i32; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let a = i32x16::from_slice(simd, &values);
    let mask = mask32x16::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i32; 16];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask64x2<S: Simd>(simd: S) {
    let values: [i64; 2] = core::array::from_fn(|i| if i % 2 == 0 { -1_i64 } else { 0_i64 });
    let a = i64x2::from_slice(simd, &values);
    let mask = mask64x2::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i64; 2];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask64x4<S: Simd>(simd: S) {
    let values: [i64; 4] = core::array::from_fn(|i| if i % 2 == 0 { -1_i64 } else { 0_i64 });
    let a = i64x4::from_slice(simd, &values);
    let mask = mask64x4::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i64; 4];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}

#[simd_test]
fn from_to_vector_mask64x8<S: Simd>(simd: S) {
    let values: [i64; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1_i64 } else { 0_i64 });
    let a = i64x8::from_slice(simd, &values);
    let mask = mask64x8::from_vector(a);
    let a = mask.to_vector();
    let mut out = [0_i64; 8];
    a.store_slice(&mut out);
    assert_eq!(out, values);
}
