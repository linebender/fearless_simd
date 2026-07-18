// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

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
