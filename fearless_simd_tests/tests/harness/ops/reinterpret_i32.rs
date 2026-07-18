// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn reinterpret_i32_f32x4<S: Simd>(simd: S) {
    let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let a = f32x4::from_slice(simd, &values);
    let result = simd.reinterpret_i32_f32x4(a);
    let roundtrip = f32x4::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_i32_f32x8<S: Simd>(simd: S) {
    let values: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = f32x8::from_slice(simd, &values);
    let result = simd.reinterpret_i32_f32x8(a);
    let roundtrip = f32x8::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_i32_f32x16<S: Simd>(simd: S) {
    let values: [f32; 16] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let a = f32x16::from_slice(simd, &values);
    let result = simd.reinterpret_i32_f32x16(a);
    let roundtrip = f32x16::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}
