// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn reinterpret_f32_f64x2<S: Simd>(simd: S) {
    let values: [f64; 2] = [1.0, 2.0];
    let a = f64x2::from_slice(simd, &values);
    let result = simd.reinterpret_f32_f64x2(a);
    let roundtrip = f64x2::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_f32_f64x4<S: Simd>(simd: S) {
    let values: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let a = f64x4::from_slice(simd, &values);
    let result = simd.reinterpret_f32_f64x4(a);
    let roundtrip = f64x4::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}

#[simd_test]
fn reinterpret_f32_f64x8<S: Simd>(simd: S) {
    let values: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = f64x8::from_slice(simd, &values);
    let result = simd.reinterpret_f32_f64x8(a);
    let roundtrip = f64x8::from_bytes(result.to_bytes());
    assert_eq!(roundtrip.as_slice(), values.as_slice());
}
