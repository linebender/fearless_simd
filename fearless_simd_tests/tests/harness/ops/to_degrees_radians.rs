// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

use std::{f32, f64};

const VALUES_F32: &[f32] = &[
    0.0,
    1.0,
    -1.0,
    45.0,
    180.0,
    360.0,
    f32::consts::PI,
    f32::consts::FRAC_PI_2,
    0.0,
    1.0,
    -1.0,
    45.0,
    180.0,
    360.0,
    f32::consts::PI,
    f32::consts::FRAC_PI_2,
];
const VALUES_F64: &[f64] = &[
    0.0,
    1.0,
    -1.0,
    45.0,
    180.0,
    360.0,
    f64::consts::PI,
    f64::consts::FRAC_PI_2,
];

#[simd_test]
fn test_f32x4<S: Simd>(simd: S) {
    let a = f32x4::<S>::from_slice(simd, &VALUES_F32[..f32x4::<S>::LEN]);
    assert_eq!(*(a.to_degrees()), a.map(f32::to_degrees));
    assert_eq!(*(a.to_radians()), a.map(f32::to_radians));
}
#[simd_test]
fn test_f32x8<S: Simd>(simd: S) {
    let a = f32x8::<S>::from_slice(simd, &VALUES_F32[..f32x8::<S>::LEN]);
    assert_eq!(*(a.to_degrees()), a.map(f32::to_degrees));
    assert_eq!(*(a.to_radians()), a.map(f32::to_radians));
}
#[simd_test]
fn test_f32x16<S: Simd>(simd: S) {
    let a = f32x16::<S>::from_slice(simd, &VALUES_F32[..f32x16::<S>::LEN]);
    assert_eq!(*(a.to_degrees()), a.map(f32::to_degrees));
    assert_eq!(*(a.to_radians()), a.map(f32::to_radians));
}
#[simd_test]
fn test_f64x2<S: Simd>(simd: S) {
    let a = f64x2::<S>::from_slice(simd, &VALUES_F64[..f64x2::<S>::LEN]);
    assert_eq!(*(a.to_degrees()), a.map(f64::to_degrees));
    assert_eq!(*(a.to_radians()), a.map(f64::to_radians));
}
#[simd_test]
fn test_f64x4<S: Simd>(simd: S) {
    let a = f64x4::<S>::from_slice(simd, &VALUES_F64[..f64x4::<S>::LEN]);
    assert_eq!(*(a.to_degrees()), a.map(f64::to_degrees));
    assert_eq!(*(a.to_radians()), a.map(f64::to_radians));
}
#[simd_test]
fn test_f64x8<S: Simd>(simd: S) {
    let a = f64x8::<S>::from_slice(simd, &VALUES_F64[..f64x8::<S>::LEN]);
    assert_eq!(*(a.to_degrees()), a.map(f64::to_degrees));
    assert_eq!(*(a.to_radians()), a.map(f64::to_radians));
}
