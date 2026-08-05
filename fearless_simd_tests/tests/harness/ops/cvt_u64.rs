// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn cvt_u64_f64x2<S: Simd>(simd: S) {
    let values = [0.0, 42.9];
    let a = f64x2::from_slice(simd, &values);
    assert_eq!(*a.to_int::<u64x2<_>>(), values.map(|x| x as u64));
}

#[simd_test]
fn cvt_u64_f64x4<S: Simd>(simd: S) {
    let values = [
        -0.0,
        0.99,
        9_223_372_036_854_775_808.0,
        18_446_744_073_709_549_568.0,
    ];
    let a = f64x4::from_slice(simd, &values);
    assert_eq!(*a.to_int::<u64x4<_>>(), values.map(|x| x as u64));
}

#[simd_test]
fn cvt_u64_f64x8<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtTruncate;
    let values = [
        0.0,
        -0.0,
        0.99,
        1.99,
        42.9,
        1e9,
        1e15,
        18_446_744_073_709_549_568.0,
    ];
    let a = f64x8::from_slice(simd, &values);
    let result = u64x8::truncate_from(a);
    assert_eq!(*result, values.map(|x| x as u64));
}
