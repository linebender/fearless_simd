// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn cvt_i64_f64x2<S: Simd>(simd: S) {
    let values = [-42.9, 42.9];
    let a = f64x2::from_slice(simd, &values);
    assert_eq!(*a.to_int::<i64x2<_>>(), values.map(|x| x as i64));
}

#[simd_test]
fn cvt_i64_f64x4<S: Simd>(simd: S) {
    let values = [
        -0.0,
        0.0,
        -9_223_372_036_854_775_808.0,
        9_223_372_036_854_774_784.0,
    ];
    let a = f64x4::from_slice(simd, &values);
    assert_eq!(*a.to_int::<i64x4<_>>(), values.map(|x| x as i64));
}

#[simd_test]
fn cvt_i64_f64x8<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtTruncate;
    let values = [-1234.75, -1.99, -0.99, -0.0, 0.0, 0.99, 1.99, 1234.75];
    let a = f64x8::from_slice(simd, &values);
    let result = i64x8::truncate_from(a);
    assert_eq!(*result, values.map(|x| x as i64));
}
