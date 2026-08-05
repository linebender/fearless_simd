// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn cvt_i64_precise_f64x2<S: Simd>(simd: S) {
    let values = [f64::NAN, f64::INFINITY];
    let a = f64x2::from_slice(simd, &values);
    assert_eq!(*a.to_int_precise::<i64x2<_>>(), values.map(|x| x as i64));
}

#[simd_test]
fn cvt_i64_precise_f64x4<S: Simd>(simd: S) {
    let values = [
        f64::NEG_INFINITY,
        -9_223_372_036_854_775_808.0,
        9_223_372_036_854_774_784.0,
        9_223_372_036_854_775_808.0,
    ];
    let a = f64x4::from_slice(simd, &values);
    assert_eq!(*a.to_int_precise::<i64x4<_>>(), values.map(|x| x as i64));
}

#[simd_test]
fn cvt_i64_precise_f64x8<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtTruncate;
    let values = [
        f64::NAN,
        f64::NEG_INFINITY,
        f64::INFINITY,
        -1e300,
        1e300,
        -42.9,
        -0.0,
        42.9,
    ];
    let a = f64x8::from_slice(simd, &values);
    let result = i64x8::truncate_from_precise(a);
    assert_eq!(*result, values.map(|x| x as i64));
}
