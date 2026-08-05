// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn cvt_f64_i64x2<S: Simd>(simd: S) {
    let values = [i64::MIN, i64::MAX];
    let a = i64x2::from_slice(simd, &values);
    assert_eq!(*a.to_float::<f64x2<_>>(), values.map(|x| x as f64));
}

#[simd_test]
fn cvt_f64_u64x2<S: Simd>(simd: S) {
    let values = [0, u64::MAX];
    let a = u64x2::from_slice(simd, &values);
    assert_eq!(*a.to_float::<f64x2<_>>(), values.map(|x| x as f64));
}

#[simd_test]
fn cvt_f64_i64x4<S: Simd>(simd: S) {
    let values = [
        i64::MIN,
        -9_007_199_254_740_993,
        9_007_199_254_740_993,
        i64::MAX,
    ];
    let a = i64x4::from_slice(simd, &values);
    assert_eq!(*a.to_float::<f64x4<_>>(), values.map(|x| x as f64));
}

#[simd_test]
fn cvt_f64_u64x4<S: Simd>(simd: S) {
    let values = [0, 9_007_199_254_740_993, 1 << 63, u64::MAX];
    let a = u64x4::from_slice(simd, &values);
    assert_eq!(*a.to_float::<f64x4<_>>(), values.map(|x| x as f64));
}

#[simd_test]
fn cvt_f64_i64x8<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtFloat;
    let values = [
        i64::MIN,
        -9_007_199_254_740_993,
        -42,
        -1,
        0,
        1,
        9_007_199_254_740_993,
        i64::MAX,
    ];
    let a = i64x8::from_slice(simd, &values);
    let result = f64x8::float_from(a);
    assert_eq!(*result, values.map(|x| x as f64));
}

#[simd_test]
fn cvt_f64_u64x8<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtFloat;
    let values = [
        0,
        1,
        42,
        1 << 53,
        (1 << 53) + 1,
        1 << 63,
        u64::MAX - 1,
        u64::MAX,
    ];
    let a = u64x8::from_slice(simd, &values);
    let result = f64x8::float_from(a);
    assert_eq!(*result, values.map(|x| x as f64));
}
