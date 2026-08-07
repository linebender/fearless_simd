// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn cvt_u64_precise_f64x2_regular<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1_234_567_890.25, 9_876_543_210.75]);
    assert_eq!(
        *a.to_int_precise::<u64x2<_>>(),
        [1_234_567_890, 9_876_543_210]
    );
}

#[simd_test]
fn cvt_u64_precise_f64x4_regular<S: Simd>(simd: S) {
    let a = f64x4::from_slice(
        simd,
        &[
            1_234_567_890.25,
            2_345_678_901.5,
            8_765_432_109.875,
            9_876_543_210.75,
        ],
    );
    assert_eq!(
        *a.to_int_precise::<u64x4<_>>(),
        [1_234_567_890, 2_345_678_901, 8_765_432_109, 9_876_543_210]
    );
}

#[simd_test]
fn cvt_u64_precise_f64x8_regular<S: Simd>(simd: S) {
    let a = f64x8::from_slice(
        simd,
        &[
            9_876_543_210.75,
            8_765_432_109.625,
            7_654_321_098.5,
            6_543_210_987.375,
            5_432_109_876.25,
            4_321_098_765.125,
            3_210_987_654.875,
            2_109_876_543.75,
        ],
    );
    assert_eq!(
        *a.to_int_precise::<u64x8<_>>(),
        [
            9_876_543_210,
            8_765_432_109,
            7_654_321_098,
            6_543_210_987,
            5_432_109_876,
            4_321_098_765,
            3_210_987_654,
            2_109_876_543,
        ]
    );
}

#[simd_test]
fn cvt_u64_precise_f64x2<S: Simd>(simd: S) {
    let values = [f64::NAN, f64::INFINITY];
    let a = f64x2::from_slice(simd, &values);
    assert_eq!(*a.to_int_precise::<u64x2<_>>(), values.map(|x| x as u64));
}

#[simd_test]
fn cvt_u64_precise_f64x4<S: Simd>(simd: S) {
    let values = [
        f64::NEG_INFINITY,
        -1.0,
        18_446_744_073_709_549_568.0,
        18_446_744_073_709_551_616.0,
    ];
    let a = f64x4::from_slice(simd, &values);
    assert_eq!(*a.to_int_precise::<u64x4<_>>(), values.map(|x| x as u64));
}

#[simd_test]
fn cvt_u64_precise_f64x8<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtTruncate;
    let values = [
        f64::NAN,
        f64::NEG_INFINITY,
        f64::INFINITY,
        -1e300,
        1e300,
        -0.0,
        42.9,
        18_446_744_073_709_549_568.0,
    ];
    let a = f64x8::from_slice(simd, &values);
    let result = u64x8::truncate_from_precise(a);
    assert_eq!(*result, values.map(|x| x as u64));
}
