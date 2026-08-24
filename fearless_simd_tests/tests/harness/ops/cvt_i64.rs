// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn cvt_i64_f64x2_regular<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[-9_876_543_210.75, 1_234_567_890.25]);
    assert_eq!(*a.to_int::<i64x2<_>>(), [-9_876_543_210, 1_234_567_890]);
}

#[simd_test]
fn cvt_i64_f64x4_regular<S: Simd>(simd: S) {
    let a = f64x4::from_slice(
        simd,
        &[
            -9_876_543_210.75,
            -1_234_567_890.25,
            2_345_678_901.5,
            8_765_432_109.875,
        ],
    );
    assert_eq!(
        *a.to_int::<i64x4<_>>(),
        [-9_876_543_210, -1_234_567_890, 2_345_678_901, 8_765_432_109]
    );
}

#[simd_test]
fn cvt_i64_f64x8_regular<S: Simd>(simd: S) {
    let a = f64x8::from_slice(
        simd,
        &[
            -9_876_543_210.75,
            -8_765_432_109.625,
            -7_654_321_098.5,
            -6_543_210_987.375,
            5_432_109_876.25,
            4_321_098_765.125,
            3_210_987_654.875,
            2_109_876_543.75,
        ],
    );
    assert_eq!(
        *a.to_int::<i64x8<_>>(),
        [
            -9_876_543_210,
            -8_765_432_109,
            -7_654_321_098,
            -6_543_210_987,
            5_432_109_876,
            4_321_098_765,
            3_210_987_654,
            2_109_876_543,
        ]
    );
}

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

#[simd_test]
#[ignore = "randomly checks 10 million in-range f64 bit patterns"]
// Run with: cargo test --release cvt_i64_f64_random -- --ignored
fn cvt_i64_f64_random<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            let mut rng = fastrand::Rng::with_seed(0x1319_8a2e_0370_7344);

            for iteration in 0..2_500_000 {
                let values: [f64; 4] = core::array::from_fn(|_| {
                    loop {
                        let value = f64::from_bits(rng.u64(..));
                        if (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0)
                            .contains(&value)
                        {
                            break value;
                        }
                    }
                });
                let expected = values.map(|value| value as i64);

                let result_x4 = f64x4::from_slice(simd, &values).to_int::<i64x4<_>>();
                assert_eq!(*result_x4, expected, "x4 iteration {iteration}");

                let result_x2 = f64x2::from_slice(simd, &values[..2]).to_int::<i64x2<_>>();
                assert_eq!(
                    *result_x2,
                    [expected[0], expected[1]],
                    "x2 iteration {iteration}",
                );
            }
        },
    );
}
