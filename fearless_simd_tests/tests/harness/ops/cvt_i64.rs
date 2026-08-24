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
fn cvt_i64_f64_exponent_and_significand_boundaries<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            const FRACTIONS: [u64; 4] = [0, 1, 0x0005_5555_aaaa_aaaa, 0x000f_ffff_ffff_ffff];

            // Every exponent field whose positive and negative values are both in range. This
            // covers both variable-shift directions, including counts 64/63, the count-zero
            // pivot at exponent field 1075, and every left shift through the signed limit.
            for exponent in 0_u64..=1085 {
                let positive = FRACTIONS.map(|fraction| f64::from_bits(exponent << 52 | fraction));
                let expected_positive = positive.map(|value| value as i64);
                let result_positive = f64x4::from_slice(simd, &positive).to_int::<i64x4<_>>();
                assert_eq!(
                    *result_positive, expected_positive,
                    "positive exponent field {exponent}",
                );
                let result_positive_x2 =
                    f64x2::from_slice(simd, &positive[..2]).to_int::<i64x2<_>>();
                assert_eq!(
                    *result_positive_x2,
                    [expected_positive[0], expected_positive[1]],
                    "positive x2 exponent field {exponent}",
                );
                let result_positive_x2_high =
                    f64x2::from_slice(simd, &positive[2..]).to_int::<i64x2<_>>();
                assert_eq!(
                    *result_positive_x2_high,
                    [expected_positive[2], expected_positive[3]],
                    "positive x2 high fractions, exponent field {exponent}",
                );

                let negative = FRACTIONS
                    .map(|fraction| f64::from_bits(1_u64 << 63 | exponent << 52 | fraction));
                let expected_negative = negative.map(|value| value as i64);
                let result_negative = f64x4::from_slice(simd, &negative).to_int::<i64x4<_>>();
                assert_eq!(
                    *result_negative, expected_negative,
                    "negative exponent field {exponent}",
                );
                let result_negative_x2 =
                    f64x2::from_slice(simd, &negative[..2]).to_int::<i64x2<_>>();
                assert_eq!(
                    *result_negative_x2,
                    [expected_negative[0], expected_negative[1]],
                    "negative x2 exponent field {exponent}",
                );
                let result_negative_x2_high =
                    f64x2::from_slice(simd, &negative[2..]).to_int::<i64x2<_>>();
                assert_eq!(
                    *result_negative_x2_high,
                    [expected_negative[2], expected_negative[3]],
                    "negative x2 high fractions, exponent field {exponent}",
                );
            }

            // Put the lowest fraction bit that survives truncation immediately before and after
            // its transition. This catches off-by-one shift counts that broad random bits hide.
            for exponent in 1024_u64..=1074 {
                let shift = 1075 - exponent;
                let retained_bit = 1_u64 << shift;
                let values = [
                    f64::from_bits((exponent << 52) | (retained_bit - 1)),
                    f64::from_bits(exponent << 52 | retained_bit),
                    f64::from_bits(1_u64 << 63 | exponent << 52 | (retained_bit - 1)),
                    f64::from_bits(1_u64 << 63 | exponent << 52 | retained_bit),
                ];
                let expected = values.map(|value| value as i64);
                let result = f64x4::from_slice(simd, &values).to_int::<i64x4<_>>();
                assert_eq!(
                    *result, expected,
                    "retained-bit transition at exponent field {exponent}, shift {shift}",
                );
                let result_positive_x2 = f64x2::from_slice(simd, &values[..2]).to_int::<i64x2<_>>();
                assert_eq!(
                    *result_positive_x2,
                    [expected[0], expected[1]],
                    "positive x2 retained-bit transition at exponent field {exponent}",
                );
                let result_negative_x2 = f64x2::from_slice(simd, &values[2..]).to_int::<i64x2<_>>();
                assert_eq!(
                    *result_negative_x2,
                    [expected[2], expected[3]],
                    "negative x2 retained-bit transition at exponent field {exponent}",
                );
            }

            let x8_bits = [
                0x0000_0000_0000_0001,
                0x8000_0000_0000_0001,
                0x3fef_ffff_ffff_ffff,
                0xbfef_ffff_ffff_ffff,
                0x3ff0_0000_0000_0000,
                0xbff0_0000_0000_0000,
                0x43df_ffff_ffff_ffff,
                0xc3e0_0000_0000_0000,
            ];
            let x8_values = x8_bits.map(f64::from_bits);
            let x8_result = f64x8::from_slice(simd, &x8_values).to_int::<i64x8<_>>();
            assert_eq!(*x8_result, x8_values.map(|value| value as i64));
        },
    );
}

#[simd_test]
#[ignore = "checks raw-bit and exponent-stratified in-range f64 values"]
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

                // Accepted uniform bit patterns overwhelmingly have magnitude below one. Cycle
                // through every exponent that can affect a nonzero in-range result, while still
                // randomizing the full significand and sign.
                let stratified: [f64; 4] = core::array::from_fn(|lane| {
                    let exponent = 1011 + (iteration * 4 + lane) as u64 % 75;
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let fraction = rng.u64(..) & 0x000f_ffff_ffff_ffff;
                    f64::from_bits(sign | exponent << 52 | fraction)
                });
                let expected_stratified = stratified.map(|value| value as i64);
                let result_stratified = f64x4::from_slice(simd, &stratified).to_int::<i64x4<_>>();
                assert_eq!(
                    *result_stratified, expected_stratified,
                    "stratified x4 iteration {iteration}",
                );
                let result_stratified_x2 =
                    f64x2::from_slice(simd, &stratified[..2]).to_int::<i64x2<_>>();
                assert_eq!(
                    *result_stratified_x2,
                    [expected_stratified[0], expected_stratified[1]],
                    "stratified x2 iteration {iteration}",
                );
            }
        },
    );
}
