// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn cvt_u64_f64x2_regular<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1_234_567_890.25, 9_876_543_210.75]);
    assert_eq!(*a.to_int::<u64x2<_>>(), [1_234_567_890, 9_876_543_210]);
}

#[simd_test]
fn cvt_u64_f64x4_regular<S: Simd>(simd: S) {
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
        *a.to_int::<u64x4<_>>(),
        [1_234_567_890, 2_345_678_901, 8_765_432_109, 9_876_543_210]
    );
}

#[simd_test]
fn cvt_u64_f64x8_regular<S: Simd>(simd: S) {
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
        *a.to_int::<u64x8<_>>(),
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

#[simd_test]
fn cvt_u64_f64_exponent_and_significand_boundaries<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            const FRACTIONS: [u64; 4] = [0, 1, 0x0005_5555_aaaa_aaaa, 0x000f_ffff_ffff_ffff];

            // Sweep every nonnegative exponent field that is still below 2^64. In particular,
            // this covers variable shift counts 64/63, the count-zero pivot at 2^52, and every
            // left shift up through the unsigned-only [2^63, 2^64) interval.
            for exponent in 0_u64..=1086 {
                let values = FRACTIONS.map(|fraction| f64::from_bits(exponent << 52 | fraction));
                let expected = values.map(|value| value as u64);
                let result_x4 = f64x4::from_slice(simd, &values).to_int::<u64x4<_>>();
                assert_eq!(*result_x4, expected, "exponent field {exponent}");

                let result_x2 = f64x2::from_slice(simd, &values[..2]).to_int::<u64x2<_>>();
                assert_eq!(
                    *result_x2,
                    [expected[0], expected[1]],
                    "x2 exponent field {exponent}",
                );
                let result_x2_high = f64x2::from_slice(simd, &values[2..]).to_int::<u64x2<_>>();
                assert_eq!(
                    *result_x2_high,
                    [expected[2], expected[3]],
                    "x2 high fractions, exponent field {exponent}",
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
                    f64::from_bits(exponent << 52 | retained_bit | 1),
                    f64::from_bits(exponent << 52 | 0x000f_ffff_ffff_ffff),
                ];
                let expected = values.map(|value| value as u64);
                let result = f64x4::from_slice(simd, &values).to_int::<u64x4<_>>();
                assert_eq!(
                    *result, expected,
                    "retained-bit transition at exponent field {exponent}, shift {shift}",
                );
                let result_x2_low = f64x2::from_slice(simd, &values[..2]).to_int::<u64x2<_>>();
                assert_eq!(
                    *result_x2_low,
                    [expected[0], expected[1]],
                    "x2 low retained-bit transition at exponent field {exponent}",
                );
                let result_x2_high = f64x2::from_slice(simd, &values[2..]).to_int::<u64x2<_>>();
                assert_eq!(
                    *result_x2_high,
                    [expected[2], expected[3]],
                    "x2 high retained-bit transition at exponent field {exponent}",
                );
            }

            let x8_bits = [
                0x0000_0000_0000_0001,
                0x000f_ffff_ffff_ffff,
                0x3fef_ffff_ffff_ffff,
                0x3ff0_0000_0000_0000,
                0x432f_ffff_ffff_ffff,
                0x4330_0000_0000_0000,
                0x43df_ffff_ffff_ffff,
                0x43ef_ffff_ffff_ffff,
            ];
            let x8_values = x8_bits.map(f64::from_bits);
            let x8_result = f64x8::from_slice(simd, &x8_values).to_int::<u64x8<_>>();
            assert_eq!(*x8_result, x8_values.map(|value| value as u64));
        },
    );
}

#[simd_test]
#[ignore = "checks raw-bit and exponent-stratified in-range f64 values"]
// Run with: cargo test --release cvt_u64_f64_random -- --ignored
fn cvt_u64_f64_random<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            let mut rng = fastrand::Rng::with_seed(0x082e_fa98_ec4e_6c89);

            for iteration in 0..2_500_000 {
                let values: [f64; 4] = core::array::from_fn(|_| {
                    loop {
                        let value = f64::from_bits(rng.u64(..));
                        if (0.0..18_446_744_073_709_551_616.0).contains(&value) {
                            break value;
                        }
                    }
                });
                let expected = values.map(|value| value as u64);

                let result_x4 = f64x4::from_slice(simd, &values).to_int::<u64x4<_>>();
                assert_eq!(*result_x4, expected, "x4 iteration {iteration}");

                let result_x2 = f64x2::from_slice(simd, &values[..2]).to_int::<u64x2<_>>();
                assert_eq!(
                    *result_x2,
                    [expected[0], expected[1]],
                    "x2 iteration {iteration}",
                );

                // Accepted uniform bit patterns overwhelmingly have magnitude below one. Cycle
                // through every exponent that can affect a nonzero in-range result, while still
                // randomizing the full significand.
                let stratified: [f64; 4] = core::array::from_fn(|lane| {
                    let exponent = 1011 + (iteration * 5 + lane) as u64 % 76;
                    let fraction = rng.u64(..) & 0x000f_ffff_ffff_ffff;
                    f64::from_bits(exponent << 52 | fraction)
                });
                let expected_stratified = stratified.map(|value| value as u64);
                let result_stratified = f64x4::from_slice(simd, &stratified).to_int::<u64x4<_>>();
                assert_eq!(
                    *result_stratified, expected_stratified,
                    "stratified x4 iteration {iteration}",
                );
                let result_stratified_x2 =
                    f64x2::from_slice(simd, &stratified[..2]).to_int::<u64x2<_>>();
                assert_eq!(
                    *result_stratified_x2,
                    [expected_stratified[0], expected_stratified[1]],
                    "stratified x2 iteration {iteration}",
                );
            }
        },
    );
}
