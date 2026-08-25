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

#[simd_test]
fn cvt_u64_precise_f64_all_exponents_and_special_values<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            const FRACTIONS: [u64; 4] = [0, 1, 0x0005_5555_aaaa_aaaa, 0x000f_ffff_ffff_ffff];

            // Cover every exponent with zero, minimal, alternating, and maximal fraction fields,
            // for both signs. Besides every reconstruction shift, exponent 2047 supplies both
            // infinities and signaling/quiet NaNs with different payloads.
            for exponent in 0_u64..=2047 {
                let positive = FRACTIONS.map(|fraction| f64::from_bits(exponent << 52 | fraction));
                let expected_positive = positive.map(|value| value as u64);
                let result_positive =
                    f64x4::from_slice(simd, &positive).to_int_precise::<u64x4<_>>();
                assert_eq!(
                    *result_positive, expected_positive,
                    "positive exponent field {exponent}",
                );
                let result_positive_x2 =
                    f64x2::from_slice(simd, &positive[..2]).to_int_precise::<u64x2<_>>();
                assert_eq!(
                    *result_positive_x2,
                    [expected_positive[0], expected_positive[1]],
                    "positive x2 exponent field {exponent}",
                );
                let result_positive_x2_high =
                    f64x2::from_slice(simd, &positive[2..]).to_int_precise::<u64x2<_>>();
                assert_eq!(
                    *result_positive_x2_high,
                    [expected_positive[2], expected_positive[3]],
                    "positive x2 high fractions, exponent field {exponent}",
                );

                let negative = FRACTIONS
                    .map(|fraction| f64::from_bits(1_u64 << 63 | exponent << 52 | fraction));
                let expected_negative = negative.map(|value| value as u64);
                let result_negative =
                    f64x4::from_slice(simd, &negative).to_int_precise::<u64x4<_>>();
                assert_eq!(
                    *result_negative, expected_negative,
                    "negative exponent field {exponent}",
                );
                let result_negative_x2 =
                    f64x2::from_slice(simd, &negative[..2]).to_int_precise::<u64x2<_>>();
                assert_eq!(
                    *result_negative_x2,
                    [expected_negative[0], expected_negative[1]],
                    "negative x2 exponent field {exponent}",
                );
                let result_negative_x2_high =
                    f64x2::from_slice(simd, &negative[2..]).to_int_precise::<u64x2<_>>();
                assert_eq!(
                    *result_negative_x2_high,
                    [expected_negative[2], expected_negative[3]],
                    "negative x2 high fractions, exponent field {exponent}",
                );
            }

            let boundary_bits = [
                0x8000_0000_0000_0001,
                0xbff0_0000_0000_0000,
                0x0000_0000_0000_0000,
                0x3ff0_0000_0000_0001,
                0x43ef_ffff_ffff_ffff,
                0x43f0_0000_0000_0000,
                0x43f0_0000_0000_0001,
                0x7fef_ffff_ffff_ffff,
                0xfff0_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0x7ff0_0000_0000_0001,
                0xfff0_0000_0000_0001,
                0x7ff8_0000_0000_0000,
                0xfff8_0000_0000_0000,
                0x7fff_ffff_ffff_ffff,
                0xffff_ffff_ffff_ffff,
            ];
            for (group_index, bits) in boundary_bits.chunks_exact(8).enumerate() {
                let values: [f64; 8] = core::array::from_fn(|lane| f64::from_bits(bits[lane]));
                let expected = values.map(|value| value as u64);
                let result = f64x8::from_slice(simd, &values).to_int_precise::<u64x8<_>>();
                assert_eq!(*result, expected, "boundary/special x8 group {group_index}");
            }
        },
    );
}

#[simd_test]
#[ignore = "checks arbitrary bits plus randomized critical exponent fields"]
// Run with: cargo test --release cvt_u64_precise_f64_random -- --ignored
fn cvt_u64_precise_f64_random<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            const CRITICAL_EXPONENTS: [u64; 16] = [
                0, 1, 1011, 1012, 1022, 1023, 1024, 1074, 1075, 1076, 1085, 1086, 1087, 1088, 2046,
                2047,
            ];
            let mut rng = fastrand::Rng::with_seed(0x4528_21e6_38d0_1377);

            for iteration in 0..2_500_000 {
                let values = [
                    f64::from_bits(rng.u64(..)),
                    f64::from_bits(rng.u64(..)),
                    f64::from_bits(rng.u64(..)),
                    f64::from_bits(rng.u64(..)),
                ];
                let expected = values.map(|value| value as u64);

                let result_x4 = f64x4::from_slice(simd, &values).to_int_precise::<u64x4<_>>();
                assert_eq!(*result_x4, expected, "x4 iteration {iteration}");

                let result_x2 = f64x2::from_slice(simd, &values[..2]).to_int_precise::<u64x2<_>>();
                assert_eq!(
                    *result_x2,
                    [expected[0], expected[1]],
                    "x2 iteration {iteration}",
                );

                let targeted: [f64; 4] = core::array::from_fn(|lane| {
                    let exponent =
                        CRITICAL_EXPONENTS[(iteration * 5 + lane) % CRITICAL_EXPONENTS.len()];
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let fraction = rng.u64(..) & 0x000f_ffff_ffff_ffff;
                    f64::from_bits(sign | exponent << 52 | fraction)
                });
                let expected_targeted = targeted.map(|value| value as u64);
                let result_targeted =
                    f64x4::from_slice(simd, &targeted).to_int_precise::<u64x4<_>>();
                assert_eq!(
                    *result_targeted, expected_targeted,
                    "targeted x4 iteration {iteration}",
                );
                let result_targeted_x2 =
                    f64x2::from_slice(simd, &targeted[..2]).to_int_precise::<u64x2<_>>();
                assert_eq!(
                    *result_targeted_x2,
                    [expected_targeted[0], expected_targeted[1]],
                    "targeted x2 iteration {iteration}",
                );
            }
        },
    );
}
