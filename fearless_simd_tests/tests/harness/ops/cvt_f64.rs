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

#[simd_test]
fn cvt_f64_i64_u64_adversarial<S: Simd>(simd: S) {
    // Keep the oracle in integer arithmetic. In release builds, LLVM can vectorize a scalar
    // u64-to-f64 map into the same split/bias algorithm that this test is meant to check.
    fn u64_to_f64_bits(value: u64) -> u64 {
        if value == 0 {
            return 0;
        }

        let fraction_mask = (1_u64 << 52) - 1;
        let mut exponent = 63 - value.leading_zeros();
        if exponent <= 52 {
            let fraction = (value << (52 - exponent)) & fraction_mask;
            return ((u64::from(exponent) + 1023) << 52) | fraction;
        }

        let shift = exponent - 52;
        let mut significand = value >> shift;
        let remainder = value & ((1_u64 << shift) - 1);
        let halfway = 1_u64 << (shift - 1);
        if remainder > halfway || (remainder == halfway && significand & 1 != 0) {
            significand += 1;
            if significand == 1_u64 << 53 {
                significand >>= 1;
                exponent += 1;
            }
        }

        ((u64::from(exponent) + 1023) << 52) | (significand & fraction_mask)
    }

    fn i64_to_f64_bits(value: i64) -> u64 {
        ((value < 0) as u64) << 63 | u64_to_f64_bits(value.unsigned_abs())
    }

    simd.vectorize(
        #[inline(always)]
        || {
            // These values exercise the 32-bit split, signed high-half bias, cancellation,
            // carry between halves, and rounding carries at the ends of the integer ranges.
            let unsigned_boundaries = [
                0,
                1,
                0x0000_0000_7fff_ffff,
                0x0000_0000_8000_0000,
                0x0000_0000_ffff_fffe,
                0x0000_0000_ffff_ffff,
                0x0000_0001_0000_0000,
                0x0000_0001_0000_0001,
                0x0000_0001_ffff_ffff,
                0x7fff_ffff_0000_0001,
                0x8000_0000_ffff_ffff,
                0xffff_ffff_0000_0001,
                (1_u64 << 53) - 1,
                1_u64 << 53,
                (1_u64 << 53) + 1,
                (1_u64 << 53) + 3,
                1_u64 << 63,
                (1_u64 << 63) + 1023,
                (1_u64 << 63) + 1024,
                (1_u64 << 63) + 1025,
                u64::MAX - 2047,
                u64::MAX - 1024,
                u64::MAX - 1023,
                u64::MAX - 1022,
                u64::MAX - 1,
                u64::MAX,
                0x5555_aaaa_ffff_0001,
                0xaaaa_5555_0000_ffff,
            ];
            for (group_index, values) in unsigned_boundaries.chunks_exact(4).enumerate() {
                let values = [values[0], values[1], values[2], values[3]];
                let expected = values.map(u64_to_f64_bits);

                let result_x4 = u64x4::from_slice(simd, &values).to_float::<f64x4<_>>();
                assert_eq!(
                    (*result_x4).map(f64::to_bits),
                    expected,
                    "unsigned split/end-point group {group_index}",
                );

                let result_x2 = u64x2::from_slice(simd, &values[..2]).to_float::<f64x2<_>>();
                assert_eq!(
                    (*result_x2).map(f64::to_bits),
                    [expected[0], expected[1]],
                    "unsigned x2 split/end-point group {group_index}",
                );
            }

            let signed_boundaries = [
                i64::MIN,
                i64::MIN + 511,
                i64::MIN + 512,
                i64::MIN + 513,
                i64::MIN + 1023,
                i64::MIN + 1024,
                i64::MAX - 512,
                i64::MAX - 511,
                i64::MAX - 510,
                i64::MAX - 1,
                i64::MAX,
                -((1_i64 << 53) + 3),
                -((1_i64 << 32) + 1),
                -(1_i64 << 32),
                -((1_i64 << 32) - 1),
                -1,
                0,
                1,
                (1_i64 << 32) - 1,
                1_i64 << 32,
                (1_i64 << 32) + 1,
                -((1_i64 << 53) + 1),
                (1_i64 << 53) + 1,
                (1_i64 << 53) + 3,
                0x0000_0001_ffff_ffff,
                0x7fff_ffff_0000_0001,
                0x5555_aaaa_ffff_0001,
                -0x5555_aaaa_ffff_0001,
            ];
            for (group_index, values) in signed_boundaries.chunks_exact(4).enumerate() {
                let values = [values[0], values[1], values[2], values[3]];
                let expected = values.map(i64_to_f64_bits);

                let result_x4 = i64x4::from_slice(simd, &values).to_float::<f64x4<_>>();
                assert_eq!(
                    (*result_x4).map(f64::to_bits),
                    expected,
                    "signed split/end-point group {group_index}",
                );

                let result_x2 = i64x2::from_slice(simd, &values[..2]).to_float::<f64x2<_>>();
                assert_eq!(
                    (*result_x2).map(f64::to_bits),
                    [expected[0], expected[1]],
                    "signed x2 split/end-point group {group_index}",
                );
            }

            // Guarantee that the two-lane kernels see the normalization-carry midpoints too;
            // their positions in the larger boundary tables are primarily chosen for x4.
            let unsigned_x2_carry_values = [u64::MAX - 1023, u64::MAX - 1022];
            let unsigned_x2_carry =
                u64x2::from_slice(simd, &unsigned_x2_carry_values).to_float::<f64x2<_>>();
            assert_eq!(
                (*unsigned_x2_carry).map(f64::to_bits),
                unsigned_x2_carry_values.map(u64_to_f64_bits),
                "unsigned x2 normalization carry",
            );

            let signed_x2_carry_values = [i64::MIN + 512, i64::MAX - 511];
            let signed_x2_carry =
                i64x2::from_slice(simd, &signed_x2_carry_values).to_float::<f64x2<_>>();
            assert_eq!(
                (*signed_x2_carry).map(f64::to_bits),
                signed_x2_carry_values.map(i64_to_f64_bits),
                "signed x2 normalization carries",
            );

            // Sweep every discarded-bit pattern in two adjacent output intervals. The first
            // retained significand is even and the second is odd, so this exhaustively covers
            // both directions of ties-to-even for every integer-to-f64 spacing from 2 to 2048.
            const EVEN_SIGNIFICAND: u64 = 0x0015_5555_aaaa_aaaa;
            for shift in 1..=11 {
                let spacing = 1_u64 << shift;
                let base = EVEN_SIGNIFICAND << shift;
                let mut offset = 0;
                while offset <= 2 * spacing {
                    let values: [u64; 4] =
                        core::array::from_fn(|lane| base + (offset + lane as u64).min(2 * spacing));
                    let expected = values.map(u64_to_f64_bits);

                    let result_x4 = u64x4::from_slice(simd, &values).to_float::<f64x4<_>>();
                    assert_eq!(
                        (*result_x4).map(f64::to_bits),
                        expected,
                        "unsigned residue sweep shift {shift}, offset {offset}",
                    );
                    let result_x2 = u64x2::from_slice(simd, &values[..2]).to_float::<f64x2<_>>();
                    assert_eq!(
                        (*result_x2).map(f64::to_bits),
                        [expected[0], expected[1]],
                        "unsigned x2 residue sweep shift {shift}, offset {offset}",
                    );
                    let result_x2_high =
                        u64x2::from_slice(simd, &values[2..]).to_float::<f64x2<_>>();
                    assert_eq!(
                        (*result_x2_high).map(f64::to_bits),
                        [expected[2], expected[3]],
                        "unsigned x2 high residue sweep shift {shift}, offset {offset}",
                    );

                    if shift <= 10 {
                        let positive = values.map(|value| value as i64);
                        let expected_positive = positive.map(i64_to_f64_bits);
                        let result_positive =
                            i64x4::from_slice(simd, &positive).to_float::<f64x4<_>>();
                        assert_eq!(
                            (*result_positive).map(f64::to_bits),
                            expected_positive,
                            "positive signed residue sweep shift {shift}, offset {offset}",
                        );
                        let result_positive_x2 =
                            i64x2::from_slice(simd, &positive[..2]).to_float::<f64x2<_>>();
                        assert_eq!(
                            (*result_positive_x2).map(f64::to_bits),
                            [expected_positive[0], expected_positive[1]],
                            "positive signed x2 residue sweep shift {shift}, offset {offset}",
                        );
                        let result_positive_x2_high =
                            i64x2::from_slice(simd, &positive[2..]).to_float::<f64x2<_>>();
                        assert_eq!(
                            (*result_positive_x2_high).map(f64::to_bits),
                            [expected_positive[2], expected_positive[3]],
                            "positive signed x2 high residue sweep shift {shift}, offset {offset}",
                        );

                        let negative = positive.map(|value| -value);
                        let expected_negative = negative.map(i64_to_f64_bits);
                        let result_negative =
                            i64x4::from_slice(simd, &negative).to_float::<f64x4<_>>();
                        assert_eq!(
                            (*result_negative).map(f64::to_bits),
                            expected_negative,
                            "negative signed residue sweep shift {shift}, offset {offset}",
                        );
                        let result_negative_x2 =
                            i64x2::from_slice(simd, &negative[..2]).to_float::<f64x2<_>>();
                        assert_eq!(
                            (*result_negative_x2).map(f64::to_bits),
                            [expected_negative[0], expected_negative[1]],
                            "negative signed x2 residue sweep shift {shift}, offset {offset}",
                        );
                        let result_negative_x2_high =
                            i64x2::from_slice(simd, &negative[2..]).to_float::<f64x2<_>>();
                        assert_eq!(
                            (*result_negative_x2_high).map(f64::to_bits),
                            [expected_negative[2], expected_negative[3]],
                            "negative signed x2 high residue sweep shift {shift}, offset {offset}",
                        );
                    }

                    offset += 4;
                }
            }

            let unsigned_x8_values = [
                0,
                0xffff_ffff,
                0x1_0000_0000,
                (1_u64 << 53) + 3,
                1_u64 << 63,
                u64::MAX - 1023,
                u64::MAX - 1,
                u64::MAX,
            ];
            let unsigned_x8 = u64x8::from_slice(simd, &unsigned_x8_values).to_float::<f64x8<_>>();
            assert_eq!(
                (*unsigned_x8).map(f64::to_bits),
                unsigned_x8_values.map(u64_to_f64_bits),
                "unsigned x8 lane order",
            );

            let signed_x8_values = [
                i64::MIN,
                i64::MIN + 512,
                -((1_i64 << 53) + 3),
                -(1_i64 << 32),
                (1_i64 << 32) - 1,
                (1_i64 << 53) + 3,
                i64::MAX - 511,
                i64::MAX,
            ];
            let signed_x8 = i64x8::from_slice(simd, &signed_x8_values).to_float::<f64x8<_>>();
            assert_eq!(
                (*signed_x8).map(f64::to_bits),
                signed_x8_values.map(i64_to_f64_bits),
                "signed x8 lane order",
            );
        },
    );
}

#[simd_test]
#[ignore = "checks uniform integers plus every rounding binade and tie parity"]
// Run with: cargo test --release cvt_f64_i64_u64_random -- --ignored
fn cvt_f64_i64_u64_random<S: Simd>(simd: S) {
    // Keep the oracle independent of LLVM's vectorized u64-to-f64 lowering.
    fn u64_to_f64_bits(value: u64) -> u64 {
        if value == 0 {
            return 0;
        }

        let fraction_mask = (1_u64 << 52) - 1;
        let mut exponent = 63 - value.leading_zeros();
        if exponent <= 52 {
            let fraction = (value << (52 - exponent)) & fraction_mask;
            return ((u64::from(exponent) + 1023) << 52) | fraction;
        }

        let shift = exponent - 52;
        let mut significand = value >> shift;
        let remainder = value & ((1_u64 << shift) - 1);
        let halfway = 1_u64 << (shift - 1);
        if remainder > halfway || (remainder == halfway && significand & 1 != 0) {
            significand += 1;
            if significand == 1_u64 << 53 {
                significand >>= 1;
                exponent += 1;
            }
        }

        ((u64::from(exponent) + 1023) << 52) | (significand & fraction_mask)
    }

    fn i64_to_f64_bits(value: i64) -> u64 {
        ((value < 0) as u64) << 63 | u64_to_f64_bits(value.unsigned_abs())
    }

    simd.vectorize(
        #[inline(always)]
        || {
            let mut rng = fastrand::Rng::with_seed(0x243f_6a88_85a3_08d3);

            for iteration in 0..2_500_000 {
                let signed = [rng.i64(..), rng.i64(..), rng.i64(..), rng.i64(..)];
                let expected_signed = signed.map(i64_to_f64_bits);
                let signed_x4 = i64x4::from_slice(simd, &signed).to_float::<f64x4<_>>();
                assert_eq!(
                    (*signed_x4).map(f64::to_bits),
                    expected_signed,
                    "signed x4 iteration {iteration}",
                );
                let signed_x2 = i64x2::from_slice(simd, &signed[..2]).to_float::<f64x2<_>>();
                assert_eq!(
                    (*signed_x2).map(f64::to_bits),
                    [expected_signed[0], expected_signed[1]],
                    "signed x2 iteration {iteration}",
                );

                let unsigned = [rng.u64(..), rng.u64(..), rng.u64(..), rng.u64(..)];
                let expected_unsigned = unsigned.map(u64_to_f64_bits);
                let unsigned_x4 = u64x4::from_slice(simd, &unsigned).to_float::<f64x4<_>>();
                assert_eq!(
                    (*unsigned_x4).map(f64::to_bits),
                    expected_unsigned,
                    "unsigned x4 iteration {iteration}",
                );
                let unsigned_x2 = u64x2::from_slice(simd, &unsigned[..2]).to_float::<f64x2<_>>();
                assert_eq!(
                    (*unsigned_x2).map(f64::to_bits),
                    [expected_unsigned[0], expected_unsigned[1]],
                    "unsigned x2 iteration {iteration}",
                );

                // Uniform integers mostly land in the largest one or two binades. Construct a
                // random retained significand in every inexact binade and force residues around
                // both an even and an odd halfway case.
                let unsigned_shift = 1 + iteration as u32 % 11;
                let unsigned_spacing = 1_u64 << unsigned_shift;
                let unsigned_halfway = unsigned_spacing / 2;
                let unsigned_significand =
                    ((1_u64 << 52) | (rng.u64(..) & ((1_u64 << 52) - 1))) & !1;
                let unsigned_even = unsigned_significand << unsigned_shift;
                let unsigned_odd = (unsigned_significand + 1) << unsigned_shift;
                let unsigned_targeted = [
                    unsigned_even + unsigned_halfway - 1,
                    unsigned_even + unsigned_halfway,
                    unsigned_even + unsigned_halfway + 1,
                    unsigned_odd + unsigned_halfway,
                ];
                let expected_unsigned_targeted = unsigned_targeted.map(u64_to_f64_bits);
                let result_unsigned_targeted =
                    u64x4::from_slice(simd, &unsigned_targeted).to_float::<f64x4<_>>();
                assert_eq!(
                    (*result_unsigned_targeted).map(f64::to_bits),
                    expected_unsigned_targeted,
                    "targeted unsigned iteration {iteration}, shift {unsigned_shift}",
                );
                let result_unsigned_targeted_x2_low =
                    u64x2::from_slice(simd, &unsigned_targeted[..2]).to_float::<f64x2<_>>();
                assert_eq!(
                    (*result_unsigned_targeted_x2_low).map(f64::to_bits),
                    [expected_unsigned_targeted[0], expected_unsigned_targeted[1]],
                    "targeted unsigned x2 low iteration {iteration}, shift {unsigned_shift}",
                );
                let result_unsigned_targeted_x2_high =
                    u64x2::from_slice(simd, &unsigned_targeted[2..]).to_float::<f64x2<_>>();
                assert_eq!(
                    (*result_unsigned_targeted_x2_high).map(f64::to_bits),
                    [expected_unsigned_targeted[2], expected_unsigned_targeted[3]],
                    "targeted unsigned x2 high iteration {iteration}, shift {unsigned_shift}",
                );

                let signed_shift = 1 + iteration as u32 % 10;
                let signed_spacing = 1_u64 << signed_shift;
                let signed_halfway = signed_spacing / 2;
                let signed_significand = ((1_u64 << 52) | (rng.u64(..) & ((1_u64 << 52) - 1))) & !1;
                let signed_even = (signed_significand << signed_shift) + signed_halfway;
                let signed_odd = ((signed_significand + 1) << signed_shift) + signed_halfway;
                let signed_targeted = [
                    signed_even as i64,
                    -(signed_even as i64),
                    signed_odd as i64,
                    -(signed_odd as i64),
                ];
                let expected_signed_targeted = signed_targeted.map(i64_to_f64_bits);
                let result_signed_targeted =
                    i64x4::from_slice(simd, &signed_targeted).to_float::<f64x4<_>>();
                assert_eq!(
                    (*result_signed_targeted).map(f64::to_bits),
                    expected_signed_targeted,
                    "targeted signed iteration {iteration}, shift {signed_shift}",
                );
                let result_signed_targeted_x2_low =
                    i64x2::from_slice(simd, &signed_targeted[..2]).to_float::<f64x2<_>>();
                assert_eq!(
                    (*result_signed_targeted_x2_low).map(f64::to_bits),
                    [expected_signed_targeted[0], expected_signed_targeted[1]],
                    "targeted signed x2 even iteration {iteration}, shift {signed_shift}",
                );
                let result_signed_targeted_x2_high =
                    i64x2::from_slice(simd, &signed_targeted[2..]).to_float::<f64x2<_>>();
                assert_eq!(
                    (*result_signed_targeted_x2_high).map(f64::to_bits),
                    [expected_signed_targeted[2], expected_signed_targeted[3]],
                    "targeted signed x2 odd iteration {iteration}, shift {signed_shift}",
                );
            }
        },
    );
}
