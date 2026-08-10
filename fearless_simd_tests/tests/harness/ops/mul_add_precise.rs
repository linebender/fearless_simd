// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// This file has a lot of coverage for 128-bit vectors specifically.
// That's because we have complex, error-prone emulation of precise FMA
// on SSE4.2 and need to test it in great depth.

#[simd_test]
#[ignore = "stress-tests 1 billion randomized safe-range and full-range f64 inputs"]
fn mul_add_precise_f64x2_random<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            let mut rng = fastrand::Rng::with_seed(0x3c6e_f372_fe94_f82b);
            let fraction_mask = (1_u64 << 52) - 1;

            for iteration in 0..1_000_000_000 {
                let safe_a: [f64; 2] = core::array::from_fn(|_| {
                    let exponent = rng.i32(-400..400);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let bits =
                        sign | (((exponent + 1023) as u64) << 52) | (rng.u64(..) & fraction_mask);
                    f64::from_bits(bits)
                });
                let safe_b: [f64; 2] = core::array::from_fn(|_| {
                    let exponent = rng.i32(-400..400);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let bits =
                        sign | (((exponent + 1023) as u64) << 52) | (rng.u64(..) & fraction_mask);
                    f64::from_bits(bits)
                });
                let safe_c: [f64; 2] = core::array::from_fn(|lane| {
                    if iteration % 32 == 0 && lane == 0 {
                        return if rng.bool() { 0.0 } else { -0.0 };
                    }
                    let exponent = rng.i32(-400..400);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let bits =
                        sign | (((exponent + 1023) as u64) << 52) | (rng.u64(..) & fraction_mask);
                    f64::from_bits(bits)
                });
                let safe_result = f64x2::from_slice(simd, &safe_a).mul_add_precise(
                    f64x2::from_slice(simd, &safe_b),
                    f64x2::from_slice(simd, &safe_c),
                );
                for lane in 0..2 {
                    let expected = safe_a[lane].mul_add(safe_b[lane], safe_c[lane]);
                    if expected.is_nan() {
                        assert!(
                            safe_result[lane].is_nan(),
                            "safe-range iteration {iteration}, lane {lane}: expected NaN, got {:?}",
                            safe_result[lane],
                        );
                    } else {
                        assert_eq!(
                            safe_result[lane].to_bits(),
                            expected.to_bits(),
                            "safe-range iteration {iteration}, lane {lane}",
                        );
                    }
                }

                let arbitrary_a = [f64::from_bits(rng.u64(..)), f64::from_bits(rng.u64(..))];
                let arbitrary_b = [f64::from_bits(rng.u64(..)), f64::from_bits(rng.u64(..))];
                let arbitrary_c = [f64::from_bits(rng.u64(..)), f64::from_bits(rng.u64(..))];
                let arbitrary_result = f64x2::from_slice(simd, &arbitrary_a).mul_add_precise(
                    f64x2::from_slice(simd, &arbitrary_b),
                    f64x2::from_slice(simd, &arbitrary_c),
                );
                for lane in 0..2 {
                    let expected = arbitrary_a[lane].mul_add(arbitrary_b[lane], arbitrary_c[lane]);
                    if expected.is_nan() {
                        assert!(
                            arbitrary_result[lane].is_nan(),
                            "full-range iteration {iteration}, lane {lane}: expected NaN, got {:?}",
                            arbitrary_result[lane],
                        );
                    } else {
                        assert_eq!(
                            arbitrary_result[lane].to_bits(),
                            expected.to_bits(),
                            "full-range iteration {iteration}, lane {lane}",
                        );
                    }
                }
            }
        },
    );
}

#[simd_test]
#[ignore = "stress-tests 1 billion safe-range, split-tie, and cancellation f64 inputs each"]
fn mul_add_precise_f64x2_adversarial_random<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            const CASES_PER_DISTRIBUTION: usize = 1_000_000_000;
            let fraction_mask = (1_u64 << 52) - 1;
            let mut rng = fastrand::Rng::with_seed(0x4d59_5df4_d0f3_3173);

            // Uniformly sample the exponent-safe packed path. Each vector supplies two cases.
            for vector_index in 0..CASES_PER_DISTRIBUTION / 2 {
                let a_values: [f64; 2] = core::array::from_fn(|_| {
                    let exponent = rng.i32(-400..400);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let bits =
                        sign | (((exponent + 1023) as u64) << 52) | (rng.u64(..) & fraction_mask);
                    f64::from_bits(bits)
                });
                let b_values: [f64; 2] = core::array::from_fn(|_| {
                    let exponent = rng.i32(-400..400);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let bits =
                        sign | (((exponent + 1023) as u64) << 52) | (rng.u64(..) & fraction_mask);
                    f64::from_bits(bits)
                });
                let c_values: [f64; 2] = core::array::from_fn(|_| {
                    let exponent = rng.i32(-400..400);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let bits =
                        sign | (((exponent + 1023) as u64) << 52) | (rng.u64(..) & fraction_mask);
                    f64::from_bits(bits)
                });
                let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
                    f64x2::from_slice(simd, &b_values),
                    f64x2::from_slice(simd, &c_values),
                );

                for lane in 0..2 {
                    let expected = a_values[lane].mul_add(b_values[lane], c_values[lane]);
                    assert_eq!(
                        result[lane].to_bits(),
                        expected.to_bits(),
                        "safe-range case {}: a={:#018x}, b={:#018x}, c={:#018x}",
                        vector_index * 2 + lane,
                        a_values[lane].to_bits(),
                        b_values[lane].to_bits(),
                        c_values[lane].to_bits(),
                    );
                }
            }

            // Put both multiplicands exactly halfway across the integer-mask split boundary.
            // Random binary64 bit patterns almost never exercise this rounding case.
            let fraction_above_split_mask = 0x000f_ffff_f800_0000_u64;
            let split_rounding_bit = 1_u64 << 26;
            for vector_index in 0..CASES_PER_DISTRIBUTION / 2 {
                let a_values: [f64; 2] = core::array::from_fn(|_| {
                    let exponent = rng.i32(-400..400);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let fraction = (rng.u64(..) & fraction_above_split_mask) | split_rounding_bit;
                    f64::from_bits(sign | (((exponent + 1023) as u64) << 52) | fraction)
                });
                let b_values: [f64; 2] = core::array::from_fn(|_| {
                    let exponent = rng.i32(-400..400);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let fraction = (rng.u64(..) & fraction_above_split_mask) | split_rounding_bit;
                    f64::from_bits(sign | (((exponent + 1023) as u64) << 52) | fraction)
                });
                let c_values: [f64; 2] = core::array::from_fn(|_| {
                    let exponent = rng.i32(-400..400);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let bits =
                        sign | (((exponent + 1023) as u64) << 52) | (rng.u64(..) & fraction_mask);
                    f64::from_bits(bits)
                });
                let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
                    f64x2::from_slice(simd, &b_values),
                    f64x2::from_slice(simd, &c_values),
                );

                for lane in 0..2 {
                    let expected = a_values[lane].mul_add(b_values[lane], c_values[lane]);
                    assert_eq!(
                        result[lane].to_bits(),
                        expected.to_bits(),
                        "split-tie case {}: a={:#018x}, b={:#018x}, c={:#018x}",
                        vector_index * 2 + lane,
                        a_values[lane].to_bits(),
                        b_values[lane].to_bits(),
                        c_values[lane].to_bits(),
                    );
                }
            }

            // Cancelling the rounded product exposes any error in the reconstructed exact tail.
            let safe_minimum_bits = 623_u64 << 52;
            let safe_maximum_bits = 1423_u64 << 52;
            for vector_index in 0..CASES_PER_DISTRIBUTION / 2 {
                let mut a_values = [0.0; 2];
                let mut b_values = [0.0; 2];
                let mut c_values = [0.0; 2];
                for lane in 0..2 {
                    loop {
                        let a_exponent = rng.i32(-400..400);
                        let a_sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                        let a_bits = a_sign
                            | (((a_exponent + 1023) as u64) << 52)
                            | (rng.u64(..) & fraction_mask);
                        let b_exponent = rng.i32(-400..400);
                        let b_sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                        let b_bits = b_sign
                            | (((b_exponent + 1023) as u64) << 52)
                            | (rng.u64(..) & fraction_mask);
                        let a = f64::from_bits(a_bits);
                        let b = f64::from_bits(b_bits);
                        let rounded_product = a * b;
                        let product_bits = rounded_product.to_bits() & i64::MAX as u64;
                        if product_bits >= safe_minimum_bits && product_bits <= safe_maximum_bits {
                            a_values[lane] = a;
                            b_values[lane] = b;
                            c_values[lane] = -rounded_product;
                            break;
                        }
                    }
                }
                let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
                    f64x2::from_slice(simd, &b_values),
                    f64x2::from_slice(simd, &c_values),
                );

                for lane in 0..2 {
                    let expected = a_values[lane].mul_add(b_values[lane], c_values[lane]);
                    assert_eq!(
                        result[lane].to_bits(),
                        expected.to_bits(),
                        "cancellation case {}: a={:#018x}, b={:#018x}, c={:#018x}",
                        vector_index * 2 + lane,
                        a_values[lane].to_bits(),
                        b_values[lane].to_bits(),
                        c_values[lane].to_bits(),
                    );
                }
            }
        },
    );
}

#[simd_test]
fn mul_add_precise_f64x2_edge_patterns<S: Simd>(simd: S) {
    let edge_bits = [
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x8000_0000_0000_0001,
        0x000f_ffff_ffff_ffff,
        0x800f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x8010_0000_0000_0000,
        (623_u64 << 52) - 1,
        (1_u64 << 63) | ((623_u64 << 52) - 1),
        623_u64 << 52,
        (1_u64 << 63) | (623_u64 << 52),
        (623_u64 << 52) + 1,
        (1_u64 << 63) | ((623_u64 << 52) + 1),
        0x3fef_ffff_ffff_ffff,
        0xbfef_ffff_ffff_ffff,
        0x3ff0_0000_0000_0000,
        0xbff0_0000_0000_0000,
        0x3ff0_0000_0000_0001,
        0xbff0_0000_0000_0001,
        (1423_u64 << 52) - 1,
        (1_u64 << 63) | ((1423_u64 << 52) - 1),
        1423_u64 << 52,
        (1_u64 << 63) | (1423_u64 << 52),
        (1423_u64 << 52) + 1,
        (1_u64 << 63) | ((1423_u64 << 52) + 1),
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0xfff8_0000_0000_1234,
        0x7ff0_0000_0000_0001,
        0xfff0_0000_0000_0001,
    ];

    let mut pending_a = [0.0; 2];
    let mut pending_b = [0.0; 2];
    let mut pending_c = [0.0; 2];
    let mut pending_lanes = 0;
    let mut case_index = 0;
    for &a_bits in &edge_bits {
        for &b_bits in &edge_bits {
            for &c_bits in &edge_bits {
                pending_a[pending_lanes] = f64::from_bits(a_bits);
                pending_b[pending_lanes] = f64::from_bits(b_bits);
                pending_c[pending_lanes] = f64::from_bits(c_bits);
                pending_lanes += 1;

                if pending_lanes == 2 {
                    let result = f64x2::from_slice(simd, &pending_a).mul_add_precise(
                        f64x2::from_slice(simd, &pending_b),
                        f64x2::from_slice(simd, &pending_c),
                    );
                    for lane in 0..2 {
                        let expected = pending_a[lane].mul_add(pending_b[lane], pending_c[lane]);
                        if expected.is_nan() {
                            // Rust does not specify the payload or signalingness of arithmetic NaNs.
                            assert!(
                                result[lane].is_nan(),
                                "edge case {}, lane {lane}: a={:#018x}, b={:#018x}, c={:#018x}",
                                case_index + lane,
                                pending_a[lane].to_bits(),
                                pending_b[lane].to_bits(),
                                pending_c[lane].to_bits(),
                            );
                        } else {
                            assert_eq!(
                                result[lane].to_bits(),
                                expected.to_bits(),
                                "edge case {}, lane {lane}: a={:#018x}, b={:#018x}, c={:#018x}",
                                case_index + lane,
                                pending_a[lane].to_bits(),
                                pending_b[lane].to_bits(),
                                pending_c[lane].to_bits(),
                            );
                        }
                    }
                    pending_lanes = 0;
                    case_index += 2;
                }
            }
        }
    }
    assert_eq!(pending_lanes, 0);

    // These cases exercise a carry from the retained split significand into the exponent,
    // including the endpoints of the exponent-safe packed path.
    let fraction_mask = (1_u64 << 52) - 1;
    let upper_fraction_ones = fraction_mask & !((1_u64 << 27) - 1);
    let exponent_fields = [623_u64, 624, 1022, 1023, 1024, 1422, 1423];
    let low_split_bits = [
        0,
        1,
        (1_u64 << 26) - 1,
        1_u64 << 26,
        (1_u64 << 26) + 1,
        (1_u64 << 27) - 2,
        (1_u64 << 27) - 1,
        0x0555_5555,
    ];
    let multipliers = [
        f64::from_bits(0x3ff0_0000_0000_0001),
        f64::from_bits(0x3fef_ffff_ffff_ffff),
        f64::from_bits(0xbff0_0000_0400_0000),
        f64::from_bits(0x4008_0000_0000_0001),
    ];

    for exponent_field in exponent_fields {
        for sign in [0, 1_u64 << 63] {
            for low_bits in low_split_bits {
                let a_value =
                    f64::from_bits(sign | (exponent_field << 52) | upper_fraction_ones | low_bits);
                for b_value in multipliers {
                    let rounded_product = a_value * b_value;
                    let c_values = [
                        0.0,
                        -0.0,
                        -rounded_product,
                        if rounded_product == f64::NEG_INFINITY {
                            f64::MAX
                        } else if rounded_product == 0.0 {
                            f64::from_bits(1)
                        } else {
                            let bits = (-rounded_product).to_bits();
                            f64::from_bits(if -rounded_product > 0.0 {
                                bits + 1
                            } else {
                                bits - 1
                            })
                        },
                        if rounded_product == f64::INFINITY {
                            -f64::MAX
                        } else if rounded_product == 0.0 {
                            f64::from_bits((1_u64 << 63) | 1)
                        } else {
                            let bits = (-rounded_product).to_bits();
                            f64::from_bits(if -rounded_product > 0.0 {
                                bits - 1
                            } else {
                                bits + 1
                            })
                        },
                    ];

                    for c_value in c_values {
                        let a_values = [a_value, -a_value];
                        let b_values = [b_value, b_value];
                        let c_values = [c_value, -c_value];
                        let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
                            f64x2::from_slice(simd, &b_values),
                            f64x2::from_slice(simd, &c_values),
                        );
                        for lane in 0..2 {
                            let expected = a_values[lane].mul_add(b_values[lane], c_values[lane]);
                            if expected.is_nan() {
                                assert!(result[lane].is_nan());
                            } else {
                                assert_eq!(
                                    result[lane].to_bits(),
                                    expected.to_bits(),
                                    "split carry: a={:#018x}, b={:#018x}, c={:#018x}, lane {lane}",
                                    a_values[lane].to_bits(),
                                    b_values[lane].to_bits(),
                                    c_values[lane].to_bits(),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // Fixed cases found while auditing the integer splitter and deferred correction path.
    // They are deliberately written as exact bits so future failures are easy to reproduce.
    let regression_cases = [
        (
            0x3ff0_0000_0200_0000,
            0x3fef_ffff_fc00_0000,
            0xb930_0000_0000_0000,
            0x3fef_ffff_ffff_ffff,
        ),
        (
            0x3ff0_0000_0200_0000,
            0x3fef_ffff_fc00_0000,
            0x3930_0000_0000_0000,
            0x3ff0_0000_0000_0000,
        ),
        (
            0x58ef_ffff_fc00_0000,
            0x26f0_0000_0400_0000,
            0xbff0_0000_0200_0000,
            0xbca0_0000_0000_0000,
        ),
        (
            0x58ef_ffff_fc00_0000,
            0x26f0_0000_0400_0000,
            0xbff0_0000_01ff_ffff,
            0x3ca0_0000_0000_0000,
        ),
        (
            0x58ef_ffff_fc00_0000,
            0x26f0_0000_0400_0000,
            0xbff0_0000_0200_0001,
            0xbcb8_0000_0000_0000,
        ),
        (
            0x26f0_0000_0400_0000,
            0x26f0_0000_0400_0000,
            0x8df0_0000_0800_0001,
            0x0000_0000_0000_0000,
        ),
        (
            0x26f0_0000_0400_0000,
            0x26f0_0000_0400_0000,
            0x8df0_0000_0800_0000,
            0x0ab0_0000_0000_0000,
        ),
        (
            0x26f0_0000_0400_0000,
            0x26f0_0000_0400_0000,
            0x8df0_0000_0800_0002,
            0x8ab0_0000_0000_0000,
        ),
        (
            0x58ef_ffff_fc00_0000,
            0x3ff0_0000_0400_0000,
            0xd8f0_0000_0200_0000,
            0xd5a0_0000_0000_0000,
        ),
        (
            0x58ef_ffff_fc00_0000,
            0x3ff0_0000_0400_0000,
            0xd8f0_0000_01ff_ffff,
            0x55a0_0000_0000_0000,
        ),
        (
            0x58ef_ffff_fc00_0000,
            0x3ff0_0000_0400_0000,
            0xd8f0_0000_0200_0001,
            0xd5b8_0000_0000_0000,
        ),
    ];

    for (a_bits, b_bits, c_bits, expected_bits) in regression_cases {
        let a_value = f64::from_bits(a_bits);
        let b_value = f64::from_bits(b_bits);
        let c_value = f64::from_bits(c_bits);
        assert_eq!(a_value.mul_add(b_value, c_value).to_bits(), expected_bits);

        let result = f64x2::splat(simd, a_value)
            .mul_add_precise(f64x2::splat(simd, b_value), f64x2::splat(simd, c_value));
        assert_eq!(result[0].to_bits(), expected_bits);
        assert_eq!(result[1].to_bits(), expected_bits);
    }
}

#[simd_test]
#[ignore = "stress-tests 1 billion split-boundary and near-cancellation f64 inputs"]
fn mul_add_precise_f64x2_random_split_boundaries<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            const CASES: usize = 1_000_000_000;
            const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
            const LOW_SPLIT_BITS: [u64; 10] = [
                0,
                1,
                (1 << 26) - 1,
                1 << 26,
                (1 << 26) + 1,
                (1 << 27) - 3,
                (1 << 27) - 2,
                (1 << 27) - 1,
                0x02aa_aaaa,
                0x0555_5555,
            ];
            let mut rng = fastrand::Rng::with_seed(0x1319_8a2e_0370_7344);

            for vector_index in 0..CASES / 2 {
                let mode = vector_index % 5;
                let a_values: [f64; 2] = core::array::from_fn(|_| {
                    let exponent = if mode == 0 {
                        rng.i32(-400..401)
                    } else {
                        rng.i32(-190..191)
                    };
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let upper_fraction = rng.u64(..) & (FRACTION_MASK & !((1_u64 << 27) - 1));
                    let low = LOW_SPLIT_BITS[rng.usize(..LOW_SPLIT_BITS.len())];
                    f64::from_bits(sign | (((exponent + 1023) as u64) << 52) | upper_fraction | low)
                });
                let b_values: [f64; 2] = core::array::from_fn(|_| {
                    let exponent = if mode == 0 {
                        rng.i32(-400..401)
                    } else {
                        rng.i32(-190..191)
                    };
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    let upper_fraction = rng.u64(..) & (FRACTION_MASK & !((1_u64 << 27) - 1));
                    let low = LOW_SPLIT_BITS[rng.usize(..LOW_SPLIT_BITS.len())];
                    f64::from_bits(sign | (((exponent + 1023) as u64) << 52) | upper_fraction | low)
                });
                let rounded_products = [a_values[0] * b_values[0], a_values[1] * b_values[1]];
                let c_values: [f64; 2] = core::array::from_fn(|lane| match mode {
                    0 => {
                        let exponent = rng.i32(-400..401);
                        let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                        f64::from_bits(
                            sign | (((exponent + 1023) as u64) << 52)
                                | (rng.u64(..) & FRACTION_MASK),
                        )
                    }
                    1 => -rounded_products[lane],
                    2 => {
                        let value = -rounded_products[lane];
                        let bits = value.to_bits();
                        f64::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
                    }
                    3 => {
                        let value = -rounded_products[lane];
                        let bits = value.to_bits();
                        f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
                    }
                    4 => {
                        let value = -rounded_products[lane];
                        let bits = value.to_bits();
                        if lane == 0 {
                            f64::from_bits(if value > 0.0 { bits + 2 } else { bits - 2 })
                        } else {
                            f64::from_bits(if value > 0.0 { bits - 2 } else { bits + 2 })
                        }
                    }
                    _ => unreachable!(),
                });

                let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
                    f64x2::from_slice(simd, &b_values),
                    f64x2::from_slice(simd, &c_values),
                );
                for lane in 0..2 {
                    let expected = a_values[lane].mul_add(b_values[lane], c_values[lane]);
                    assert_eq!(
                        result[lane].to_bits(),
                        expected.to_bits(),
                        "split-boundary case {}, lane {lane}: a={:#018x}, b={:#018x}, c={:#018x}",
                        vector_index * 2 + lane,
                        a_values[lane].to_bits(),
                        b_values[lane].to_bits(),
                        c_values[lane].to_bits(),
                    );
                }
            }
        },
    );
}

#[simd_test]
#[ignore = "stress-tests 1 billion forced rare-correction f64 inputs"]
fn mul_add_precise_f64x2_random_rare_correction<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            const CASES: usize = 1_000_000_000;
            const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
            let mut rng = fastrand::Rng::with_seed(0x082e_fa98_ec4e_6c89);

            for vector_index in 0..CASES / 2 {
                // These products have exact tails of -2^-54 and -3*2^-56. They force the
                // power-of-two and three-times-power-of-two significand shapes for which the
                // residual of the second Fast2Sum can change the final rounding.
                let (base_a, base_b, tiny_offset) = if vector_index % 2 == 0 {
                    (
                        f64::from_bits(0x3ff0_0000_0200_0000),
                        f64::from_bits(0x3fef_ffff_fc00_0000),
                        108,
                    )
                } else {
                    (1.0 + 3.0 * 2.0_f64.powi(-28), 1.0 - 2.0_f64.powi(-28), 110)
                };
                let product_exponent = rng.i32(-150..151);
                let a_exponent = rng.i32(-100..101);
                let b_exponent = product_exponent - a_exponent;

                let scale = |value: f64, exponent: i32| {
                    let bits = value.to_bits();
                    let exponent_field = ((bits >> 52) & 0x7ff) as i32;
                    f64::from_bits(
                        (((exponent_field + exponent) as u64) << 52) | (bits & FRACTION_MASK),
                    )
                };
                let a_value = scale(base_a, a_exponent);
                let b_value = scale(base_b, b_exponent);
                let tiny = f64::from_bits(((product_exponent - tiny_offset + 1023) as u64) << 52);
                let c_value = if rng.bool() { tiny } else { -tiny };
                let a_values = [a_value, -a_value];
                let b_values = [b_value, b_value];
                let c_values = [c_value, c_value];
                let expected = [
                    a_values[0].mul_add(b_values[0], c_values[0]),
                    a_values[1].mul_add(b_values[1], c_values[1]),
                ];

                if vector_index % 2 == 0 {
                    // The power-of-two-tail cases are true midpoint corrections: one lane must
                    // distinguish fused evaluation from a rounded multiply followed by an add.
                    assert!(
                        expected[0].to_bits()
                            != (a_values[0] * b_values[0] + c_values[0]).to_bits()
                            || expected[1].to_bits()
                                != (a_values[1] * b_values[1] + c_values[1]).to_bits(),
                        "case {} must distinguish fused from unfused evaluation",
                        vector_index * 2,
                    );
                }

                let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
                    f64x2::from_slice(simd, &b_values),
                    f64x2::from_slice(simd, &c_values),
                );
                for lane in 0..2 {
                    assert_eq!(
                        result[lane].to_bits(),
                        expected[lane].to_bits(),
                        "rare-correction case {}, lane {lane}: a={:#018x}, b={:#018x}, c={:#018x}",
                        vector_index * 2 + lane,
                        a_values[lane].to_bits(),
                        b_values[lane].to_bits(),
                        c_values[lane].to_bits(),
                    );
                }
            }
        },
    );
}

#[simd_test]
#[ignore = "stress-tests 1 billion mixed packed-path and scalar-fallback f64 lanes"]
fn mul_add_precise_f64x2_random_mixed_safe_and_unsafe<S: Simd>(simd: S) {
    simd.vectorize(
        #[inline(always)]
        || {
            const CASES: usize = 1_000_000_000;
            const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
            const UNSAFE_BITS: [u64; 20] = [
                0x0000_0000_0000_0000,
                0x8000_0000_0000_0000,
                0x0000_0000_0000_0001,
                0x8000_0000_0000_0001,
                0x000f_ffff_ffff_ffff,
                0x800f_ffff_ffff_ffff,
                0x0010_0000_0000_0000,
                0x8010_0000_0000_0000,
                (623_u64 << 52) - 1,
                (1_u64 << 63) | ((623_u64 << 52) - 1),
                (1423_u64 << 52) + 1,
                (1_u64 << 63) | ((1423_u64 << 52) + 1),
                0x7fef_ffff_ffff_ffff,
                0xffef_ffff_ffff_ffff,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x7ff8_0000_0000_0042,
                0xfff8_0000_0000_0042,
                0x7ff0_0000_0000_0042,
                0xfff0_0000_0000_0042,
            ];
            let mut rng = fastrand::Rng::with_seed(0xbe54_66cf_34e9_0c6c);

            for vector_index in 0..CASES / 2 {
                let safe_a = {
                    let exponent = rng.i32(-400..401);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    f64::from_bits(
                        sign | (((exponent + 1023) as u64) << 52) | (rng.u64(..) & FRACTION_MASK),
                    )
                };
                let safe_b = {
                    let exponent = rng.i32(-400..401);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    f64::from_bits(
                        sign | (((exponent + 1023) as u64) << 52) | (rng.u64(..) & FRACTION_MASK),
                    )
                };
                let safe_c = {
                    let exponent = rng.i32(-400..401);
                    let sign = if rng.bool() { 1_u64 << 63 } else { 0 };
                    f64::from_bits(
                        sign | (((exponent + 1023) as u64) << 52) | (rng.u64(..) & FRACTION_MASK),
                    )
                };
                let unsafe_a = f64::from_bits(UNSAFE_BITS[rng.usize(..UNSAFE_BITS.len())]);
                let unsafe_b = f64::from_bits(UNSAFE_BITS[rng.usize(..UNSAFE_BITS.len())]);
                let unsafe_c = f64::from_bits(UNSAFE_BITS[rng.usize(..UNSAFE_BITS.len())]);
                let (a_values, b_values, c_values) = if vector_index % 2 == 0 {
                    ([safe_a, unsafe_a], [safe_b, unsafe_b], [safe_c, unsafe_c])
                } else {
                    ([unsafe_a, safe_a], [unsafe_b, safe_b], [unsafe_c, safe_c])
                };

                let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
                    f64x2::from_slice(simd, &b_values),
                    f64x2::from_slice(simd, &c_values),
                );
                for lane in 0..2 {
                    let expected = a_values[lane].mul_add(b_values[lane], c_values[lane]);
                    if expected.is_nan() {
                        assert!(
                            result[lane].is_nan(),
                            "mixed-lane case {}, lane {lane}: a={:#018x}, b={:#018x}, c={:#018x}",
                            vector_index * 2 + lane,
                            a_values[lane].to_bits(),
                            b_values[lane].to_bits(),
                            c_values[lane].to_bits(),
                        );
                    } else {
                        assert_eq!(
                            result[lane].to_bits(),
                            expected.to_bits(),
                            "mixed-lane case {}, lane {lane}: a={:#018x}, b={:#018x}, c={:#018x}",
                            vector_index * 2 + lane,
                            a_values[lane].to_bits(),
                            b_values[lane].to_bits(),
                            c_values[lane].to_bits(),
                        );
                    }
                }
            }
        },
    );
}

#[simd_test]
fn mul_add_precise_f32x4<S: Simd>(simd: S) {
    let midpoint_a = f32::from_bits(0x3f80_1000); // 1 + 2^-11
    let midpoint_b = f32::from_bits(0x3f7f_f800); // 1 - 2^-13
    let tiny = f32::from_bits(0x0d80_0000); // 2^-100
    let a_values = [1.0 + f32::EPSILON, midpoint_a, midpoint_a, f32::MAX];
    let b_values = [1.0 - f32::EPSILON, midpoint_b, midpoint_b, 2.0];
    let c_values = [-1.0, tiny, -tiny, -f32::MAX];
    let expected = [
        a_values[0].mul_add(b_values[0], c_values[0]),
        a_values[1].mul_add(b_values[1], c_values[1]),
        a_values[2].mul_add(b_values[2], c_values[2]),
        a_values[3].mul_add(b_values[3], c_values[3]),
    ];

    let a = f32x4::from_slice(simd, &a_values);
    let b = f32x4::from_slice(simd, &b_values);
    let c = f32x4::from_slice(simd, &c_values);
    assert_eq!(*a.mul_add_precise(b, c), expected);

    // The negative tiny addend moves the exact result below an f32 midpoint, but is lost by a
    // naive f64 add before narrowing.
    assert_ne!(
        expected[2],
        ((midpoint_a as f64) * (midpoint_b as f64) - (tiny as f64)) as f32
    );
}

#[simd_test]
fn mul_add_precise_f32x4_overflow_rounding<S: Simd>(simd: S) {
    // a*b is exactly the overflow midpoint 2^128 - 2^103. The much smaller addend is
    // lost by the rounded f64 addition, but decides whether the exact FMA is finite.
    let a_value = f32::from_bits(0x5ff8_0000);
    let b_value = f32::from_bits(0x5f04_2108);
    let delta = f32::from_bits(0x6280_0000); // 2^70
    let a = f32x4::from_slice(simd, &[a_value, a_value, -a_value, -a_value]);
    let b = f32x4::splat(simd, b_value);
    let c = f32x4::from_slice(simd, &[delta, -delta, -delta, delta]);
    let actual = a.mul_add_precise(b, c);
    let actual_bits = [
        actual[0].to_bits(),
        actual[1].to_bits(),
        actual[2].to_bits(),
        actual[3].to_bits(),
    ];

    assert_eq!(
        actual_bits,
        [
            f32::INFINITY.to_bits(),
            f32::MAX.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            (-f32::MAX).to_bits(),
        ]
    );
}

#[simd_test]
fn mul_add_precise_f32x4_special_values<S: Simd>(simd: S) {
    let a_values = [-0.0, f32::MIN_POSITIVE, f32::INFINITY, f32::NAN];
    let b_values = [2.0, 0.5, 2.0, 1.0];
    let c_values = [-0.0, 0.0, f32::NEG_INFINITY, 1.0];
    let a = f32x4::from_slice(simd, &a_values);
    let b = f32x4::from_slice(simd, &b_values);
    let c = f32x4::from_slice(simd, &c_values);
    let result = a.mul_add_precise(b, c);

    assert_eq!(
        result[0].to_bits(),
        a_values[0].mul_add(b_values[0], c_values[0]).to_bits()
    );
    assert_eq!(
        result[1].to_bits(),
        a_values[1].mul_add(b_values[1], c_values[1]).to_bits()
    );
    assert!(result[2].is_nan());
    assert!(result[3].is_nan());
}

#[simd_test]
fn mul_add_precise_f64x2<S: Simd>(simd: S) {
    let a_values = [1.0 + f64::EPSILON, f64::MAX];
    let b_values = [1.0 - f64::EPSILON, 2.0];
    let c_values = [-1.0, -f64::MAX];
    let expected = [
        a_values[0].mul_add(b_values[0], c_values[0]),
        a_values[1].mul_add(b_values[1], c_values[1]),
    ];

    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let c = f64x2::from_slice(simd, &c_values);
    assert_eq!(*a.mul_add_precise(b, c), expected);
}

#[simd_test]
fn mul_add_precise_f64x2_midpoint<S: Simd>(simd: S) {
    let midpoint_a = 1.0 + 2.0_f64.powi(-27);
    let midpoint_b = 1.0 - 2.0_f64.powi(-27);
    let tiny = 2.0_f64.powi(-150);
    let a_values = [midpoint_a, midpoint_a];
    let b_values = [midpoint_b, midpoint_b];
    let c_values = [tiny, -tiny];
    let expected = [
        a_values[0].mul_add(b_values[0], c_values[0]),
        a_values[1].mul_add(b_values[1], c_values[1]),
    ];

    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let c = f64x2::from_slice(simd, &c_values);
    assert_eq!(*a.mul_add_precise(b, c), expected);

    assert_ne!(
        expected[1],
        midpoint_a * midpoint_b - tiny,
        "the negative perturbation must differ from a naive multiply-add",
    );
}

#[simd_test]
fn mul_add_precise_f64x2_split_rounding_regression<S: Simd>(simd: S) {
    // Clearing the low split bits without first rounding them loses two ULPs of the
    // exact product residual for this cancellation. Exercise both result signs.
    let a_values = [
        f64::from_bits(0xc168_bab3_2de0_0aeb),
        f64::from_bits(0x4168_bab3_2de0_0aeb),
    ];
    let b_values = [
        f64::from_bits(0x4f79_aedc_867f_cdc1),
        f64::from_bits(0x4f79_aedc_867f_cdc1),
    ];
    let c_values = [
        f64::from_bits(0x50f3_d8fd_95a0_e870),
        f64::from_bits(0xd0f3_d8fd_95a0_e870),
    ];
    let expected = [
        a_values[0].mul_add(b_values[0], c_values[0]),
        a_values[1].mul_add(b_values[1], c_values[1]),
    ];
    assert_eq!(expected[0].to_bits(), 0x4d95_3ebb_2989_2baa);
    assert_eq!(expected[1].to_bits(), 0xcd95_3ebb_2989_2baa);

    let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
        f64x2::from_slice(simd, &b_values),
        f64x2::from_slice(simd, &c_values),
    );
    assert_eq!(result[0].to_bits(), expected[0].to_bits());
    assert_eq!(result[1].to_bits(), expected[1].to_bits());
}

#[simd_test]
fn mul_add_precise_f64x2_special_values<S: Simd>(simd: S) {
    let a_values = [-0.0, f64::INFINITY];
    let b_values = [2.0, 2.0];
    let c_values = [-0.0, f64::NEG_INFINITY];
    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let c = f64x2::from_slice(simd, &c_values);
    let result = a.mul_add_precise(b, c);

    assert_eq!(
        result[0].to_bits(),
        a_values[0].mul_add(b_values[0], c_values[0]).to_bits()
    );
    assert!(result[1].is_nan());
}

#[simd_test]
fn mul_add_precise_f64x2_safe_range_boundaries<S: Simd>(simd: S) {
    let safe_minimum = f64::from_bits(623_u64 << 52);
    let below_safe_minimum = f64::from_bits((623_u64 << 52) - 1);
    let safe_maximum = f64::from_bits(1423_u64 << 52);
    let above_safe_maximum = f64::from_bits((1423_u64 << 52) + 1);

    let a_values = [safe_minimum, safe_maximum];
    let b_values = [safe_maximum, safe_minimum];
    let c_values = [-0.0, safe_minimum];
    let expected = [
        a_values[0].mul_add(b_values[0], c_values[0]),
        a_values[1].mul_add(b_values[1], c_values[1]),
    ];
    let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
        f64x2::from_slice(simd, &b_values),
        f64x2::from_slice(simd, &c_values),
    );
    assert_eq!(result[0].to_bits(), expected[0].to_bits());
    assert_eq!(result[1].to_bits(), expected[1].to_bits());

    let a_values = [below_safe_minimum, above_safe_maximum];
    let b_values = [1.0, 1.0];
    let c_values = [safe_minimum, -safe_maximum];
    let expected = [
        a_values[0].mul_add(b_values[0], c_values[0]),
        a_values[1].mul_add(b_values[1], c_values[1]),
    ];
    let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
        f64x2::from_slice(simd, &b_values),
        f64x2::from_slice(simd, &c_values),
    );
    assert_eq!(result[0].to_bits(), expected[0].to_bits());
    assert_eq!(result[1].to_bits(), expected[1].to_bits());
}

#[simd_test]
fn mul_add_precise_f64x2_mixed_safe_and_unsafe<S: Simd>(simd: S) {
    let midpoint_a = 1.0 + 2.0_f64.powi(-27);
    let midpoint_b = 1.0 - 2.0_f64.powi(-27);
    let tiny = 2.0_f64.powi(-150);
    let a_values = [midpoint_a, f64::from_bits(1)];
    let b_values = [midpoint_b, f64::MAX];
    let c_values = [-tiny, -f64::MAX];
    let expected = [
        a_values[0].mul_add(b_values[0], c_values[0]),
        a_values[1].mul_add(b_values[1], c_values[1]),
    ];

    let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
        f64x2::from_slice(simd, &b_values),
        f64x2::from_slice(simd, &c_values),
    );
    assert_eq!(result[0].to_bits(), expected[0].to_bits());
    assert_eq!(result[1].to_bits(), expected[1].to_bits());
}

#[simd_test]
fn mul_add_precise_f64x2_full_range_fallback<S: Simd>(simd: S) {
    let cases = [
        ([0.0, -0.0], [-2.0, 2.0], [-0.0, 0.0]),
        (
            [f64::from_bits(1), f64::MIN_POSITIVE],
            [0.5, 0.5],
            [f64::from_bits(1), -f64::from_bits(1)],
        ),
        (
            [f64::MAX, f64::INFINITY],
            [2.0, 2.0],
            [-f64::MAX, f64::NEG_INFINITY],
        ),
        ([f64::NAN, 1.0], [1.0, f64::NAN], [2.0, 3.0]),
    ];

    for (a_values, b_values, c_values) in cases {
        let result = f64x2::from_slice(simd, &a_values).mul_add_precise(
            f64x2::from_slice(simd, &b_values),
            f64x2::from_slice(simd, &c_values),
        );
        for lane in 0..2 {
            let expected = a_values[lane].mul_add(b_values[lane], c_values[lane]);
            if expected.is_nan() {
                assert!(result[lane].is_nan());
            } else {
                assert_eq!(result[lane].to_bits(), expected.to_bits());
            }
        }
    }
}

#[simd_test]
fn mul_add_precise_f32x8<S: Simd>(simd: S) {
    let midpoint_a = f32::from_bits(0x3f80_1000);
    let midpoint_b = f32::from_bits(0x3f7f_f800);
    let tiny = f32::from_bits(0x0d80_0000);
    let a_values = [
        1.0 + f32::EPSILON,
        midpoint_a,
        midpoint_a,
        f32::MAX,
        -0.0,
        f32::MIN_POSITIVE,
        3.0,
        -7.0,
    ];
    let b_values = [
        1.0 - f32::EPSILON,
        midpoint_b,
        midpoint_b,
        2.0,
        2.0,
        0.5,
        4.0,
        5.0,
    ];
    let c_values = [-1.0, tiny, -tiny, -f32::MAX, -0.0, 0.0, 0.5, 2.0];
    let expected = [
        a_values[0].mul_add(b_values[0], c_values[0]),
        a_values[1].mul_add(b_values[1], c_values[1]),
        a_values[2].mul_add(b_values[2], c_values[2]),
        a_values[3].mul_add(b_values[3], c_values[3]),
        a_values[4].mul_add(b_values[4], c_values[4]),
        a_values[5].mul_add(b_values[5], c_values[5]),
        a_values[6].mul_add(b_values[6], c_values[6]),
        a_values[7].mul_add(b_values[7], c_values[7]),
    ];

    let a = f32x8::from_slice(simd, &a_values);
    let b = f32x8::from_slice(simd, &b_values);
    let c = f32x8::from_slice(simd, &c_values);
    assert_eq!(*a.mul_add_precise(b, c), expected);
}

#[simd_test]
fn mul_add_precise_f64x4<S: Simd>(simd: S) {
    let midpoint_a = 1.0 + 2.0_f64.powi(-27);
    let midpoint_b = 1.0 - 2.0_f64.powi(-27);
    let tiny = 2.0_f64.powi(-150);
    let a_values = [1.0 + f64::EPSILON, f64::MAX, midpoint_a, midpoint_a];
    let b_values = [1.0 - f64::EPSILON, 2.0, midpoint_b, midpoint_b];
    let c_values = [-1.0, -f64::MAX, tiny, -tiny];
    let expected = [
        a_values[0].mul_add(b_values[0], c_values[0]),
        a_values[1].mul_add(b_values[1], c_values[1]),
        a_values[2].mul_add(b_values[2], c_values[2]),
        a_values[3].mul_add(b_values[3], c_values[3]),
    ];

    let a = f64x4::from_slice(simd, &a_values);
    let b = f64x4::from_slice(simd, &b_values);
    let c = f64x4::from_slice(simd, &c_values);
    assert_eq!(*a.mul_add_precise(b, c), expected);
}

#[simd_test]
fn mul_add_precise_f32x16<S: Simd>(simd: S) {
    let midpoint_a = f32::from_bits(0x3f80_1000);
    let midpoint_b = f32::from_bits(0x3f7f_f800);
    let tiny = f32::from_bits(0x0d80_0000);
    let a_values = [
        1.0 + f32::EPSILON,
        midpoint_a,
        midpoint_a,
        f32::MAX,
        -0.0,
        f32::MIN_POSITIVE,
        3.0,
        -7.0,
        1.0 + f32::EPSILON,
        midpoint_a,
        midpoint_a,
        f32::MAX,
        -0.0,
        f32::MIN_POSITIVE,
        3.0,
        -7.0,
    ];
    let b_values = [
        1.0 - f32::EPSILON,
        midpoint_b,
        midpoint_b,
        2.0,
        2.0,
        0.5,
        4.0,
        5.0,
        1.0 - f32::EPSILON,
        midpoint_b,
        midpoint_b,
        2.0,
        2.0,
        0.5,
        4.0,
        5.0,
    ];
    let c_values = [
        -1.0,
        tiny,
        -tiny,
        -f32::MAX,
        -0.0,
        0.0,
        0.5,
        2.0,
        -1.0,
        tiny,
        -tiny,
        -f32::MAX,
        -0.0,
        0.0,
        0.5,
        2.0,
    ];
    let expected: [f32; 16] =
        core::array::from_fn(|i| a_values[i].mul_add(b_values[i], c_values[i]));

    let a = f32x16::from_slice(simd, &a_values);
    let b = f32x16::from_slice(simd, &b_values);
    let c = f32x16::from_slice(simd, &c_values);
    assert_eq!(*a.mul_add_precise(b, c), expected);
}

#[simd_test]
fn mul_add_precise_f64x8<S: Simd>(simd: S) {
    let midpoint_a = 1.0 + 2.0_f64.powi(-27);
    let midpoint_b = 1.0 - 2.0_f64.powi(-27);
    let tiny = 2.0_f64.powi(-150);
    let a_values = [
        1.0 + f64::EPSILON,
        f64::MAX,
        midpoint_a,
        midpoint_a,
        -0.0,
        f64::MIN_POSITIVE,
        3.0,
        -7.0,
    ];
    let b_values = [
        1.0 - f64::EPSILON,
        2.0,
        midpoint_b,
        midpoint_b,
        2.0,
        0.5,
        4.0,
        5.0,
    ];
    let c_values = [-1.0, -f64::MAX, tiny, -tiny, -0.0, 0.0, 0.5, 2.0];
    let expected = [
        a_values[0].mul_add(b_values[0], c_values[0]),
        a_values[1].mul_add(b_values[1], c_values[1]),
        a_values[2].mul_add(b_values[2], c_values[2]),
        a_values[3].mul_add(b_values[3], c_values[3]),
        a_values[4].mul_add(b_values[4], c_values[4]),
        a_values[5].mul_add(b_values[5], c_values[5]),
        a_values[6].mul_add(b_values[6], c_values[6]),
        a_values[7].mul_add(b_values[7], c_values[7]),
    ];

    let a = f64x8::from_slice(simd, &a_values);
    let b = f64x8::from_slice(simd, &b_values);
    let c = f64x8::from_slice(simd, &c_values);
    assert_eq!(*a.mul_add_precise(b, c), expected);
}

#[inline(always)]
fn reference_fma_f32<S: Simd>(simd: S, x: f32, y: f32, z: f32) -> f32 {
    // Targets with hardware FMA use it; other use our own software emulation
    match simd.level() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(_) | Level::Avx512(_) => x.mul_add(y, z),
        #[cfg(target_arch = "aarch64")]
        Level::Neon(_) => x.mul_add(y, z),
        _ => soft_fma(x, y, z),
    }
}

// Rust std and even musl libc have buggy software FMA:
// https://github.com/rust-lang/compiler-builtins/issues/1262
// We're using a formally verified algorithm in prod,
// but comparing an algorithm against itself is silly,
// so we need something else for verification, and this is the
// "something else", sourced from
// https://github.com/linebender/fearless_simd/pull/323#issuecomment-5234072329
#[inline]
fn soft_fma(x: f32, y: f32, z: f32) -> f32 {
    let xy = f64::from(x) * f64::from(y);
    let z = f64::from(z);
    let result = xy + z;
    let mut u = result.to_bits();
    if u & 0x0fff_ffff != 0 {
        return result as f32;
    }

    if u & 0x1000_0000 == 0 && (u >> 52) & 0x7ff > 1023 - 126 {
        return result as f32;
    }

    if result - xy == z && result - z == xy {
        return result as f32;
    }

    let neg = u >> 63 != 0;
    let err = if neg == (z > xy) {
        xy - result + z
    } else {
        z - result + xy
    };
    if neg == (err < 0.0) {
        u += 1;
    } else {
        u -= 1;
    }
    f64::from_bits(u) as f32
}

#[simd_test]
#[ignore = "exhaustively checks all non-NaN f32 bit patterns"]
fn mul_add_precise_f32x4_exhaustive<S: Simd>(simd: S) {
    // Using each value for all three operands exercises every non-NaN bit pattern in every input
    // position while keeping the exhaustive domain tractable.
    simd.vectorize(
        #[inline(always)]
        || {
            for base in (0..u32::MAX).step_by(4) {
                let values = f32x4::from_fn(simd, |lane| f32::from_bits(base + lane as u32));
                let actual = values.mul_add_precise(values, values);

                for lane in 0..4 {
                    let value = values[lane];
                    if value.is_nan() {
                        continue;
                    }

                    let expected = value.mul_add(value, value);
                    if expected.is_nan() {
                        assert!(
                            actual[lane].is_nan(),
                            "mul_add_precise({value:?}, {value:?}, {value:?}) returned {:?}, expected NaN (input bits: {:#010x})",
                            actual[lane],
                            value.to_bits(),
                        );
                    } else {
                        assert_eq!(
                            actual[lane].to_bits(),
                            expected.to_bits(),
                            "mul_add_precise({value:?}, {value:?}, {value:?}) differs from scalar mul_add (input bits: {:#010x})",
                            value.to_bits(),
                        );
                    }
                }
            }
        },
    );
}

#[simd_test]
#[ignore = "randomly checks independent f32 bit patterns"]
// Run with: cargo test --release mul_add_precise_f32x4_random_independent -- --ignored
fn mul_add_precise_f32x4_random_independent_bit_patterns<S: Simd>(simd: S) {
    let mut rng = fastrand::Rng::with_seed(0xd1b5_4a32_d192_ed03);

    for iteration in 0..1_000_000_000 {
        let a_bits: [u32; 4] = core::array::from_fn(|_| rng.u32(..));
        let b_bits: [u32; 4] = core::array::from_fn(|_| rng.u32(..));
        let c_bits: [u32; 4] = core::array::from_fn(|_| rng.u32(..));
        let a_values = a_bits.map(f32::from_bits);
        let b_values = b_bits.map(f32::from_bits);
        let c_values = c_bits.map(f32::from_bits);
        let actual = f32x4::from_slice(simd, &a_values).mul_add_precise(
            f32x4::from_slice(simd, &b_values),
            f32x4::from_slice(simd, &c_values),
        );

        for lane in 0..4 {
            let expected = reference_fma_f32(simd, a_values[lane], b_values[lane], c_values[lane]);
            if expected.is_nan() {
                assert!(
                    actual[lane].is_nan(),
                    "iteration {iteration}, lane {lane}: mul_add_precise(a={:#010x}, b={:#010x}, c={:#010x}) returned {:#010x}, expected NaN",
                    a_bits[lane],
                    b_bits[lane],
                    c_bits[lane],
                    actual[lane].to_bits(),
                );
            } else {
                assert_eq!(
                    actual[lane].to_bits(),
                    expected.to_bits(),
                    "iteration {iteration}, lane {lane}: mul_add_precise inputs a={:#010x}, b={:#010x}, c={:#010x}",
                    a_bits[lane],
                    b_bits[lane],
                    c_bits[lane],
                );
            }
        }
    }
}

#[simd_test]
fn mul_add_precise_f32x4_subnormal_rounding_regression<S: Simd>(simd: S) {
    // Each product is slightly more than half of the smallest f32 subnormal. The exact sum must
    // therefore round to the subnormal adjacent to c, even when a widened f64 addition rounds to
    // the midpoint between c and that adjacent value.
    // Regression test for https://github.com/linebender/fearless_simd/pull/323#issuecomment-5233682925
    let cases: [(u32, u32, u32, u32); 26] = [
        (0x9700_0800, 0x1cff_f001, 0x0001_0002, 0x0001_0001),
        (0x9700_0800, 0x1cff_f001, 0x0001_0002, 0x0001_0001),
        (0x9700_0800, 0x1cff_f001, 0x0020_0002, 0x0020_0001),
        (0x19ff_e002, 0x1a00_1001, 0x0000_0040, 0x0000_0041),
        (0x99ff_e002, 0x1a00_1001, 0x8000_0040, 0x8000_0041),
        (0x19ff_e002, 0x1a00_1001, 0x0000_0400, 0x0000_0401),
        (0x99ff_e002, 0x1a00_1001, 0x8000_0400, 0x8000_0401),
        (0x19ff_e002, 0x1a00_1001, 0x0001_0000, 0x0001_0001),
        (0x99ff_e002, 0x1a00_1001, 0x8001_0000, 0x8001_0001),
        (0x19ff_e002, 0x1a00_1001, 0x0020_0000, 0x0020_0001),
        (0x99ff_e002, 0x1a00_1001, 0x8020_0000, 0x8020_0001),
        (0x19ff_e002, 0x1a00_1001, 0x0040_0000, 0x0040_0001),
        (0x99ff_e002, 0x1a00_1001, 0x8040_0000, 0x8040_0001),
        (0x99ff_e002, 0x1a00_1001, 0x0000_0042, 0x0000_0041),
        (0x19ff_e002, 0x1a00_1001, 0x8000_0042, 0x8000_0041),
        (0x99ff_e002, 0x1a00_1001, 0x0000_0402, 0x0000_0401),
        (0x19ff_e002, 0x1a00_1001, 0x8000_0402, 0x8000_0401),
        (0x99ff_e002, 0x1a00_1001, 0x0001_0002, 0x0001_0001),
        (0x19ff_e002, 0x1a00_1001, 0x8001_0002, 0x8001_0001),
        (0x99ff_e002, 0x1a00_1001, 0x0020_0002, 0x0020_0001),
        (0x19ff_e002, 0x1a00_1001, 0x8020_0002, 0x8020_0001),
        (0x1700_0800, 0x1cff_f001, 0x0001_0000, 0x0001_0001),
        (0x1700_0800, 0x1cff_f001, 0x0004_0000, 0x0004_0001),
        (0x1700_0800, 0x1cff_f001, 0x0020_0000, 0x0020_0001),
        (0x9700_0800, 0x1cff_f001, 0x0001_0002, 0x0001_0001),
        (0x9700_0800, 0x1cff_f001, 0x0020_0002, 0x0020_0001),
    ];

    for (a_bits, b_bits, c_bits, expected_bits) in cases {
        let a = f32x4::splat(simd, f32::from_bits(a_bits));
        let b = f32x4::splat(simd, f32::from_bits(b_bits));
        let c = f32x4::splat(simd, f32::from_bits(c_bits));
        let actual = a.mul_add_precise(b, c);
        let actual_bits = [
            actual[0].to_bits(),
            actual[1].to_bits(),
            actual[2].to_bits(),
            actual[3].to_bits(),
        ];

        assert_eq!(
            actual_bits, [expected_bits; 4],
            "mul_add_precise inputs: a={a_bits:#010x}, b={b_bits:#010x}, c={c_bits:#010x}",
        );
    }
}

#[simd_test]
#[ignore = "randomly stress-tests subnormal results"]
// Run with: cargo test --release mul_add_precise_f32x4_random -- --ignored
fn mul_add_precise_f32x4_random_reported_product_patterns<S: Simd>(simd: S) {
    let mut rng = fastrand::Rng::with_seed(0x8b7d_3a21_19c4_e605);
    let significand_pairs = [(0x00ff_e002, 0x0080_1001), (0x0080_0800, 0x00ff_f001)];

    for iteration in 0..1_000_000 {
        let mut a_bits = [0_u32; 4];
        let mut b_bits = [0_u32; 4];
        let mut c_bits = [0_u32; 4];

        for lane in 0..4 {
            let mut significands = significand_pairs[rng.usize(..significand_pairs.len())];
            if rng.bool() {
                significands = (significands.1, significands.0);
            }

            // The unbiased exponents sum to -151. Both significand products are just above 2^47,
            // so the resulting product is just above half of the smallest f32 subnormal.
            let a_exponent = rng.i32(-126..-24);
            let b_exponent = -151 - a_exponent;
            let a_negative = rng.bool();
            let product_negative = rng.bool();
            let b_negative = a_negative ^ product_negative;
            a_bits[lane] = ((a_negative as u32) << 31)
                | (((a_exponent + 127) as u32) << 23)
                | (significands.0 & 0x007f_ffff);
            b_bits[lane] = ((b_negative as u32) << 31)
                | (((b_exponent + 127) as u32) << 23)
                | (significands.1 & 0x007f_ffff);

            let magnitude = match lane {
                0 => rng.u32(1..0x0080_0000),
                1 => {
                    let highest_bit = rng.u32(0..23);
                    let lower_bits = if highest_bit == 0 {
                        0
                    } else {
                        rng.u32(0..1 << highest_bit)
                    };
                    (1 << highest_bit) | lower_bits
                }
                2 => 0x007f_ffff - rng.u32(0..1024),
                3 => rng.u32(0..512),
                _ => unreachable!(),
            };
            c_bits[lane] = ((rng.bool() as u32) << 31) | magnitude;
        }

        let a_values = a_bits.map(f32::from_bits);
        let b_values = b_bits.map(f32::from_bits);
        let c_values = c_bits.map(f32::from_bits);
        let expected_bits: [u32; 4] = core::array::from_fn(|lane| {
            reference_fma_f32(simd, a_values[lane], b_values[lane], c_values[lane]).to_bits()
        });
        let actual = f32x4::from_slice(simd, &a_values).mul_add_precise(
            f32x4::from_slice(simd, &b_values),
            f32x4::from_slice(simd, &c_values),
        );
        let actual_bits = [
            actual[0].to_bits(),
            actual[1].to_bits(),
            actual[2].to_bits(),
            actual[3].to_bits(),
        ];

        assert_eq!(
            actual_bits, expected_bits,
            "iteration {iteration}: a={a_bits:x?}, b={b_bits:x?}, c={c_bits:x?}",
        );
    }
}

#[simd_test]
#[ignore = "randomly stress-tests subnormal results"]
fn mul_add_precise_f32x4_random_half_subnormal_products<S: Simd>(simd: S) {
    let mut rng = fastrand::Rng::with_seed(0x31f2_b984_7d06_ca5e);

    for iteration in 0..1_000_000 {
        let mut a_bits = [0_u32; 4];
        let mut b_bits = [0_u32; 4];
        let mut c_bits = [0_u32; 4];

        for lane in 0..4 {
            // Construct random 24-bit significands whose product is the closest representable
            // integer product above or below 2^47. With exponents summing to -151, this places the
            // floating-point product immediately around half of the smallest f32 subnormal.
            let offset = rng.u32(1..0x0080_0000) as u64;
            let low_significand = 1_u64 << 23;
            let high_significand = 1_u64 << 24;
            let b_significand = low_significand + offset;
            let numerator = high_significand * offset;
            let quotient = numerator / b_significand;
            let remainder = numerator % b_significand;
            let distance_from_high = if rng.bool() || remainder == 0 {
                quotient
            } else {
                quotient + 1
            };
            let mut significands = (high_significand - distance_from_high, b_significand);
            if rng.bool() {
                significands = (significands.1, significands.0);
            }

            let a_exponent = rng.i32(-126..-24);
            let b_exponent = -151 - a_exponent;
            let a_negative = rng.bool();
            let product_negative = rng.bool();
            let b_negative = a_negative ^ product_negative;
            a_bits[lane] = ((a_negative as u32) << 31)
                | (((a_exponent + 127) as u32) << 23)
                | (significands.0 as u32 & 0x007f_ffff);
            b_bits[lane] = ((b_negative as u32) << 31)
                | (((b_exponent + 127) as u32) << 23)
                | (significands.1 as u32 & 0x007f_ffff);

            let highest_bit = rng.u32(0..23);
            let lower_bits = if highest_bit == 0 {
                0
            } else {
                rng.u32(0..1 << highest_bit)
            };
            let magnitude = (1 << highest_bit) | lower_bits;
            c_bits[lane] = ((rng.bool() as u32) << 31) | magnitude;
        }

        let a_values = a_bits.map(f32::from_bits);
        let b_values = b_bits.map(f32::from_bits);
        let c_values = c_bits.map(f32::from_bits);
        let expected_bits: [u32; 4] = core::array::from_fn(|lane| {
            reference_fma_f32(simd, a_values[lane], b_values[lane], c_values[lane]).to_bits()
        });
        let actual = f32x4::from_slice(simd, &a_values).mul_add_precise(
            f32x4::from_slice(simd, &b_values),
            f32x4::from_slice(simd, &c_values),
        );
        let actual_bits = [
            actual[0].to_bits(),
            actual[1].to_bits(),
            actual[2].to_bits(),
            actual[3].to_bits(),
        ];

        assert_eq!(
            actual_bits, expected_bits,
            "iteration {iteration}: a={a_bits:x?}, b={b_bits:x?}, c={c_bits:x?}",
        );
    }
}
