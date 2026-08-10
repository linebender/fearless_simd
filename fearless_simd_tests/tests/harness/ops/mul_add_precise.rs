// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::any::TypeId;
use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// This file has a lot of coverage for 128-bit vectors specifically.
// That's because we have complex, error-prone emulation of precise FMA
// on SSE4.2 and need to test it in great depth.

#[inline(always)]
fn reference_fma_f32<S: Simd>(_: S, x: f32, y: f32, z: f32) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if TypeId::of::<S>() == TypeId::of::<Avx2>() || TypeId::of::<S>() == TypeId::of::<Avx512>() {
        return x.mul_add(y, z);
    }

    #[cfg(target_arch = "aarch64")]
    if TypeId::of::<S>() == TypeId::of::<Neon>() {
        return x.mul_add(y, z);
    }

    soft_fma(x, y, z)
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
