// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn round_ties_even_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.3, -3.2, 2.7, -3.6]);
    assert_eq!(*a.round_ties_even(), [2.0, -3.0, 3.0, -4.0]);
    let b = f32x4::from_slice(simd, &[-3.5, -2.5, 1.5, 0.5]);
    assert_eq!(*b.round_ties_even(), [-4.0, -2.0, 2.0, 0.0]);
}

#[simd_test]
fn round_ties_even_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[2.3, -3.2]);
    assert_eq!(*a.round_ties_even(), [2.0, -3.0]);
    let b = f64x2::from_slice(simd, &[2.7, -3.6]);
    assert_eq!(*b.round_ties_even(), [3.0, -4.0]);
    let c = f64x2::from_slice(simd, &[-3.5, -2.5]);
    assert_eq!(*c.round_ties_even(), [-4.0, -2.0]);
    let d = f64x2::from_slice(simd, &[1.5, 0.5]);
    assert_eq!(*d.round_ties_even(), [2.0, 0.0]);
}

#[simd_test]
fn round_ties_even_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[2.3, -3.2, 2.7, -3.6, -3.5, -2.5, 1.5, 0.5]);
    assert_eq!(
        *a.round_ties_even(),
        [2.0, -3.0, 3.0, -4.0, -4.0, -2.0, 2.0, 0.0]
    );
}

#[simd_test]
fn round_ties_even_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            2.3, -3.2, 2.7, -3.6, -3.5, -2.5, 1.5, 0.5, 2.3, -3.2, 2.7, -3.6, -3.5, -2.5, 1.5, 0.5,
        ],
    );
    assert_eq!(
        *a.round_ties_even(),
        [
            2.0, -3.0, 3.0, -4.0, -4.0, -2.0, 2.0, 0.0, 2.0, -3.0, 3.0, -4.0, -4.0, -2.0, 2.0, 0.0
        ]
    );
}

#[simd_test]
fn round_ties_even_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[2.3, -3.2, 2.7, -3.6, -3.5, -2.5, 1.5, 0.5]);
    assert_eq!(
        *a.round_ties_even(),
        [2.0, -3.0, 3.0, -4.0, -4.0, -2.0, 2.0, 0.0]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn round_ties_even_f64x4<S: Simd>(simd: S) {
    let values: [f64; 4] = core::array::from_fn(|i| i as f64 - 3.5_f64);
    let a = f64x4::from_slice(simd, &values);
    let expected: [f64; 4] = core::array::from_fn(|i| values[i].round_ties_even());
    let result = simd.round_ties_even_f64x4(a);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn round_ties_even_f32x16_special_bit_patterns<S: Simd>(simd: S) {
    let values = [
        -0.0,
        0.0,
        -0.5,
        0.5,
        -1.5,
        1.5,
        -2.5,
        2.5,
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x0000_0001),
        -((1_u32 << 23) as f32),
        (1_u32 << 23) as f32,
        f32::NEG_INFINITY,
        f32::INFINITY,
        -42.25,
        42.75,
    ];
    let expected = values.map(|value| value.round_ties_even().to_bits());
    let result = simd.round_ties_even_f32x16(f32x16::from_slice(simd, &values));

    assert_eq!((*result).map(f32::to_bits), expected);
}

#[simd_test]
fn round_ties_even_f64x8_special_bit_patterns<S: Simd>(simd: S) {
    let values = [
        -0.0,
        0.0,
        -0.5,
        0.5,
        -1.5,
        1.5,
        f64::NEG_INFINITY,
        f64::INFINITY,
    ];
    let expected = values.map(|value| value.round_ties_even().to_bits());
    let result = simd.round_ties_even_f64x8(f64x8::from_slice(simd, &values));

    assert_eq!((*result).map(f64::to_bits), expected);
}

#[simd_test]
#[ignore = "exhaustive testing to verify bit-exactness of f32xN::round_ties_even"]
fn bit_exact_f32_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=u32::MAX {
        let input = f32::from_bits(bits);
        let expected = input.round_ties_even();
        let actual = f32x4::splat(simd, input).round_ties_even()[0];

        // NaN bit-patterns aren't preserved
        if input.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
        }
    }
}

#[simd_test]
fn round_ties_even_f32x16_range_boundaries<S: Simd>(simd: S) {
    let values = [
        8388607.5_f32,
        -8388609.0,
        -8388607.5,
        8388609.0,
        8388608.0,
        -8388608.0,
        0.5,
        -0.5,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_0001),
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffc0_0001),
    ];
    let result = f32x16::from_slice(simd, &values).round_ties_even();
    for i in 0..16 {
        if values[i].is_nan() {
            assert!(result[i].is_nan());
        } else {
            assert_eq!(result[i].to_bits(), values[i].round_ties_even().to_bits());
        }
    }
}

#[simd_test]
fn round_ties_even_f64x8_range_boundaries<S: Simd>(simd: S) {
    let values = [
        4503599627370495.5_f64,
        -4503599627370497.0,
        -4503599627370495.5,
        4503599627370497.0,
        4503599627370496.0,
        -4503599627370496.0,
        f64::MAX,
        -f64::MAX,
    ];
    let expected = values.map(|value| value.round_ties_even().to_bits());
    let result = f64x8::from_slice(simd, &values).round_ties_even();

    assert_eq!((*result).map(f64::to_bits), expected);
}

#[simd_test]
fn round_ties_even_f64x8_mixed_nan_lanes<S: Simd>(simd: S) {
    let values = [
        f64::from_bits(0x7ff0_0000_0000_0001),
        -0.5,
        1.5,
        f64::from_bits(0xfff0_0000_0000_0001),
        f64::from_bits(0x7ff8_0000_0000_0001),
        -2.5,
        3.5,
        f64::from_bits(0xfff8_0000_0000_0001),
    ];
    let result = f64x8::from_slice(simd, &values).round_ties_even();
    for i in 0..8 {
        if values[i].is_nan() {
            assert!(result[i].is_nan());
        } else {
            assert_eq!(result[i].to_bits(), values[i].round_ties_even().to_bits());
        }
    }
}

#[simd_test]
fn bit_exact_f64_randomized<S: Simd>(simd: S) {
    let mut rng = fastrand::Rng::with_seed(0xa409_3822_299f_31d0);
    // exhaustively check the top K bits and randomize the rest
    let exhaustive_bits = 16;
    for bits in 0..(1_u64 << exhaustive_bits) {
        let bits = (bits << (64 - exhaustive_bits)) | rng.u64(..) >> exhaustive_bits;
        let input = f64::from_bits(bits);
        let expected = input.round_ties_even();
        let actual = f64x2::splat(simd, input).round_ties_even()[0];

        // NaN bit-patterns aren't preserved
        if input.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
        }
    }
}
