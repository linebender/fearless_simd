// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
#[ignore = "exhaustively checks all non-NaN f32 bit patterns"]
fn mul_sub_precise_f32x4_exhaustive<S: Simd>(simd: S) {
    // Using each value for all three operands exercises every non-NaN bit pattern in every input
    // position while keeping the exhaustive domain tractable.
    simd.vectorize(
        #[inline(always)]
        || {
            for base in (0..u32::MAX).step_by(4) {
                let values = f32x4::from_fn(simd, |lane| f32::from_bits(base + lane as u32));
                let actual = values.mul_sub_precise(values, values);

                for lane in 0..4 {
                    let value = values[lane];
                    if value.is_nan() {
                        continue;
                    }

                    let expected = value.mul_add(value, -value);
                    if expected.is_nan() {
                        assert!(
                            actual[lane].is_nan(),
                            "mul_sub_precise({value:?}, {value:?}, {value:?}) returned {:?}, expected NaN (input bits: {:#010x})",
                            actual[lane],
                            value.to_bits(),
                        );
                    } else {
                        assert_eq!(
                            actual[lane].to_bits(),
                            expected.to_bits(),
                            "mul_sub_precise({value:?}, {value:?}, {value:?}) differs from scalar mul_add with a negated addend (input bits: {:#010x})",
                            value.to_bits(),
                        );
                    }
                }
            }
        },
    );
}

#[simd_test]
fn mul_sub_precise_f32x4<S: Simd>(simd: S) {
    let midpoint_a = f32::from_bits(0x3f80_1000); // 1 + 2^-11
    let midpoint_b = f32::from_bits(0x3f7f_f800); // 1 - 2^-13
    let tiny = f32::from_bits(0x0d80_0000); // 2^-100
    let a_values = [1.0 + f32::EPSILON, midpoint_a, midpoint_a, f32::MAX];
    let b_values = [1.0 - f32::EPSILON, midpoint_b, midpoint_b, 2.0];
    let c_values = [1.0, -tiny, tiny, f32::MAX];
    let expected = [
        a_values[0].mul_add(b_values[0], -c_values[0]),
        a_values[1].mul_add(b_values[1], -c_values[1]),
        a_values[2].mul_add(b_values[2], -c_values[2]),
        a_values[3].mul_add(b_values[3], -c_values[3]),
    ];

    let a = f32x4::from_slice(simd, &a_values);
    let b = f32x4::from_slice(simd, &b_values);
    let c = f32x4::from_slice(simd, &c_values);
    assert_eq!(*a.mul_sub_precise(b, c), expected);

    // The positive tiny subtrahend moves the exact result below an f32 midpoint, but is lost by a
    // naive f64 subtraction before narrowing.
    assert_ne!(
        expected[2],
        ((midpoint_a as f64) * (midpoint_b as f64) - (tiny as f64)) as f32
    );
}

#[simd_test]
fn mul_sub_precise_f32x4_special_values<S: Simd>(simd: S) {
    let a_values = [-0.0_f32, f32::MIN_POSITIVE, f32::INFINITY, f32::NAN];
    let b_values = [2.0_f32, 0.5, 2.0, 1.0];
    let c_values = [0.0_f32, -0.0, f32::INFINITY, -1.0];
    let a = f32x4::from_slice(simd, &a_values);
    let b = f32x4::from_slice(simd, &b_values);
    let c = f32x4::from_slice(simd, &c_values);
    let result = a.mul_sub_precise(b, c);

    assert_eq!(
        result[0].to_bits(),
        a_values[0].mul_add(b_values[0], -c_values[0]).to_bits()
    );
    assert_eq!(
        result[1].to_bits(),
        a_values[1].mul_add(b_values[1], -c_values[1]).to_bits()
    );
    assert!(result[2].is_nan());
    assert!(result[3].is_nan());
}

#[simd_test]
fn mul_sub_precise_f64x2<S: Simd>(simd: S) {
    let a_values = [1.0 + f64::EPSILON, f64::MAX];
    let b_values = [1.0 - f64::EPSILON, 2.0];
    let c_values = [1.0, f64::MAX];
    let expected = [
        a_values[0].mul_add(b_values[0], -c_values[0]),
        a_values[1].mul_add(b_values[1], -c_values[1]),
    ];

    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let c = f64x2::from_slice(simd, &c_values);
    assert_eq!(*a.mul_sub_precise(b, c), expected);
}

#[simd_test]
fn mul_sub_precise_f64x2_special_values<S: Simd>(simd: S) {
    let a_values = [-0.0_f64, f64::INFINITY];
    let b_values = [2.0_f64, 2.0];
    let c_values = [0.0_f64, f64::INFINITY];
    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let c = f64x2::from_slice(simd, &c_values);
    let result = a.mul_sub_precise(b, c);

    assert_eq!(
        result[0].to_bits(),
        a_values[0].mul_add(b_values[0], -c_values[0]).to_bits()
    );
    assert!(result[1].is_nan());
}

#[simd_test]
fn mul_sub_precise_f32x8<S: Simd>(simd: S) {
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
    let c_values = [1.0, -tiny, tiny, f32::MAX, 0.0, -0.0, -0.5, -2.0];
    let expected: [f32; 8] =
        core::array::from_fn(|i| a_values[i].mul_add(b_values[i], -c_values[i]));

    let a = f32x8::from_slice(simd, &a_values);
    let b = f32x8::from_slice(simd, &b_values);
    let c = f32x8::from_slice(simd, &c_values);
    assert_eq!(*a.mul_sub_precise(b, c), expected);
}

#[simd_test]
fn mul_sub_precise_f64x4<S: Simd>(simd: S) {
    let midpoint_a = 1.0 + 2.0_f64.powi(-27);
    let midpoint_b = 1.0 - 2.0_f64.powi(-27);
    let tiny = 2.0_f64.powi(-150);
    let a_values = [1.0 + f64::EPSILON, f64::MAX, midpoint_a, midpoint_a];
    let b_values = [1.0 - f64::EPSILON, 2.0, midpoint_b, midpoint_b];
    let c_values = [1.0, f64::MAX, -tiny, tiny];
    let expected: [f64; 4] =
        core::array::from_fn(|i| a_values[i].mul_add(b_values[i], -c_values[i]));

    let a = f64x4::from_slice(simd, &a_values);
    let b = f64x4::from_slice(simd, &b_values);
    let c = f64x4::from_slice(simd, &c_values);
    assert_eq!(*a.mul_sub_precise(b, c), expected);
}

#[simd_test]
fn mul_sub_precise_f32x16<S: Simd>(simd: S) {
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
        1.0,
        -tiny,
        tiny,
        f32::MAX,
        0.0,
        -0.0,
        -0.5,
        -2.0,
        1.0,
        -tiny,
        tiny,
        f32::MAX,
        0.0,
        -0.0,
        -0.5,
        -2.0,
    ];
    let expected: [f32; 16] =
        core::array::from_fn(|i| a_values[i].mul_add(b_values[i], -c_values[i]));

    let a = f32x16::from_slice(simd, &a_values);
    let b = f32x16::from_slice(simd, &b_values);
    let c = f32x16::from_slice(simd, &c_values);
    assert_eq!(*a.mul_sub_precise(b, c), expected);
}

#[simd_test]
fn mul_sub_precise_f64x8<S: Simd>(simd: S) {
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
    let c_values = [1.0, f64::MAX, -tiny, tiny, 0.0, -0.0, -0.5, -2.0];
    let expected: [f64; 8] =
        core::array::from_fn(|i| a_values[i].mul_add(b_values[i], -c_values[i]));

    let a = f64x8::from_slice(simd, &a_values);
    let b = f64x8::from_slice(simd, &b_values);
    let c = f64x8::from_slice(simd, &c_values);
    assert_eq!(*a.mul_sub_precise(b, c), expected);
}
