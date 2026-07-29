// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn abs_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-1.0, 2.0, -3.0, 4.0]);
    assert_eq!(*a.abs(), [1.0, 2.0, 3.0, 4.0]);
}

#[simd_test]
fn abs_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[-1.5, 2.5]);
    assert_eq!(*a.abs(), [1.5, 2.5]);
}

#[simd_test]
fn abs_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]);
    assert_eq!(*a.abs(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[simd_test]
fn abs_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0,
            -15.0, 16.0,
        ],
    );
    assert_eq!(
        *a.abs(),
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
        ]
    );
}

#[simd_test]
fn abs_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]);
    assert_eq!(*a.abs(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn abs_f64x4<S: Simd>(simd: S) {
    let values: [f64; 4] = core::array::from_fn(|i| i as f64 - 3.5_f64);
    let a = f64x4::from_slice(simd, &values);
    let expected: [f64; 4] = core::array::from_fn(|i| values[i].abs());
    let result = simd.abs_f64x4(a);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn abs_f32x16_special_bit_patterns<S: Simd>(simd: S) {
    let values = [
        -0.0,
        0.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::from_bits(0xffc0_0001),
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x0000_0001),
        -f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
        -f32::MAX,
        f32::MAX,
        -1.0,
        1.0,
        -42.5,
        42.5,
    ];
    let expected = values.map(|value| value.abs().to_bits());
    let result = simd.abs_f32x16(f32x16::from_slice(simd, &values));
    let result_bits = (*result).map(|value| value.to_bits());

    assert_eq!(result_bits, expected);
}

#[simd_test]
fn abs_f64x8_special_bit_patterns<S: Simd>(simd: S) {
    let values = [
        -0.0,
        0.0,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::from_bits(0xfff8_0000_0000_0001),
        f64::from_bits(0x7ff8_0000_0000_1234),
        f64::from_bits(0x8000_0000_0000_0001),
        f64::from_bits(0x0000_0000_0000_0001),
    ];
    let expected = values.map(|value| value.abs().to_bits());
    let result = simd.abs_f64x8(f64x8::from_slice(simd, &values));
    let result_bits = (*result).map(|value| value.to_bits());

    assert_eq!(result_bits, expected);
}
