// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn copysign_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, -2.0, -3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[-1.0, 1.0, -1.0, 1.0]);
    assert_eq!(*a.copysign(b), [-1.0, 2.0, -3.0, 4.0]);
}

#[simd_test]
fn copysign_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.5, -2.5]);
    let b = f64x2::from_slice(simd, &[-1.0, 1.0]);
    assert_eq!(*a.copysign(b), [-1.5, 2.5]);
}

#[simd_test]
fn copysign_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, -2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0]);
    let b = f32x8::from_slice(simd, &[-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]);
    assert_eq!(*a.copysign(b), [-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, -7.0, 8.0]);
}

#[simd_test]
fn copysign_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, -2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0, 9.0, -10.0, -11.0, 12.0, -13.0, 14.0, 15.0,
            -16.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
        ],
    );
    assert_eq!(
        *a.copysign(b),
        [
            -1.0, 2.0, -3.0, 4.0, 5.0, -6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, 13.0, -14.0,
            -15.0, 16.0
        ]
    );
}

#[simd_test]
fn copysign_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, -2.0, -3.0, 4.0, -5.0, 6.0, 7.0, -8.0]);
    let b = f64x8::from_slice(simd, &[-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]);
    assert_eq!(*a.copysign(b), [-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, -7.0, 8.0]);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn copysign_f64x4<S: Simd>(simd: S) {
    let a_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.0_f64);
    let sign_values: [f64; 4] =
        core::array::from_fn(|i| if i % 2 == 0 { -1.0_f64 } else { 1.0_f64 });
    let a = f64x4::from_slice(simd, &a_values);
    let signs = f64x4::from_slice(simd, &sign_values);
    let expected: [f64; 4] = core::array::from_fn(|i| a_values[i].copysign(sign_values[i]));
    let result = simd.copysign_f64x4(a, signs);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn copysign_f32x16_preserves_magnitude_bits<S: Simd>(simd: S) {
    let magnitudes = [
        f32::from_bits(0xffc0_0001),
        f32::from_bits(0x7fc0_1234),
        -0.0,
        0.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
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
    let signs: [f32; 16] = core::array::from_fn(|i| if i % 2 == 0 { 0.0 } else { -0.0 });
    let expected: [u32; 16] = core::array::from_fn(|i| magnitudes[i].copysign(signs[i]).to_bits());
    let result = simd.copysign_f32x16(
        f32x16::from_slice(simd, &magnitudes),
        f32x16::from_slice(simd, &signs),
    );
    let result_bits = (*result).map(|value| value.to_bits());

    assert_eq!(result_bits, expected);
}

#[simd_test]
fn copysign_f64x8_preserves_magnitude_bits<S: Simd>(simd: S) {
    let magnitudes = [
        f64::from_bits(0xfff8_0000_0000_0001),
        f64::from_bits(0x7ff8_0000_0000_1234),
        -0.0,
        0.0,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::from_bits(0x8000_0000_0000_0001),
        f64::from_bits(0x0000_0000_0000_0001),
    ];
    let signs: [f64; 8] = core::array::from_fn(|i| if i % 2 == 0 { 0.0 } else { -0.0 });
    let expected: [u64; 8] = core::array::from_fn(|i| magnitudes[i].copysign(signs[i]).to_bits());
    let result = simd.copysign_f64x8(
        f64x8::from_slice(simd, &magnitudes),
        f64x8::from_slice(simd, &signs),
    );
    let result_bits = (*result).map(|value| value.to_bits());

    assert_eq!(result_bits, expected);
}
