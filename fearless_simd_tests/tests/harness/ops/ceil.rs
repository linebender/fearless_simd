// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn ceil_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.1, -3.2, -2.7, 0.5]);
    assert_eq!(*a.ceil(), [3.0, -3.0, -2.0, 1.0]);
}

#[simd_test]
fn ceil_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[2.1, -3.2]);
    assert_eq!(*a.ceil(), [3.0, -3.0]);
    let b = f64x2::from_slice(simd, &[-2.7, 0.5]);
    assert_eq!(*b.ceil(), [-2.0, 1.0]);
}

#[simd_test]
fn ceil_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[2.1, -3.2, -2.7, 0.5, 1.1, -2.9, 3.0, -4.1]);
    assert_eq!(*a.ceil(), [3.0, -3.0, -2.0, 1.0, 2.0, -2.0, 3.0, -4.0]);
}

#[simd_test]
fn ceil_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            2.1, -3.2, -2.7, 0.5, 1.1, -2.9, 3.0, -4.1, 2.1, -3.2, -2.7, 0.5, 1.1, -2.9, 3.0, -4.1,
        ],
    );
    assert_eq!(
        *a.ceil(),
        [
            3.0, -3.0, -2.0, 1.0, 2.0, -2.0, 3.0, -4.0, 3.0, -3.0, -2.0, 1.0, 2.0, -2.0, 3.0, -4.0
        ]
    );
}

#[simd_test]
fn ceil_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[2.1, -3.2, -2.7, 0.5, 1.1, -2.9, 3.0, -4.1]);
    assert_eq!(*a.ceil(), [3.0, -3.0, -2.0, 1.0, 2.0, -2.0, 3.0, -4.0]);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn ceil_f64x4<S: Simd>(simd: S) {
    let values: [f64; 4] = core::array::from_fn(|i| i as f64 - 3.5_f64);
    let a = f64x4::from_slice(simd, &values);
    let expected: [f64; 4] = core::array::from_fn(|i| values[i].ceil());
    let result = simd.ceil_f64x4(a);
    assert_eq!(result.as_slice(), expected.as_slice());
}

// zeros should be bit-exact
#[simd_test]
fn ceil_neg_zero<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-0.2, -0.0, 0.0, 0.2]);
    assert_eq!(
        *a.ceil().to_bits(),
        [-0.0, -0.0, 0.0, 1.0].map(f32::to_bits)
    );

    let a = f64x4::from_slice(simd, &[-0.2, -0.0, 0.0, 0.2]);
    assert_eq!(
        *a.ceil().to_bits(),
        [-0.0, -0.0, 0.0, 1.0].map(f64::to_bits)
    );
}

#[simd_test]
#[ignore = "exhaustive testing to verify bit-exactness of f32xN::ceil"]
fn bit_exact_f32_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=u32::MAX {
        let input = f32::from_bits(bits);
        let expected = input.ceil();
        let actual = f32x4::splat(simd, input).ceil()[0];

        // NaN bit-patterns aren't preserved
        if input.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
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
        let expected = input.ceil();
        let actual = f64x2::splat(simd, input).ceil()[0];

        // NaN bit-patterns aren't preserved
        if input.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(expected.to_bits(), actual.to_bits(), "{input} ({bits:x})");
        }
    }
}
