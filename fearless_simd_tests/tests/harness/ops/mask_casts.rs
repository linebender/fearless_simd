// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

fn test_a_to_b<S: Simd, A, B>(simd: S)
where
    A: SimdMask<S>,
    B: SimdMask<S> + From<A>,
{
    let n = A::LEN;
    let mut rng = fastrand::Rng::with_seed(0xE397_FACE_F768_F459);
    for _ in 0..100 {
        let bits = rng.u64(0..(1 << n));
        let a: A = A::from_bitmask(simd, bits);
        assert_eq!(B::from(a).to_bitmask(), bits);
    }
}

#[simd_test]
fn cast_mask8x16_to_mask16x16<S: Simd>(simd: S) {
    test_a_to_b::<S, mask8x16<S>, mask16x16<S>>(simd);
}

#[simd_test]
fn cast_mask8x16_to_mask32x16<S: Simd>(simd: S) {
    test_a_to_b::<S, mask8x16<S>, mask32x16<S>>(simd);
}

#[simd_test]
fn cast_mask16x8_to_mask32x8<S: Simd>(simd: S) {
    test_a_to_b::<S, mask16x8<S>, mask32x8<S>>(simd);
}

#[simd_test]
fn cast_mask16x8_to_mask64x8<S: Simd>(simd: S) {
    test_a_to_b::<S, mask16x8<S>, mask64x8<S>>(simd);
}

#[simd_test]
fn cast_mask32x4_to_mask64x4<S: Simd>(simd: S) {
    test_a_to_b::<S, mask32x4<S>, mask64x4<S>>(simd);
}

#[simd_test]
fn cast_mask8x32_to_mask16x32<S: Simd>(simd: S) {
    test_a_to_b::<S, mask8x32<S>, mask16x32<S>>(simd);
}

#[simd_test]
fn cast_mask16x16_to_mask8x16<S: Simd>(simd: S) {
    test_a_to_b::<S, mask16x16<S>, mask8x16<S>>(simd);
}

#[simd_test]
fn cast_mask16x16_to_mask32x16<S: Simd>(simd: S) {
    test_a_to_b::<S, mask16x16<S>, mask32x16<S>>(simd);
}

#[simd_test]
fn cast_mask32x8_to_mask16x8<S: Simd>(simd: S) {
    test_a_to_b::<S, mask32x8<S>, mask16x8<S>>(simd);
}

#[simd_test]
fn cast_mask32x8_to_mask64x8<S: Simd>(simd: S) {
    test_a_to_b::<S, mask32x8<S>, mask64x8<S>>(simd);
}

#[simd_test]
fn cast_mask64x4_to_mask32x4<S: Simd>(simd: S) {
    test_a_to_b::<S, mask64x4<S>, mask32x4<S>>(simd);
}

#[simd_test]
fn cast_mask16x32_to_mask8x32<S: Simd>(simd: S) {
    test_a_to_b::<S, mask16x32<S>, mask8x32<S>>(simd);
}

#[simd_test]
fn cast_mask32x16_to_mask8x16<S: Simd>(simd: S) {
    test_a_to_b::<S, mask32x16<S>, mask8x16<S>>(simd);
}

#[simd_test]
fn cast_mask32x16_to_mask16x16<S: Simd>(simd: S) {
    test_a_to_b::<S, mask32x16<S>, mask16x16<S>>(simd);
}

#[simd_test]
fn cast_mask64x8_to_mask16x8<S: Simd>(simd: S) {
    test_a_to_b::<S, mask64x8<S>, mask16x8<S>>(simd);
}

#[simd_test]
fn cast_mask64x8_to_mask32x8<S: Simd>(simd: S) {
    test_a_to_b::<S, mask64x8<S>, mask32x8<S>>(simd);
}
