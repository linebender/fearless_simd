// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

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
