// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// Integer support is trivial, because they all just use min/max.
// So we just test one for completeness.
#[simd_test]
fn clamp_int<S: Simd>(simd: S) {
    let i = i32x8::from_slice(simd, &[-2, -1, 0, 1, 2, 3, 4, 5]);
    assert_eq!(*i.clamp(-1, 3), [-1, -1, 0, 1, 2, 3, 3, 3]);
}

// One concrete test row per supported vector type.

#[simd_test]
fn clamp_f32<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[-1.0, -0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f32x8::from_slice(simd, &[0.0, -1.0, -1.0, 0.0, 30.0, 1.0, 1.0, 8.0]);
    let c = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 1.0, 50.0, 6.0, 5.0, 8.0]);
    assert_eq!(*a.clamp(b, c), [0.0, -0.0, 0.0, 1.0, 30.0, 4.0, 5.0, 8.0]);

    // split to test f32x4 with the same numbers
    let (a0, a1) = a.split();
    let (b0, b1) = b.split();
    let (c0, c1) = c.split();
    let (clamp0, clamp1) = (a0.clamp(b0, c0), a1.clamp(b1, c1));
    let clamp = clamp0.combine(clamp1);
    assert_eq!(*a.clamp(b, c), *clamp);
}

#[simd_test]
fn clamp_f64<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[-1.0, -0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64x8::from_slice(simd, &[0.0, -1.0, -1.0, 0.0, 30.0, 1.0, 1.0, 8.0]);
    let c = f64x8::from_slice(simd, &[1.0, 2.0, 3.0, 1.0, 50.0, 6.0, 5.0, 8.0]);
    assert_eq!(*a.clamp(b, c), [0.0, -0.0, 0.0, 1.0, 30.0, 4.0, 5.0, 8.0]);

    // split to test f64x4 with the same numbers
    let (a0, a1) = a.split();
    let (b0, b1) = b.split();
    let (c0, c1) = c.split();
    let (clamp0, clamp1) = (a0.clamp(b0, c0), a1.clamp(b1, c1));
    let clamp = clamp0.combine(clamp1);
    assert_eq!(*a.clamp(b, c), *clamp);
}

#[simd_test]
fn clamp_f32_nan<S: Simd>(simd: S) {
    let a = f32x8::splat(simd, f32::NAN);
    let b = f32x8::from_slice(
        simd,
        &[
            0.0,
            -0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
    );
    let c = f32x8::from_slice(
        simd,
        &[
            1.0,
            1.0,
            3.0,
            0.0,
            f32::INFINITY,
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
    );
    let res = a.clamp(b, c);
    assert!(res.is_nan().all_true(), "Failed: {:?}", *res);
}

#[simd_test]
fn clamp_f64_nan<S: Simd>(simd: S) {
    let a = f64x8::splat(simd, f64::NAN);
    let b = f64x8::from_slice(
        simd,
        &[
            0.0,
            -0.0,
            0.0,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ],
    );
    let c = f64x8::from_slice(
        simd,
        &[
            1.0,
            1.0,
            3.0,
            0.0,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ],
    );
    let res = a.clamp(b, c);
    assert!(res.is_nan().all_true(), "Failed: {:?}", *res);
}
