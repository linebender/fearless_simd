// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn approximate_recip_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, -2.0, 23.0, 9.0]);
    let result = a.approximate_recip();
    let expected = [1.0, -0.5, 1. / 23., 1. / 9.];
    for i in 0..4 {
        let rel_error = ((result[i] - expected[i]) / expected[i]).abs();
        assert!(
            rel_error < 0.005,
            "approximate_recip({}) rel_error = {rel_error}",
            a[i]
        );
    }
}

#[simd_test]
fn approximate_recip_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.0, -2.0]);
    let result = a.approximate_recip();
    let expected = [1.0, -0.5];
    for i in 0..2 {
        let rel_error = ((result[i] - expected[i]) / expected[i]).abs();
        assert!(
            rel_error < 0.005,
            "approximate_recip({}) rel_error = {rel_error}",
            a[i]
        );
    }
}

#[simd_test]
fn approximate_recip_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, -2.0, 23.0, 9.0]);
    let result = a.approximate_recip();
    let expected = [1.0, -0.5, 1. / 23., 1. / 9.];
    for i in 0..4 {
        let rel_error = ((result[i] - expected[i]) / expected[i]).abs();
        assert!(
            rel_error < 0.005,
            "approximate_recip({}) rel_error = {rel_error}",
            a[i]
        );
    }
}

#[simd_test]
fn approximate_recip_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, -2.0, 23.0, 9.0, 3.5, -7.25, 13.0, 0.25]);
    let result = a.approximate_recip();
    let expected = [
        1.0,
        -0.5,
        1. / 23.,
        1. / 9.,
        1. / 3.5,
        1. / -7.25,
        1. / 13.,
        4.0,
    ];
    for i in 0..8 {
        let rel_error = ((result[i] - expected[i]) / expected[i]).abs();
        assert!(
            rel_error < 0.005,
            "approximate_recip({}) rel_error = {rel_error}",
            a[i]
        );
    }
}

#[simd_test]
fn approximate_recip_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, -2.0, 23.0, 9.0, 0.5, -0.25, 128.0, -1024.0, 3.0, -7.0, 11.0, -13.0, 19.0, -29.0,
            37.0, -41.0,
        ],
    );
    let result = a.approximate_recip();
    for i in 0..16 {
        let expected = 1.0 / a[i];
        let rel_error = ((result[i] - expected) / expected).abs();
        assert!(
            rel_error < 0.005,
            "approximate_recip({}) rel_error = {rel_error}",
            a[i]
        );
    }
}

#[simd_test]
fn approximate_recip_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, -2.0, 23.0, 9.0, 0.5, -0.25, 128.0, -1024.0]);
    let result = a.approximate_recip();
    for i in 0..8 {
        let expected = 1.0 / a[i];
        let rel_error = ((result[i] - expected) / expected).abs();
        assert!(
            rel_error < 0.005,
            "approximate_recip({}) rel_error = {rel_error}",
            a[i]
        );
    }
}
