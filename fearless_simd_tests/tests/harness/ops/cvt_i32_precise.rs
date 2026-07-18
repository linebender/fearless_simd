// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn cvt_i32_precise_f32x4_sat<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, f32::NAN, 5e9, -5e9]);

    assert_eq!(
        *a.to_int_precise::<i32x4<_>>(),
        [-10, 0, i32::MAX, i32::MIN]
    );
}

#[simd_test]
fn cvt_i32_precise_f32x4_inf<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);

    assert_eq!(
        *a.to_int_precise::<i32x4<_>>(),
        [-10, 0, i32::MAX, i32::MIN]
    );
}

#[simd_test]
fn cvt_i32_precise_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(
        simd,
        &[
            -10.3,
            f32::NAN,
            5e9,
            -5e9,
            f32::INFINITY,
            f32::NEG_INFINITY,
            42.7,
            -0.9,
        ],
    );
    assert_eq!(
        *a.to_int_precise::<i32x8<_>>(),
        [-10, 0, i32::MAX, i32::MIN, i32::MAX, i32::MIN, 42, 0]
    );
}

#[simd_test]
fn cvt_i32_precise_f32x16<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtTruncate;
    // Test precise truncation with special values
    let a = f32x16::from_slice(
        simd,
        &[
            1.7,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1e20,
            1e20,
            0.0,
            -0.0,
            i32::MAX as f32,
            i32::MIN as f32,
            0.5,
            -0.5,
            0.9999,
            -0.9999,
            2.5,
            -2.5,
        ],
    );
    let result = i32x16::truncate_from_precise(a);
    // NaN -> 0, infinity -> saturated
    assert_eq!(result[0], 1);
    assert_eq!(result[1], 0); // NaN
    assert_eq!(result[2], i32::MAX); // +inf saturates
    assert_eq!(result[3], i32::MIN); // -inf saturates
    assert_eq!(result[4], i32::MIN); // -1e20 saturates to MIN
    assert_eq!(result[5], i32::MAX); // 1e20 saturates to MAX
    assert_eq!(result[6], 0);
    assert_eq!(result[7], 0);
}
