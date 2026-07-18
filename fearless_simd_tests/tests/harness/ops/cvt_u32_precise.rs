// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn cvt_u32_precise_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-1.0, 42.7, 5e9, f32::NAN]);
    assert_eq!(*a.to_int_precise::<u32x4<_>>(), [0, 42, u32::MAX, 0]);
}

#[simd_test]
fn cvt_u32_precise_f32x4_sat<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, 3000000000.0, 5e9, -5e9]);
    assert_eq!(
        *a.to_int_precise::<u32x4<_>>(),
        [0, 3000000000, u32::MAX, 0]
    );
}

#[simd_test]
fn cvt_u32_precise_f32x4_inf<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);

    assert_eq!(*a.to_int_precise::<u32x4<_>>(), [0, 0, u32::MAX, u32::MIN]);
}

#[simd_test]
fn cvt_u32_precise_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(
        simd,
        &[-1.0, 42.7, 5e9, f32::NAN, 0.0, 1.9, 3000000000.0, -5e9],
    );
    assert_eq!(
        *a.to_int_precise::<u32x8<_>>(),
        [0, 42, u32::MAX, 0, 0, 1, 3000000000, 0]
    );
}

#[simd_test]
fn cvt_u32_precise_f32x8_inf<S: Simd>(simd: S) {
    let a = f32x8::from_slice(
        simd,
        &[
            -10.3,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            u32::MAX as f32,
            4294967040.0,
            4294967296.0,
            -0.5,
        ],
    );

    assert_eq!(
        *a.to_int_precise::<u32x8<_>>(),
        [0, 0, u32::MAX, u32::MIN, u32::MAX, 4294967040, u32::MAX, 0]
    );
}

#[simd_test]
fn cvt_u32_precise_f32x16<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtTruncate;
    // Test precise truncation with special values
    let a = f32x16::from_slice(
        simd,
        &[
            1.7,
            f32::NAN,
            f32::INFINITY,
            0.0,
            1e20,
            0.5,
            -1.0,
            u32::MAX as f32,
            2.5,
            3.9,
            100.1,
            200.9,
            0.001,
            999.999,
            1.0,
            2.0,
        ],
    );
    let result = u32x16::truncate_from_precise(a);
    // NaN -> 0, infinity -> saturated, negative -> 0
    assert_eq!(result[0], 1);
    assert_eq!(result[1], 0); // NaN
    assert_eq!(result[2], u32::MAX); // +inf saturates
    assert_eq!(result[3], 0);
    assert_eq!(result[4], u32::MAX); // 1e20 saturates to MAX
    assert_eq!(result[5], 0); // 0.5 truncates to 0
    assert_eq!(result[6], 0); // -1.0 clamps to 0
}
