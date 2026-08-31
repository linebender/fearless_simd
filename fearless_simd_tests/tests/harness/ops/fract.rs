// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn fract_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.7, -2.3, 3.9, -4.1]);
    assert_eq!(
        *simd.fract_f32x4(a),
        [0.70000005, -0.29999995, 0.9000001, -0.099999905]
    );
}

#[simd_test]
fn fract_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.7, -2.3]);
    assert_eq!(*a.fract(), [0.7, -0.2999999999999998]);
}

#[simd_test]
fn fract_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.7, -2.3, 3.9, -4.1, 5.5, -6.6, 7.2, -8.8]);
    let result = simd.fract_f32x8(a);
    assert_eq!(
        *result,
        [
            0.70000005,
            -0.29999995,
            0.9000001,
            -0.099999905,
            0.5,
            -0.5999999,
            0.19999981, // 7.2 - 7.0 has precision differences
            -0.8000002
        ]
    );
}

#[simd_test]
fn fract_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.7, -2.3, 3.9, -4.1, 5.5, -6.6, 7.2, -8.8, 1.25, -2.75, 0.0, -0.5, 10.125, -10.875,
            100.0, -100.0,
        ],
    );
    let result = simd.fract_f32x16(a);
    assert_eq!(
        *result,
        [
            0.70000005,
            -0.29999995,
            0.9000001,
            -0.099999905,
            0.5,
            -0.5999999,
            0.19999981,
            -0.8000002,
            0.25,
            -0.75,
            0.0,
            -0.5,
            0.125,
            -0.875,
            0.0,
            0.0
        ]
    );
}

#[simd_test]
fn fract_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.7, -2.3, 3.9, -4.1, 5.5, -6.6, 7.2, -8.8]);
    let result = simd.fract_f64x8(a);
    assert_eq!(
        *result,
        [
            0.7,
            -0.2999999999999998,
            0.8999999999999999,
            -0.09999999999999964,
            0.5,
            -0.5999999999999996,
            0.20000000000000018,
            -0.8000000000000007
        ]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn fract_f64x4<S: Simd>(simd: S) {
    let values: [f64; 4] = core::array::from_fn(|i| i as f64 - 3.5_f64);
    let a = f64x4::from_slice(simd, &values);
    let expected: [f64; 4] = core::array::from_fn(|i| values[i].fract());
    let result = simd.fract_f64x4(a);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn fract_f32x4_edge_cases<S: Simd>(simd: S) {
    let values = [
        -0.0,
        2_147_483_904.0_f32, // 2^31 + one f32 ULP
        f32::MAX,
        f32::INFINITY,
    ];
    let a = f32x4::from_slice(simd, &values);
    let result = simd.fract_f32x4(a);

    assert_eq!(result[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(result[1].to_bits(), 0.0_f32.to_bits());
    assert_eq!(result[2].to_bits(), 0.0_f32.to_bits());
    assert!(result[3].is_nan());

    let values = [
        -2_147_483_904.0_f32, // -2^31 - one f32 ULP
        f32::MIN,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let a = f32x4::from_slice(simd, &values);
    let result = simd.fract_f32x4(a);

    assert_eq!(result[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(result[1].to_bits(), 0.0_f32.to_bits());
    assert!(result[2].is_nan());
    assert!(result[3].is_nan());
}

#[simd_test]
fn fract_f64x2_edge_cases<S: Simd>(simd: S) {
    let values = [
        -0.0,
        9_223_372_036_854_777_856.0_f64, // 2^63 + one f64 ULP
    ];
    let a = f64x2::from_slice(simd, &values);
    let result = simd.fract_f64x2(a);

    assert_eq!(result[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(result[1].to_bits(), 0.0_f64.to_bits());

    let values = [f64::MAX, f64::INFINITY];
    let a = f64x2::from_slice(simd, &values);
    let result = simd.fract_f64x2(a);

    assert_eq!(result[0].to_bits(), 0.0_f64.to_bits());
    assert!(result[1].is_nan());

    let values = [
        -9_223_372_036_854_777_856.0_f64, // -2^63 - one f64 ULP
        f64::MIN,
    ];
    let a = f64x2::from_slice(simd, &values);
    let result = simd.fract_f64x2(a);

    assert_eq!(result[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(result[1].to_bits(), 0.0_f64.to_bits());

    let values = [f64::NEG_INFINITY, f64::NAN];
    let a = f64x2::from_slice(simd, &values);
    let result = simd.fract_f64x2(a);

    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
}
