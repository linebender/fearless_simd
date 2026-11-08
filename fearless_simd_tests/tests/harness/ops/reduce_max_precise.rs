// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

const Q_NAN_F32: f32 = f32::NAN;
const Q_NAN_F64: f64 = f64::NAN;
const S_NAN_F32: f32 = f32::from_bits(0x7FFF_FFFF);
const S_NAN_F64: f64 = f64::from_bits(0x7FFF_FFFF_FFFF_FFFF);

#[simd_test]
fn reduce_max_precise_f32x4<S: Simd>(simd: S) {
    assert_eq!(
        f32x4::from_slice(simd, &[Q_NAN_F32, 4.0, 12.0, 7.0]).reduce_max_precise(),
        12.0
    );
    assert_eq!(
        f32x4::from_slice(simd, &[4.0, 12.0, 7.0, S_NAN_F32]).reduce_max_precise(),
        12.0
    );
    assert_eq!(
        f32x4::from_slice(simd, &[Q_NAN_F32, 4.0, S_NAN_F32, 12.0]).reduce_max_precise(),
        12.0
    );
    assert!(
        f32x4::from_slice(simd, &[S_NAN_F32; 4])
            .reduce_max_precise()
            .is_nan()
    );
    assert_eq!(
        f32x4::from_slice(simd, &[f32::NEG_INFINITY, S_NAN_F32, 3.0, f32::INFINITY])
            .reduce_max_precise(),
        f32::INFINITY
    );
    assert_eq!(
        f32x4::from_slice(simd, &[0.0, -0.0, 0.0, -0.0]).reduce_max_precise(),
        0.0
    );
}

#[simd_test]
fn reduce_max_precise_f64x2<S: Simd>(simd: S) {
    assert_eq!(
        f64x2::from_slice(simd, &[Q_NAN_F64, 12.0]).reduce_max_precise(),
        12.0
    );
    assert_eq!(
        f64x2::from_slice(simd, &[12.0, S_NAN_F64]).reduce_max_precise(),
        12.0
    );
    assert!(
        f64x2::from_slice(simd, &[S_NAN_F64; 2])
            .reduce_max_precise()
            .is_nan()
    );
    assert_eq!(
        f64x2::from_slice(simd, &[f64::NEG_INFINITY, f64::INFINITY]).reduce_max_precise(),
        f64::INFINITY
    );
    assert_eq!(
        f64x2::from_slice(simd, &[0.0, -0.0]).reduce_max_precise(),
        0.0
    );
}

#[simd_test]
fn reduce_max_precise_f32x8<S: Simd>(simd: S) {
    assert_eq!(
        f32x8::from_slice(simd, &[Q_NAN_F32, 4.0, 12.0, 7.0, 9.0, 3.0, 6.0, 8.0])
            .reduce_max_precise(),
        12.0
    );
    assert_eq!(
        f32x8::from_slice(simd, &[4.0, 12.0, 7.0, 9.0, 3.0, 6.0, 8.0, S_NAN_F32])
            .reduce_max_precise(),
        12.0
    );
    assert_eq!(
        f32x8::from_slice(
            simd,
            &[Q_NAN_F32, 4.0, S_NAN_F32, 7.0, 12.0, S_NAN_F32, 6.0, 8.0]
        )
        .reduce_max_precise(),
        12.0
    );
    assert!(
        f32x8::from_slice(simd, &[S_NAN_F32; 8])
            .reduce_max_precise()
            .is_nan()
    );
    assert_eq!(
        f32x8::from_slice(
            simd,
            &[
                f32::NEG_INFINITY,
                S_NAN_F32,
                3.0,
                8.0,
                2.0,
                f32::INFINITY,
                5.0,
                7.0
            ],
        )
        .reduce_max_precise(),
        f32::INFINITY
    );
    let zeros = f32x8::from_fn(simd, |i| if i % 2 == 0 { 0.0 } else { -0.0 });
    assert_eq!(zeros.reduce_max_precise(), 0.0);
}

#[simd_test]
fn reduce_max_precise_f64x4<S: Simd>(simd: S) {
    assert_eq!(
        f64x4::from_slice(simd, &[Q_NAN_F64, 4.0, 12.0, 7.0]).reduce_max_precise(),
        12.0
    );
    assert_eq!(
        f64x4::from_slice(simd, &[4.0, 12.0, 7.0, S_NAN_F64]).reduce_max_precise(),
        12.0
    );
    assert_eq!(
        f64x4::from_slice(simd, &[Q_NAN_F64, 4.0, S_NAN_F64, 12.0]).reduce_max_precise(),
        12.0
    );
    assert!(
        f64x4::from_slice(simd, &[Q_NAN_F64; 4])
            .reduce_max_precise()
            .is_nan()
    );
    assert_eq!(
        f64x4::from_slice(simd, &[f64::NEG_INFINITY, S_NAN_F64, 3.0, f64::INFINITY])
            .reduce_max_precise(),
        f64::INFINITY
    );
    assert_eq!(
        f64x4::from_slice(simd, &[0.0, -0.0, 0.0, -0.0]).reduce_max_precise(),
        0.0
    );
}

#[simd_test]
fn reduce_max_precise_f32x16<S: Simd>(simd: S) {
    let mut first = [4.0; 16];
    first[0] = Q_NAN_F32;
    first[13] = 12.0;
    assert_eq!(f32x16::from_slice(simd, &first).reduce_max_precise(), 12.0);

    let mut last = [4.0; 16];
    last[15] = S_NAN_F32;
    last[2] = 12.0;
    assert_eq!(f32x16::from_slice(simd, &last).reduce_max_precise(), 12.0);

    let multiple = f32x16::from_fn(simd, |i| if i % 3 == 0 { S_NAN_F32 } else { i as f32 });
    assert_eq!(multiple.reduce_max_precise(), 14.0);
    assert!(
        f32x16::from_slice(simd, &[S_NAN_F32; 16])
            .reduce_max_precise()
            .is_nan()
    );

    let infinity = f32x16::from_fn(simd, |i| {
        if i == 11 {
            f32::INFINITY
        } else if i == 3 {
            f32::NEG_INFINITY
        } else {
            i as f32
        }
    });
    assert_eq!(infinity.reduce_max_precise(), f32::INFINITY);
    let zeros = f32x16::from_fn(simd, |i| if i % 2 == 0 { 0.0 } else { -0.0 });
    assert_eq!(zeros.reduce_max_precise(), 0.0);
}

#[simd_test]
fn reduce_max_precise_f64x8<S: Simd>(simd: S) {
    assert_eq!(
        f64x8::from_slice(simd, &[Q_NAN_F64, 4.0, 12.0, 7.0, 9.0, 3.0, 6.0, 8.0])
            .reduce_max_precise(),
        12.0
    );
    assert_eq!(
        f64x8::from_slice(simd, &[4.0, 12.0, 7.0, 9.0, 3.0, 6.0, 8.0, S_NAN_F64])
            .reduce_max_precise(),
        12.0
    );
    assert_eq!(
        f64x8::from_slice(
            simd,
            &[Q_NAN_F64, 4.0, S_NAN_F64, 7.0, 12.0, S_NAN_F64, 6.0, 8.0]
        )
        .reduce_max_precise(),
        12.0
    );
    assert!(
        f64x8::from_slice(simd, &[Q_NAN_F64; 8])
            .reduce_max_precise()
            .is_nan()
    );
    assert_eq!(
        f64x8::from_slice(
            simd,
            &[
                f64::NEG_INFINITY,
                S_NAN_F64,
                3.0,
                8.0,
                2.0,
                f64::INFINITY,
                5.0,
                7.0
            ],
        )
        .reduce_max_precise(),
        f64::INFINITY
    );
    let zeros = f64x8::from_fn(simd, |i| if i % 2 == 0 { 0.0 } else { -0.0 });
    assert_eq!(zeros.reduce_max_precise(), 0.0);
}

#[simd_test]
fn reduce_max_precise_i8x16_alias<S: Simd>(simd: S) {
    let value = i8x16::from_slice(
        simd,
        &[
            12,
            -7,
            44,
            i8::MAX,
            -99,
            0,
            63,
            -42,
            8,
            91,
            -1,
            37,
            -64,
            i8::MIN,
            5,
            72,
        ],
    );
    assert_eq!(value.reduce_max_precise(), i8::MAX);
}

#[simd_test]
fn reduce_max_precise_u64x8_alias<S: Simd>(simd: S) {
    let value = u64x8::from_slice(
        simd,
        &[
            42,
            u64::MAX,
            7_000_000_000,
            81,
            19_000_000,
            u64::MIN,
            5_400_000_000,
            12_345,
        ],
    );
    assert_eq!(value.reduce_max_precise(), u64::MAX);
}
