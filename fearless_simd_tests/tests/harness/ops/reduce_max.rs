// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn reduce_max_f32x4<S: Simd>(simd: S) {
    let value = f32x4::from_slice(simd, &[3.0, f32::INFINITY, -2.0, f32::NEG_INFINITY]);
    assert_eq!(value.reduce_max(), f32::INFINITY);
    assert_eq!(
        f32x4::from_slice(simd, &[0.0, -0.0, 0.0, -0.0]).reduce_max(),
        0.0
    );
}

#[simd_test]
fn reduce_max_i8x16<S: Simd>(simd: S) {
    let value = i8x16::from_fn(simd, |i| if i == 13 { i8::MAX } else { i8::MIN });
    assert_eq!(value.reduce_max(), i8::MAX);
}

#[simd_test]
fn reduce_max_u8x16<S: Simd>(simd: S) {
    let value = u8x16::from_fn(simd, |i| if i == 11 { u8::MAX } else { u8::MIN });
    assert_eq!(value.reduce_max(), u8::MAX);
}

#[simd_test]
fn reduce_max_i16x8<S: Simd>(simd: S) {
    let value = i16x8::from_fn(simd, |i| if i == 6 { i16::MAX } else { i16::MIN });
    assert_eq!(value.reduce_max(), i16::MAX);
}

#[simd_test]
fn reduce_max_u16x8<S: Simd>(simd: S) {
    let value = u16x8::from_fn(simd, |i| if i == 5 { u16::MAX } else { u16::MIN });
    assert_eq!(value.reduce_max(), u16::MAX);
}

#[simd_test]
fn reduce_max_i32x4<S: Simd>(simd: S) {
    let value = i32x4::from_fn(simd, |i| if i == 2 { i32::MAX } else { i32::MIN });
    assert_eq!(value.reduce_max(), i32::MAX);
}

#[simd_test]
fn reduce_max_u32x4<S: Simd>(simd: S) {
    let value = u32x4::from_fn(simd, |i| if i == 3 { u32::MAX } else { u32::MIN });
    assert_eq!(value.reduce_max(), u32::MAX);
}

#[simd_test]
fn reduce_max_f64x2<S: Simd>(simd: S) {
    let value = f64x2::from_slice(simd, &[f64::NEG_INFINITY, f64::INFINITY]);
    assert_eq!(value.reduce_max(), f64::INFINITY);
    assert_eq!(f64x2::from_slice(simd, &[0.0, -0.0]).reduce_max(), 0.0);
}

#[simd_test]
fn reduce_max_i64x2<S: Simd>(simd: S) {
    let value = i64x2::from_slice(simd, &[i64::MIN, i64::MAX]);
    assert_eq!(value.reduce_max(), i64::MAX);
}

#[simd_test]
fn reduce_max_u64x2<S: Simd>(simd: S) {
    let value = u64x2::from_slice(simd, &[u64::MIN, u64::MAX]);
    assert_eq!(value.reduce_max(), u64::MAX);
}

#[simd_test]
fn reduce_max_f32x8<S: Simd>(simd: S) {
    let value = f32x8::from_slice(
        simd,
        &[
            3.0,
            -2.0,
            f32::NEG_INFINITY,
            9.0,
            1.0,
            f32::INFINITY,
            4.0,
            8.0,
        ],
    );
    assert_eq!(value.reduce_max(), f32::INFINITY);
    assert_eq!(
        f32x8::from_slice(simd, &[0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0]).reduce_max(),
        0.0
    );
}

#[simd_test]
fn reduce_max_i8x32<S: Simd>(simd: S) {
    let value = i8x32::from_fn(simd, |i| if i == 29 { i8::MAX } else { i8::MIN });
    assert_eq!(value.reduce_max(), i8::MAX);
}

#[simd_test]
fn reduce_max_u8x32<S: Simd>(simd: S) {
    let value = u8x32::from_fn(simd, |i| if i == 17 { u8::MAX } else { u8::MIN });
    assert_eq!(value.reduce_max(), u8::MAX);
}

#[simd_test]
fn reduce_max_i16x16<S: Simd>(simd: S) {
    let value = i16x16::from_fn(simd, |i| if i == 14 { i16::MAX } else { i16::MIN });
    assert_eq!(value.reduce_max(), i16::MAX);
}

#[simd_test]
fn reduce_max_u16x16<S: Simd>(simd: S) {
    let value = u16x16::from_fn(simd, |i| if i == 9 { u16::MAX } else { u16::MIN });
    assert_eq!(value.reduce_max(), u16::MAX);
}

#[simd_test]
fn reduce_max_i32x8<S: Simd>(simd: S) {
    let value = i32x8::from_fn(simd, |i| if i == 7 { i32::MAX } else { i32::MIN });
    assert_eq!(value.reduce_max(), i32::MAX);
}

#[simd_test]
fn reduce_max_u32x8<S: Simd>(simd: S) {
    let value = u32x8::from_fn(simd, |i| if i == 4 { u32::MAX } else { u32::MIN });
    assert_eq!(value.reduce_max(), u32::MAX);
}

#[simd_test]
fn reduce_max_f64x4<S: Simd>(simd: S) {
    let value = f64x4::from_slice(simd, &[4.0, f64::NEG_INFINITY, -7.0, f64::INFINITY]);
    assert_eq!(value.reduce_max(), f64::INFINITY);
    assert_eq!(
        f64x4::from_slice(simd, &[0.0, -0.0, 0.0, -0.0]).reduce_max(),
        0.0
    );
}

#[simd_test]
fn reduce_max_i64x4<S: Simd>(simd: S) {
    let value = i64x4::from_fn(simd, |i| if i == 2 { i64::MAX } else { i64::MIN });
    assert_eq!(value.reduce_max(), i64::MAX);
}

#[simd_test]
fn reduce_max_u64x4<S: Simd>(simd: S) {
    let value = u64x4::from_fn(simd, |i| if i == 1 { u64::MAX } else { u64::MIN });
    assert_eq!(value.reduce_max(), u64::MAX);
}

#[simd_test]
fn reduce_max_f32x16<S: Simd>(simd: S) {
    let value = f32x16::from_fn(simd, |i| {
        if i == 13 {
            f32::INFINITY
        } else if i == 2 {
            f32::NEG_INFINITY
        } else {
            i as f32 - 4.0
        }
    });
    assert_eq!(value.reduce_max(), f32::INFINITY);
    let zeros = f32x16::from_fn(simd, |i| if i % 2 == 0 { 0.0 } else { -0.0 });
    assert_eq!(zeros.reduce_max(), 0.0);
}

#[simd_test]
fn reduce_max_i8x64<S: Simd>(simd: S) {
    let value = i8x64::from_fn(simd, |i| if i == 61 { i8::MAX } else { i8::MIN });
    assert_eq!(value.reduce_max(), i8::MAX);
}

#[simd_test]
fn reduce_max_u8x64<S: Simd>(simd: S) {
    let value = u8x64::from_fn(simd, |i| if i == 33 { u8::MAX } else { u8::MIN });
    assert_eq!(value.reduce_max(), u8::MAX);
}

#[simd_test]
fn reduce_max_i16x32<S: Simd>(simd: S) {
    let value = i16x32::from_fn(simd, |i| if i == 27 { i16::MAX } else { i16::MIN });
    assert_eq!(value.reduce_max(), i16::MAX);
}

#[simd_test]
fn reduce_max_u16x32<S: Simd>(simd: S) {
    let value = u16x32::from_fn(simd, |i| if i == 18 { u16::MAX } else { u16::MIN });
    assert_eq!(value.reduce_max(), u16::MAX);
}

#[simd_test]
fn reduce_max_i32x16<S: Simd>(simd: S) {
    let value = i32x16::from_fn(simd, |i| if i == 15 { i32::MAX } else { i32::MIN });
    assert_eq!(value.reduce_max(), i32::MAX);
}

#[simd_test]
fn reduce_max_u32x16<S: Simd>(simd: S) {
    let value = u32x16::from_fn(simd, |i| if i == 8 { u32::MAX } else { u32::MIN });
    assert_eq!(value.reduce_max(), u32::MAX);
}

#[simd_test]
fn reduce_max_f64x8<S: Simd>(simd: S) {
    let value = f64x8::from_slice(
        simd,
        &[
            4.0,
            f64::NEG_INFINITY,
            -7.0,
            2.0,
            8.0,
            1.0,
            f64::INFINITY,
            5.0,
        ],
    );
    assert_eq!(value.reduce_max(), f64::INFINITY);
    assert_eq!(
        f64x8::from_slice(simd, &[0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0]).reduce_max(),
        0.0
    );
}

#[simd_test]
fn reduce_max_i64x8<S: Simd>(simd: S) {
    let value = i64x8::from_fn(simd, |i| if i == 6 { i64::MAX } else { i64::MIN });
    assert_eq!(value.reduce_max(), i64::MAX);
}

#[simd_test]
fn reduce_max_u64x8<S: Simd>(simd: S) {
    let value = u64x8::from_fn(simd, |i| if i == 5 { u64::MAX } else { u64::MIN });
    assert_eq!(value.reduce_max(), u64::MAX);
}
