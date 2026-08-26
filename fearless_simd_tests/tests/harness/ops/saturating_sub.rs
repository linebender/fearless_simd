// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported integer vector type.

#[simd_test]
fn saturating_sub_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            100,
            -100,
            50,
            -50,
            0,
            0,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            40,
            -40,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            -1,
            1,
            -1,
            1,
            40,
            -40,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            -9,
            9,
            1,
            -1,
            -20,
            20,
        ],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            125,
            -125,
            -i8::MAX,
            i8::MAX,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            60,
            -60
        ]
    );
}

#[simd_test]
fn saturating_sub_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            100,
            -100,
            50,
            -50,
            0,
            0,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            40,
            -40,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            100,
            -100,
            50,
            -50,
            0,
            0,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            40,
            -40,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            -1,
            1,
            -1,
            1,
            40,
            -40,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            -9,
            9,
            1,
            -1,
            -20,
            20,
            -1,
            1,
            -1,
            1,
            40,
            -40,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            -9,
            9,
            1,
            -1,
            -20,
            20,
        ],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            125,
            -125,
            -i8::MAX,
            i8::MAX,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            60,
            -60,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            125,
            -125,
            -i8::MAX,
            i8::MAX,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            60,
            -60
        ]
    );
}

#[simd_test]
fn saturating_sub_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            100,
            -100,
            50,
            -50,
            0,
            0,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            40,
            -40,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            100,
            -100,
            50,
            -50,
            0,
            0,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            40,
            -40,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            100,
            -100,
            50,
            -50,
            0,
            0,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            40,
            -40,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            100,
            -100,
            50,
            -50,
            0,
            0,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            40,
            -40,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            -1,
            1,
            -1,
            1,
            40,
            -40,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            -9,
            9,
            1,
            -1,
            -20,
            20,
            -1,
            1,
            -1,
            1,
            40,
            -40,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            -9,
            9,
            1,
            -1,
            -20,
            20,
            -1,
            1,
            -1,
            1,
            40,
            -40,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            -9,
            9,
            1,
            -1,
            -20,
            20,
            -1,
            1,
            -1,
            1,
            40,
            -40,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            -9,
            9,
            1,
            -1,
            -20,
            20,
        ],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            125,
            -125,
            -i8::MAX,
            i8::MAX,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            60,
            -60,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            125,
            -125,
            -i8::MAX,
            i8::MAX,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            60,
            -60,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            125,
            -125,
            -i8::MAX,
            i8::MAX,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            60,
            -60,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            125,
            -125,
            -i8::MAX,
            i8::MAX,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            60,
            -60
        ]
    );
}

#[simd_test]
fn saturating_sub_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[
            100,
            5,
            50,
            0,
            u8::MAX,
            u8::MAX,
            1,
            20,
            200,
            10,
            0,
            u8::MAX - 1,
            60,
            40,
            u8::MAX,
            128,
        ],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            40,
            10,
            50,
            0,
            1,
            0,
            1,
            30,
            100,
            10,
            1,
            u8::MAX,
            20,
            60,
            u8::MAX,
            128,
        ],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            60,
            0,
            0,
            0,
            u8::MAX - 1,
            u8::MAX,
            0,
            0,
            100,
            0,
            0,
            0,
            40,
            0,
            0,
            0
        ]
    );
}

#[simd_test]
fn saturating_sub_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            100,
            5,
            50,
            0,
            u8::MAX,
            u8::MAX,
            1,
            20,
            200,
            10,
            0,
            u8::MAX - 1,
            60,
            40,
            u8::MAX,
            128,
            100,
            5,
            50,
            0,
            u8::MAX,
            u8::MAX,
            1,
            20,
            200,
            10,
            0,
            u8::MAX - 1,
            60,
            40,
            u8::MAX,
            128,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            40,
            10,
            50,
            0,
            1,
            0,
            1,
            30,
            100,
            10,
            1,
            u8::MAX,
            20,
            60,
            u8::MAX,
            128,
            40,
            10,
            50,
            0,
            1,
            0,
            1,
            30,
            100,
            10,
            1,
            u8::MAX,
            20,
            60,
            u8::MAX,
            128,
        ],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            60,
            0,
            0,
            0,
            u8::MAX - 1,
            u8::MAX,
            0,
            0,
            100,
            0,
            0,
            0,
            40,
            0,
            0,
            0,
            60,
            0,
            0,
            0,
            u8::MAX - 1,
            u8::MAX,
            0,
            0,
            100,
            0,
            0,
            0,
            40,
            0,
            0,
            0
        ]
    );
}

#[simd_test]
fn saturating_sub_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            100,
            5,
            50,
            0,
            u8::MAX,
            u8::MAX,
            1,
            20,
            200,
            10,
            0,
            u8::MAX - 1,
            60,
            40,
            u8::MAX,
            128,
            100,
            5,
            50,
            0,
            u8::MAX,
            u8::MAX,
            1,
            20,
            200,
            10,
            0,
            u8::MAX - 1,
            60,
            40,
            u8::MAX,
            128,
            100,
            5,
            50,
            0,
            u8::MAX,
            u8::MAX,
            1,
            20,
            200,
            10,
            0,
            u8::MAX - 1,
            60,
            40,
            u8::MAX,
            128,
            100,
            5,
            50,
            0,
            u8::MAX,
            u8::MAX,
            1,
            20,
            200,
            10,
            0,
            u8::MAX - 1,
            60,
            40,
            u8::MAX,
            128,
        ],
    );
    let b = u8x64::from_slice(
        simd,
        &[
            40,
            10,
            50,
            0,
            1,
            0,
            1,
            30,
            100,
            10,
            1,
            u8::MAX,
            20,
            60,
            u8::MAX,
            128,
            40,
            10,
            50,
            0,
            1,
            0,
            1,
            30,
            100,
            10,
            1,
            u8::MAX,
            20,
            60,
            u8::MAX,
            128,
            40,
            10,
            50,
            0,
            1,
            0,
            1,
            30,
            100,
            10,
            1,
            u8::MAX,
            20,
            60,
            u8::MAX,
            128,
            40,
            10,
            50,
            0,
            1,
            0,
            1,
            30,
            100,
            10,
            1,
            u8::MAX,
            20,
            60,
            u8::MAX,
            128,
        ],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            60,
            0,
            0,
            0,
            u8::MAX - 1,
            u8::MAX,
            0,
            0,
            100,
            0,
            0,
            0,
            40,
            0,
            0,
            0,
            60,
            0,
            0,
            0,
            u8::MAX - 1,
            u8::MAX,
            0,
            0,
            100,
            0,
            0,
            0,
            40,
            0,
            0,
            0,
            60,
            0,
            0,
            0,
            u8::MAX - 1,
            u8::MAX,
            0,
            0,
            100,
            0,
            0,
            0,
            40,
            0,
            0,
            0,
            60,
            0,
            0,
            0,
            u8::MAX - 1,
            u8::MAX,
            0,
            0,
            100,
            0,
            0,
            0,
            40,
            0,
            0,
            0
        ]
    );
}

#[simd_test]
fn saturating_sub_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(
        simd,
        &[
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            100,
            -100,
            0,
            0,
        ],
    );
    let b = i16x8::from_slice(simd, &[-1, 1, -1, 1, 40, -40, i16::MAX, i16::MIN]);
    assert_eq!(
        *a.saturating_sub(b),
        [
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            60,
            -60,
            -i16::MAX,
            i16::MAX
        ]
    );
}

#[simd_test]
fn saturating_sub_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            100,
            -100,
            0,
            0,
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            100,
            -100,
            0,
            0,
        ],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            -1,
            1,
            -1,
            1,
            40,
            -40,
            i16::MAX,
            i16::MIN,
            -1,
            1,
            -1,
            1,
            40,
            -40,
            i16::MAX,
            i16::MIN,
        ],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            60,
            -60,
            -i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            60,
            -60,
            -i16::MAX,
            i16::MAX
        ]
    );
}

#[simd_test]
fn saturating_sub_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            100,
            -100,
            0,
            0,
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            100,
            -100,
            0,
            0,
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            100,
            -100,
            0,
            0,
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            100,
            -100,
            0,
            0,
        ],
    );
    let b = i16x32::from_slice(
        simd,
        &[
            -1,
            1,
            -1,
            1,
            40,
            -40,
            i16::MAX,
            i16::MIN,
            -1,
            1,
            -1,
            1,
            40,
            -40,
            i16::MAX,
            i16::MIN,
            -1,
            1,
            -1,
            1,
            40,
            -40,
            i16::MAX,
            i16::MIN,
            -1,
            1,
            -1,
            1,
            40,
            -40,
            i16::MAX,
            i16::MIN,
        ],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            60,
            -60,
            -i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            60,
            -60,
            -i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            60,
            -60,
            -i16::MAX,
            i16::MAX,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            60,
            -60,
            -i16::MAX,
            i16::MAX
        ]
    );
}

#[simd_test]
fn saturating_sub_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[100, 5, 50, 0, u16::MAX, u16::MAX, 1, 20]);
    let b = u16x8::from_slice(simd, &[40, 10, 50, 0, 1, 0, 1, 30]);
    assert_eq!(
        *a.saturating_sub(b),
        [60, 0, 0, 0, u16::MAX - 1, u16::MAX, 0, 0]
    );
}

#[simd_test]
fn saturating_sub_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[
            100,
            5,
            50,
            0,
            u16::MAX,
            u16::MAX,
            1,
            20,
            100,
            5,
            50,
            0,
            u16::MAX,
            u16::MAX,
            1,
            20,
        ],
    );
    let b = u16x16::from_slice(
        simd,
        &[40, 10, 50, 0, 1, 0, 1, 30, 40, 10, 50, 0, 1, 0, 1, 30],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            60,
            0,
            0,
            0,
            u16::MAX - 1,
            u16::MAX,
            0,
            0,
            60,
            0,
            0,
            0,
            u16::MAX - 1,
            u16::MAX,
            0,
            0
        ]
    );
}

#[simd_test]
fn saturating_sub_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            100,
            5,
            50,
            0,
            u16::MAX,
            u16::MAX,
            1,
            20,
            100,
            5,
            50,
            0,
            u16::MAX,
            u16::MAX,
            1,
            20,
            100,
            5,
            50,
            0,
            u16::MAX,
            u16::MAX,
            1,
            20,
            100,
            5,
            50,
            0,
            u16::MAX,
            u16::MAX,
            1,
            20,
        ],
    );
    let b = u16x32::from_slice(
        simd,
        &[
            40, 10, 50, 0, 1, 0, 1, 30, 40, 10, 50, 0, 1, 0, 1, 30, 40, 10, 50, 0, 1, 0, 1, 30, 40,
            10, 50, 0, 1, 0, 1, 30,
        ],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            60,
            0,
            0,
            0,
            u16::MAX - 1,
            u16::MAX,
            0,
            0,
            60,
            0,
            0,
            0,
            u16::MAX - 1,
            u16::MAX,
            0,
            0,
            60,
            0,
            0,
            0,
            u16::MAX - 1,
            u16::MAX,
            0,
            0,
            60,
            0,
            0,
            0,
            u16::MAX - 1,
            u16::MAX,
            0,
            0
        ]
    );
}

#[simd_test]
fn saturating_sub_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[i32::MAX, i32::MIN, 100, 0]);
    let b = i32x4::from_slice(simd, &[-1, 1, 40, i32::MIN]);
    assert_eq!(*a.saturating_sub(b), [i32::MAX, i32::MIN, 60, i32::MAX]);
    let scalar_rhs = i32x4::from_slice(simd, &[i32::MAX, i32::MIN, 40, -40]);
    assert_eq!(
        *scalar_rhs.saturating_sub(20),
        [i32::MAX - 20, i32::MIN, 20, -60]
    );
}

#[simd_test]
fn saturating_sub_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(
        simd,
        &[i32::MAX, i32::MIN, 100, 0, i32::MAX, i32::MIN, 100, 0],
    );
    let b = i32x8::from_slice(simd, &[-1, 1, 40, i32::MIN, -1, 1, 40, i32::MIN]);
    assert_eq!(
        *a.saturating_sub(b),
        [
            i32::MAX,
            i32::MIN,
            60,
            i32::MAX,
            i32::MAX,
            i32::MIN,
            60,
            i32::MAX
        ]
    );
}

#[simd_test]
fn saturating_sub_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[
            i32::MAX,
            i32::MIN,
            100,
            0,
            i32::MAX,
            i32::MIN,
            100,
            0,
            i32::MAX,
            i32::MIN,
            100,
            0,
            i32::MAX,
            i32::MIN,
            100,
            0,
        ],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            -1,
            1,
            40,
            i32::MIN,
            -1,
            1,
            40,
            i32::MIN,
            -1,
            1,
            40,
            i32::MIN,
            -1,
            1,
            40,
            i32::MIN,
        ],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [
            i32::MAX,
            i32::MIN,
            60,
            i32::MAX,
            i32::MAX,
            i32::MIN,
            60,
            i32::MAX,
            i32::MAX,
            i32::MIN,
            60,
            i32::MAX,
            i32::MAX,
            i32::MIN,
            60,
            i32::MAX
        ]
    );
}

#[simd_test]
fn saturating_sub_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[100, 5, 50, 0]);
    let b = u32x4::from_slice(simd, &[40, 10, 50, 1]);
    assert_eq!(*a.saturating_sub(b), [60, 0, 0, 0]);
}

#[simd_test]
fn saturating_sub_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[100, 5, 50, 0, 100, 5, 50, 0]);
    let b = u32x8::from_slice(simd, &[40, 10, 50, 1, 40, 10, 50, 1]);
    assert_eq!(*a.saturating_sub(b), [60, 0, 0, 0, 60, 0, 0, 0]);
}

#[simd_test]
fn saturating_sub_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[100, 5, 50, 0, 100, 5, 50, 0, 100, 5, 50, 0, 100, 5, 50, 0],
    );
    let b = u32x16::from_slice(
        simd,
        &[40, 10, 50, 1, 40, 10, 50, 1, 40, 10, 50, 1, 40, 10, 50, 1],
    );
    assert_eq!(
        *a.saturating_sub(b),
        [60, 0, 0, 0, 60, 0, 0, 0, 60, 0, 0, 0, 60, 0, 0, 0]
    );
}

#[simd_test]
fn saturating_sub_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[i64::MAX, i64::MIN]);
    let b = i64x2::from_slice(simd, &[-1, 1]);
    assert_eq!(*a.saturating_sub(b), [i64::MAX, i64::MIN]);
    let boundary = i64x2::from_slice(simd, &[i64::MAX - 1, i64::MIN + 1]);
    let boundary_rhs = i64x2::from_slice(simd, &[-1, 1]);
    assert_eq!(*boundary.saturating_sub(boundary_rhs), [i64::MAX, i64::MIN]);

    let ordinary = i64x2::from_slice(simd, &[100, -100]);
    let ordinary_rhs = i64x2::from_slice(simd, &[40, -40]);
    assert_eq!(*ordinary.saturating_sub(ordinary_rhs), [60, -60]);

    let mixed = i64x2::from_slice(simd, &[50, -50]);
    let mixed_rhs = i64x2::from_slice(simd, &[-75, 75]);
    assert_eq!(*mixed.saturating_sub(mixed_rhs), [125, -125]);

    let zero = i64x2::from_slice(simd, &[0, 0]);
    let zero_rhs = i64x2::from_slice(simd, &[i64::MAX, i64::MIN]);
    assert_eq!(*zero.saturating_sub(zero_rhs), [-i64::MAX, i64::MAX]);
}

#[simd_test]
fn saturating_sub_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[i64::MAX, i64::MIN, 100, -100]);
    let b = i64x4::from_slice(simd, &[-1, 1, 40, -40]);
    assert_eq!(*a.saturating_sub(b), [i64::MAX, i64::MIN, 60, -60]);
    let boundary = i64x4::from_slice(simd, &[i64::MAX - 1, i64::MIN + 1, 50, -50]);
    let boundary_rhs = i64x4::from_slice(simd, &[-1, 1, -75, 75]);
    assert_eq!(
        *boundary.saturating_sub(boundary_rhs),
        [i64::MAX, i64::MIN, 125, -125]
    );
}

#[simd_test]
fn saturating_sub_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[
            i64::MAX,
            i64::MIN,
            i64::MAX - 1,
            i64::MIN + 1,
            100,
            -100,
            0,
            0,
        ],
    );
    let b = i64x8::from_slice(simd, &[-1, 1, -1, 1, 40, -40, i64::MAX, i64::MIN]);
    assert_eq!(
        *a.saturating_sub(b),
        [
            i64::MAX,
            i64::MIN,
            i64::MAX,
            i64::MIN,
            60,
            -60,
            -i64::MAX,
            i64::MAX
        ]
    );
}

#[simd_test]
fn saturating_sub_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[100, 5]);
    let b = u64x2::from_slice(simd, &[40, 10]);
    assert_eq!(*a.saturating_sub(b), [60, 0]);
    let boundary = u64x2::from_slice(simd, &[u64::MAX, u64::MAX]);
    let boundary_rhs = u64x2::from_slice(simd, &[1, 0]);
    assert_eq!(
        *boundary.saturating_sub(boundary_rhs),
        [u64::MAX - 1, u64::MAX]
    );

    let exact = u64x2::from_slice(simd, &[50, 0]);
    let exact_rhs = u64x2::from_slice(simd, &[50, 0]);
    assert_eq!(*exact.saturating_sub(exact_rhs), [0, 0]);

    let floor = u64x2::from_slice(simd, &[0, 1]);
    let floor_rhs = u64x2::from_slice(simd, &[1, 1]);
    assert_eq!(*floor.saturating_sub(floor_rhs), [0, 0]);
}

#[simd_test]
fn saturating_sub_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[100, 5, u64::MAX, 0]);
    let b = u64x4::from_slice(simd, &[40, 10, 1, 1]);
    assert_eq!(*a.saturating_sub(b), [60, 0, u64::MAX - 1, 0]);
}

#[simd_test]
fn saturating_sub_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[100, 5, 50, 0, u64::MAX, u64::MAX, 1, 20]);
    let b = u64x8::from_slice(simd, &[40, 10, 50, 0, 1, 0, 1, 30]);
    assert_eq!(
        *a.saturating_sub(b),
        [60, 0, 0, 0, u64::MAX - 1, u64::MAX, 0, 0]
    );
}
