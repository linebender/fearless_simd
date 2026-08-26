// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported integer vector type.

#[simd_test]
fn saturating_add_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            -25,
            25,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
        ]
    );
}

#[simd_test]
fn saturating_add_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            -25,
            25,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            -25,
            25,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
        ]
    );
}

#[simd_test]
fn saturating_add_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX - 10,
            i8::MIN + 10,
            1,
            -1,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i8::MAX,
            i8::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            -25,
            25,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            -25,
            25,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            -25,
            25,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            60,
            -60,
            -25,
            25,
            i8::MAX,
            i8::MIN,
            i8::MAX,
            i8::MIN,
            i8::MAX - 1,
            i8::MIN + 1,
            0,
            0,
        ]
    );
}

#[simd_test]
fn saturating_add_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[
            u8::MAX,
            u8::MAX - 1,
            u8::MAX - 1,
            0,
            40,
            100,
            u8::MAX - 10,
            10,
            0,
            1,
            0,
            u8::MAX,
            u8::MAX / 2,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            17,
        ],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            1,
            2,
            1,
            u8::MAX,
            20,
            20,
            9,
            u8::MAX - 5,
            0,
            0,
            1,
            0,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            u8::MAX / 2 + 1,
            23,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            60,
            120,
            u8::MAX - 1,
            u8::MAX,
            0,
            1,
            1,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX,
            u8::MAX,
            40,
        ]
    );
}

#[simd_test]
fn saturating_add_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            u8::MAX,
            u8::MAX - 1,
            u8::MAX - 1,
            0,
            40,
            100,
            u8::MAX - 10,
            10,
            0,
            1,
            0,
            u8::MAX,
            u8::MAX / 2,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            17,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX - 1,
            0,
            40,
            100,
            u8::MAX - 10,
            10,
            0,
            1,
            0,
            u8::MAX,
            u8::MAX / 2,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            17,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            1,
            2,
            1,
            u8::MAX,
            20,
            20,
            9,
            u8::MAX - 5,
            0,
            0,
            1,
            0,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            u8::MAX / 2 + 1,
            23,
            1,
            2,
            1,
            u8::MAX,
            20,
            20,
            9,
            u8::MAX - 5,
            0,
            0,
            1,
            0,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            u8::MAX / 2 + 1,
            23,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            60,
            120,
            u8::MAX - 1,
            u8::MAX,
            0,
            1,
            1,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX,
            u8::MAX,
            40,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            60,
            120,
            u8::MAX - 1,
            u8::MAX,
            0,
            1,
            1,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX,
            u8::MAX,
            40,
        ]
    );
}

#[simd_test]
fn saturating_add_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            u8::MAX,
            u8::MAX - 1,
            u8::MAX - 1,
            0,
            40,
            100,
            u8::MAX - 10,
            10,
            0,
            1,
            0,
            u8::MAX,
            u8::MAX / 2,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            17,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX - 1,
            0,
            40,
            100,
            u8::MAX - 10,
            10,
            0,
            1,
            0,
            u8::MAX,
            u8::MAX / 2,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            17,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX - 1,
            0,
            40,
            100,
            u8::MAX - 10,
            10,
            0,
            1,
            0,
            u8::MAX,
            u8::MAX / 2,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            17,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX - 1,
            0,
            40,
            100,
            u8::MAX - 10,
            10,
            0,
            1,
            0,
            u8::MAX,
            u8::MAX / 2,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            17,
        ],
    );
    let b = u8x64::from_slice(
        simd,
        &[
            1,
            2,
            1,
            u8::MAX,
            20,
            20,
            9,
            u8::MAX - 5,
            0,
            0,
            1,
            0,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            u8::MAX / 2 + 1,
            23,
            1,
            2,
            1,
            u8::MAX,
            20,
            20,
            9,
            u8::MAX - 5,
            0,
            0,
            1,
            0,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            u8::MAX / 2 + 1,
            23,
            1,
            2,
            1,
            u8::MAX,
            20,
            20,
            9,
            u8::MAX - 5,
            0,
            0,
            1,
            0,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            u8::MAX / 2 + 1,
            23,
            1,
            2,
            1,
            u8::MAX,
            20,
            20,
            9,
            u8::MAX - 5,
            0,
            0,
            1,
            0,
            u8::MAX / 2,
            u8::MAX / 2 + 1,
            u8::MAX / 2 + 1,
            23,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            60,
            120,
            u8::MAX - 1,
            u8::MAX,
            0,
            1,
            1,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX,
            u8::MAX,
            40,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            60,
            120,
            u8::MAX - 1,
            u8::MAX,
            0,
            1,
            1,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX,
            u8::MAX,
            40,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            60,
            120,
            u8::MAX - 1,
            u8::MAX,
            0,
            1,
            1,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX,
            u8::MAX,
            40,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            u8::MAX,
            60,
            120,
            u8::MAX - 1,
            u8::MAX,
            0,
            1,
            1,
            u8::MAX,
            u8::MAX - 1,
            u8::MAX,
            u8::MAX,
            40,
        ]
    );
}

#[simd_test]
fn saturating_add_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(
        simd,
        &[
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            40,
            -40,
            50,
            -50,
        ],
    );
    let b = i16x8::from_slice(simd, &[1, -1, 1, -1, 20, -20, -75, 75]);
    assert_eq!(
        *a.saturating_add(b),
        [i16::MAX, i16::MIN, i16::MAX, i16::MIN, 60, -60, -25, 25]
    );
}

#[simd_test]
fn saturating_add_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i16::MAX,
            i16::MIN,
            i16::MAX - 10,
            i16::MIN + 10,
            1,
            -1,
        ],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i16::MAX,
            i16::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            60,
            -60,
            -25,
            25,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            0,
            0,
        ]
    );
}

#[simd_test]
fn saturating_add_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i16::MAX,
            i16::MIN,
            i16::MAX - 10,
            i16::MIN + 10,
            1,
            -1,
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i16::MAX,
            i16::MIN,
            i16::MAX - 10,
            i16::MIN + 10,
            1,
            -1,
        ],
    );
    let b = i16x32::from_slice(
        simd,
        &[
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i16::MAX,
            i16::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i16::MAX,
            i16::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            60,
            -60,
            -25,
            25,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            0,
            0,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            60,
            -60,
            -25,
            25,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            i16::MAX - 1,
            i16::MIN + 1,
            0,
            0,
        ]
    );
}

#[simd_test]
fn saturating_add_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(
        simd,
        &[
            u16::MAX,
            u16::MAX - 1,
            u16::MAX - 1,
            0,
            40,
            100,
            u16::MAX - 10,
            10,
        ],
    );
    let b = u16x8::from_slice(simd, &[1, 2, 1, u16::MAX, 20, 20, 9, u16::MAX - 5]);
    assert_eq!(
        *a.saturating_add(b),
        [
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            60,
            120,
            u16::MAX - 1,
            u16::MAX
        ]
    );
}

#[simd_test]
fn saturating_add_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[
            u16::MAX,
            u16::MAX - 1,
            u16::MAX - 1,
            0,
            40,
            100,
            u16::MAX - 10,
            10,
            0,
            1,
            0,
            u16::MAX,
            u16::MAX / 2,
            u16::MAX / 2,
            u16::MAX / 2 + 1,
            17,
        ],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            1,
            2,
            1,
            u16::MAX,
            20,
            20,
            9,
            u16::MAX - 5,
            0,
            0,
            1,
            0,
            u16::MAX / 2,
            u16::MAX / 2 + 1,
            u16::MAX / 2 + 1,
            23,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            60,
            120,
            u16::MAX - 1,
            u16::MAX,
            0,
            1,
            1,
            u16::MAX,
            u16::MAX - 1,
            u16::MAX,
            u16::MAX,
            40,
        ]
    );
}

#[simd_test]
fn saturating_add_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            u16::MAX,
            u16::MAX - 1,
            u16::MAX - 1,
            0,
            40,
            100,
            u16::MAX - 10,
            10,
            0,
            1,
            0,
            u16::MAX,
            u16::MAX / 2,
            u16::MAX / 2,
            u16::MAX / 2 + 1,
            17,
            u16::MAX,
            u16::MAX - 1,
            u16::MAX - 1,
            0,
            40,
            100,
            u16::MAX - 10,
            10,
            0,
            1,
            0,
            u16::MAX,
            u16::MAX / 2,
            u16::MAX / 2,
            u16::MAX / 2 + 1,
            17,
        ],
    );
    let b = u16x32::from_slice(
        simd,
        &[
            1,
            2,
            1,
            u16::MAX,
            20,
            20,
            9,
            u16::MAX - 5,
            0,
            0,
            1,
            0,
            u16::MAX / 2,
            u16::MAX / 2 + 1,
            u16::MAX / 2 + 1,
            23,
            1,
            2,
            1,
            u16::MAX,
            20,
            20,
            9,
            u16::MAX - 5,
            0,
            0,
            1,
            0,
            u16::MAX / 2,
            u16::MAX / 2 + 1,
            u16::MAX / 2 + 1,
            23,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            60,
            120,
            u16::MAX - 1,
            u16::MAX,
            0,
            1,
            1,
            u16::MAX,
            u16::MAX - 1,
            u16::MAX,
            u16::MAX,
            40,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            u16::MAX,
            60,
            120,
            u16::MAX - 1,
            u16::MAX,
            0,
            1,
            1,
            u16::MAX,
            u16::MAX - 1,
            u16::MAX,
            u16::MAX,
            40,
        ]
    );
}

#[simd_test]
fn saturating_add_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[i32::MAX, i32::MIN, i32::MAX - 1, i32::MIN + 1]);
    let b = i32x4::from_slice(simd, &[1, -1, 1, -1]);
    assert_eq!(
        *a.saturating_add(b),
        [i32::MAX, i32::MIN, i32::MAX, i32::MIN]
    );

    let scalar_rhs = i32x4::from_slice(simd, &[i32::MAX, i32::MIN, 40, -40]);
    assert_eq!(
        *scalar_rhs.saturating_add(20),
        [i32::MAX, i32::MIN + 20, 60, -20]
    );
}

#[simd_test]
fn saturating_add_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(
        simd,
        &[
            i32::MAX,
            i32::MIN,
            i32::MAX - 1,
            i32::MIN + 1,
            40,
            -40,
            50,
            -50,
        ],
    );
    let b = i32x8::from_slice(simd, &[1, -1, 1, -1, 20, -20, -75, 75]);
    assert_eq!(
        *a.saturating_add(b),
        [i32::MAX, i32::MIN, i32::MAX, i32::MIN, 60, -60, -25, 25]
    );
}

#[simd_test]
fn saturating_add_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[
            i32::MAX,
            i32::MIN,
            i32::MAX - 1,
            i32::MIN + 1,
            40,
            -40,
            50,
            -50,
            0,
            0,
            i32::MAX,
            i32::MIN,
            i32::MAX - 10,
            i32::MIN + 10,
            1,
            -1,
        ],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            1,
            -1,
            1,
            -1,
            20,
            -20,
            -75,
            75,
            i32::MAX,
            i32::MIN,
            0,
            0,
            9,
            -9,
            -1,
            1,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            i32::MAX,
            i32::MIN,
            i32::MAX,
            i32::MIN,
            60,
            -60,
            -25,
            25,
            i32::MAX,
            i32::MIN,
            i32::MAX,
            i32::MIN,
            i32::MAX - 1,
            i32::MIN + 1,
            0,
            0,
        ]
    );
}

#[simd_test]
fn saturating_add_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[u32::MAX, u32::MAX - 1, 40, 0]);
    let b = u32x4::from_slice(simd, &[1, 2, 20, u32::MAX]);
    assert_eq!(*a.saturating_add(b), [u32::MAX, u32::MAX, 60, u32::MAX]);
}

#[simd_test]
fn saturating_add_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(
        simd,
        &[
            u32::MAX,
            u32::MAX - 1,
            u32::MAX - 1,
            0,
            40,
            100,
            u32::MAX - 10,
            10,
        ],
    );
    let b = u32x8::from_slice(simd, &[1, 2, 1, u32::MAX, 20, 20, 9, u32::MAX - 5]);
    assert_eq!(
        *a.saturating_add(b),
        [
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            60,
            120,
            u32::MAX - 1,
            u32::MAX
        ]
    );
}

#[simd_test]
fn saturating_add_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            u32::MAX,
            u32::MAX - 1,
            u32::MAX - 1,
            0,
            40,
            100,
            u32::MAX - 10,
            10,
            0,
            1,
            0,
            u32::MAX,
            u32::MAX / 2,
            u32::MAX / 2,
            u32::MAX / 2 + 1,
            17,
        ],
    );
    let b = u32x16::from_slice(
        simd,
        &[
            1,
            2,
            1,
            u32::MAX,
            20,
            20,
            9,
            u32::MAX - 5,
            0,
            0,
            1,
            0,
            u32::MAX / 2,
            u32::MAX / 2 + 1,
            u32::MAX / 2 + 1,
            23,
        ],
    );
    assert_eq!(
        *a.saturating_add(b),
        [
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            60,
            120,
            u32::MAX - 1,
            u32::MAX,
            0,
            1,
            1,
            u32::MAX,
            u32::MAX - 1,
            u32::MAX,
            u32::MAX,
            40,
        ]
    );
}

#[simd_test]
fn saturating_add_i64x2<S: Simd>(simd: S) {
    let overflow = i64x2::from_slice(simd, &[i64::MAX, i64::MIN]);
    let overflow_rhs = i64x2::from_slice(simd, &[1, -1]);
    assert_eq!(*overflow.saturating_add(overflow_rhs), [i64::MAX, i64::MIN]);

    let boundary = i64x2::from_slice(simd, &[i64::MAX - 1, i64::MIN + 1]);
    let boundary_rhs = i64x2::from_slice(simd, &[1, -1]);
    assert_eq!(*boundary.saturating_add(boundary_rhs), [i64::MAX, i64::MIN]);

    let ordinary = i64x2::from_slice(simd, &[40, -40]);
    let ordinary_rhs = i64x2::from_slice(simd, &[20, -20]);
    assert_eq!(*ordinary.saturating_add(ordinary_rhs), [60, -60]);

    let mixed = i64x2::from_slice(simd, &[50, -50]);
    let mixed_rhs = i64x2::from_slice(simd, &[-75, 75]);
    assert_eq!(*mixed.saturating_add(mixed_rhs), [-25, 25]);
}

#[simd_test]
fn saturating_add_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[i64::MAX, i64::MIN, 40, -40]);
    let b = i64x4::from_slice(simd, &[1, -1, 20, -20]);
    assert_eq!(*a.saturating_add(b), [i64::MAX, i64::MIN, 60, -60]);

    let boundary = i64x4::from_slice(simd, &[i64::MAX - 1, i64::MIN + 1, 50, -50]);
    let boundary_rhs = i64x4::from_slice(simd, &[1, -1, -75, 75]);
    assert_eq!(
        *boundary.saturating_add(boundary_rhs),
        [i64::MAX, i64::MIN, -25, 25]
    );
}

#[simd_test]
fn saturating_add_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[
            i64::MAX,
            i64::MIN,
            i64::MAX - 1,
            i64::MIN + 1,
            40,
            -40,
            50,
            -50,
        ],
    );
    let b = i64x8::from_slice(simd, &[1, -1, 1, -1, 20, -20, -75, 75]);
    assert_eq!(
        *a.saturating_add(b),
        [i64::MAX, i64::MIN, i64::MAX, i64::MIN, 60, -60, -25, 25]
    );
}

#[simd_test]
fn saturating_add_u64x2<S: Simd>(simd: S) {
    let overflow = u64x2::from_slice(simd, &[u64::MAX, u64::MAX - 1]);
    let overflow_rhs = u64x2::from_slice(simd, &[1, 2]);
    assert_eq!(*overflow.saturating_add(overflow_rhs), [u64::MAX, u64::MAX]);

    let boundary = u64x2::from_slice(simd, &[u64::MAX - 1, 0]);
    let boundary_rhs = u64x2::from_slice(simd, &[1, u64::MAX]);
    assert_eq!(*boundary.saturating_add(boundary_rhs), [u64::MAX, u64::MAX]);

    let ordinary = u64x2::from_slice(simd, &[40, u64::MAX - 10]);
    let ordinary_rhs = u64x2::from_slice(simd, &[20, 9]);
    assert_eq!(*ordinary.saturating_add(ordinary_rhs), [60, u64::MAX - 1]);

    let mixed = u64x2::from_slice(simd, &[10, u64::MAX / 2 + 1]);
    let mixed_rhs = u64x2::from_slice(simd, &[u64::MAX - 5, u64::MAX / 2 + 1]);
    assert_eq!(*mixed.saturating_add(mixed_rhs), [u64::MAX, u64::MAX]);
}

#[simd_test]
fn saturating_add_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[u64::MAX, u64::MAX - 1, 40, 0]);
    let b = u64x4::from_slice(simd, &[1, 2, 20, u64::MAX]);
    assert_eq!(*a.saturating_add(b), [u64::MAX, u64::MAX, 60, u64::MAX]);

    let boundary = u64x4::from_slice(simd, &[u64::MAX - 1, u64::MAX - 10, u64::MAX / 2, 17]);
    let boundary_rhs = u64x4::from_slice(simd, &[1, 9, u64::MAX / 2, 23]);
    assert_eq!(
        *boundary.saturating_add(boundary_rhs),
        [u64::MAX, u64::MAX - 1, u64::MAX - 1, 40]
    );
}

#[simd_test]
fn saturating_add_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[
            u64::MAX,
            u64::MAX - 1,
            u64::MAX - 1,
            0,
            40,
            100,
            u64::MAX - 10,
            10,
        ],
    );
    let b = u64x8::from_slice(simd, &[1, 2, 1, u64::MAX, 20, 20, 9, u64::MAX - 5]);
    assert_eq!(
        *a.saturating_add(b),
        [
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            60,
            120,
            u64::MAX - 1,
            u64::MAX
        ]
    );
}
