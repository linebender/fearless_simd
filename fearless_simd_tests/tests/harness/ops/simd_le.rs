// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn simd_le_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 3.0, 2.0, 1.0]);
    let b = f32x4::from_slice(simd, &[1.0, 2.0, 2.0, 4.0]);
    assert_eq!(<[i32; 4]>::from(a.simd_le(b)), [0, 0, -1, -1]);
}

#[simd_test]
fn simd_le_u8x16<S: Simd>(simd: S) {
    let vals = u8x16::from_slice(
        simd,
        &[
            0, 12, 34, 50, 220, 180, 127, 128, 255, 50, 33, 126, 0, 0, 0, 0,
        ],
    );
    let mask = vals.simd_le(u8x16::splat(simd, 128));

    assert_eq!(
        <[i8; 16]>::from(mask),
        [-1, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1]
    );
}

#[simd_test]
fn simd_le_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::from_slice(simd, &[1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 8.0, 7.0]);
    assert_eq!(
        <[i32; 8]>::from(a.simd_le(b)),
        [0, 0, -1, -1, -1, -1, -1, 0]
    );
}

#[simd_test]
fn simd_le_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0, 4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 8.0, 7.0, 1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 8.0, 7.0,
        ],
    );
    assert_eq!(
        <[i32; 16]>::from(a.simd_le(b)),
        [0, 0, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0]
    );
}

#[simd_test]
fn simd_le_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f64x8::from_slice(simd, &[1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 8.0, 7.0]);
    assert_eq!(
        <[i64; 8]>::from(a.simd_le(b)),
        [0, 0, -1, -1, -1, -1, -1, 0]
    );
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn simd_le_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[0_i64, 0_i64]);
    assert_eq!(<[i64; 2]>::from(simd.simd_le_i64x2(a, b)), [0_i64, -1_i64]);
}

#[simd_test]
fn simd_le_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[0_i64, 0_i64, 0_i64, 0_i64]);
    assert_eq!(
        <[i64; 4]>::from(simd.simd_le_i64x4(a, b)),
        [0_i64, -1_i64, 0_i64, -1_i64]
    );
}

#[simd_test]
fn simd_le_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64],
    );
    assert_eq!(
        <[i64; 8]>::from(simd.simd_le_i64x8(a, b)),
        [0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64]
    );
}

#[simd_test]
fn simd_le_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[0_u64, 0_u64]);
    assert_eq!(<[i64; 2]>::from(simd.simd_le_u64x2(a, b)), [0_i64, 0_i64]);
}

#[simd_test]
fn simd_le_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[0_u64, 0_u64, 0_u64, 0_u64]);
    assert_eq!(
        <[i64; 4]>::from(simd.simd_le_u64x4(a, b)),
        [0_i64, 0_i64, 0_i64, 0_i64]
    );
}

#[simd_test]
fn simd_le_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64],
    );
    assert_eq!(
        <[i64; 8]>::from(simd.simd_le_u64x8(a, b)),
        [0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64]
    );
}
