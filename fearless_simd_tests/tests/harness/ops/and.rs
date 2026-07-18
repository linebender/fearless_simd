// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn and_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
        ],
    );
    assert_eq!(
        *(a & b),
        [85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0]
    );
}

#[simd_test]
fn and_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(simd, &[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
    let b = u8x16::from_slice(
        simd,
        &[
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
        ],
    );
    assert_eq!(*(a & b), [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
}

#[simd_test]
fn and_mask8x16<S: Simd>(simd: S) {
    let a = mask8x16::from_slice(
        simd,
        &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    let b = mask8x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert_eq!(
        <[i8; 16]>::from(a & b),
        [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]
    );
}

#[simd_test]
fn and_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
        ],
    );
    assert_eq!(
        *(a & b),
        [
            85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85,
            0, 85, 0, 85, 0, 85, 0
        ]
    );
}

#[simd_test]
fn and_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
        ],
    );
    assert_eq!(
        *(a & b),
        [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0
        ]
    );
}

#[simd_test]
fn and_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
        ],
    );
    assert_eq!(
        *(a & b),
        [
            85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85,
            0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0,
            85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0
        ]
    );
}

#[simd_test]
fn and_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0,
        ],
    );
    let b = u8x64::from_slice(
        simd,
        &[
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
        ],
    );
    assert_eq!(
        *(a & b),
        [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0
        ]
    );
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn and_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[3_i64, 3_i64]);
    assert_eq!(*simd.and_i64x2(a, b), [1_i64, 2_i64]);
}

#[simd_test]
fn and_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[3_i64, 3_i64, 3_i64, 3_i64]);
    assert_eq!(*simd.and_i64x4(a, b), [1_i64, 2_i64, 3_i64, 0_i64]);
}

#[simd_test]
fn and_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64],
    );
    assert_eq!(
        *simd.and_i64x8(a, b),
        [1_i64, 2_i64, 3_i64, 0_i64, 1_i64, 2_i64, 3_i64, 0_i64]
    );
}

#[simd_test]
fn and_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[3_u64, 3_u64]);
    assert_eq!(*simd.and_u64x2(a, b), [1_u64, 2_u64]);
}

#[simd_test]
fn and_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[3_u64, 3_u64, 3_u64, 3_u64]);
    assert_eq!(*simd.and_u64x4(a, b), [1_u64, 2_u64, 3_u64, 0_u64]);
}

#[simd_test]
fn and_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[3_u64, 3_u64, 3_u64, 3_u64, 3_u64, 3_u64, 3_u64, 3_u64],
    );
    assert_eq!(
        *simd.and_u64x8(a, b),
        [1_u64, 2_u64, 3_u64, 0_u64, 1_u64, 2_u64, 3_u64, 0_u64]
    );
}

#[simd_test]
fn and_mask64x2<S: Simd>(simd: S) {
    let a = mask64x2::from_slice(simd, &[-1_i64, 0_i64]);
    let b = mask64x2::from_slice(simd, &[0_i64, -1_i64]);
    assert_eq!(<[i64; 2]>::from(simd.and_mask64x2(a, b)), [0_i64, 0_i64]);
}

#[simd_test]
fn and_mask64x4<S: Simd>(simd: S) {
    let a = mask64x4::from_slice(simd, &[-1_i64, 0_i64, -1_i64, 0_i64]);
    let b = mask64x4::from_slice(simd, &[0_i64, -1_i64, 0_i64, -1_i64]);
    assert_eq!(
        <[i64; 4]>::from(simd.and_mask64x4(a, b)),
        [0_i64, 0_i64, 0_i64, 0_i64]
    );
}

#[simd_test]
fn and_mask64x8<S: Simd>(simd: S) {
    let a = mask64x8::from_slice(
        simd,
        &[-1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64],
    );
    let b = mask64x8::from_slice(
        simd,
        &[0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64],
    );
    assert_eq!(
        <[i64; 8]>::from(simd.and_mask64x8(a, b)),
        [0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64]
    );
}
