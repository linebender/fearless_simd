// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn not_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8],
    );
    assert_eq!(
        *(!a),
        [-1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[simd_test]
fn not_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        *(!a),
        [
            255, 254, 253, 252, 251, 250, 249, 248, 254, 253, 252, 251, 250, 249, 248, 247
        ]
    );
}

#[simd_test]
fn not_mask8x16<S: Simd>(simd: S) {
    let a = mask8x16::from_slice(
        simd,
        &[0, -1, -1, 0, -1, 0, 0, -1, -1, 0, -1, 0, 0, -1, 0, -1],
    );
    assert_eq!(
        <[i8; 16]>::from(!a),
        [-1, 0, 0, -1, 0, -1, -1, 0, 0, -1, 0, -1, -1, 0, -1, 0]
    );
}

#[simd_test]
fn not_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7, -1, -2,
            -3, -4, -5, -6, -7, -8,
        ],
    );
    assert_eq!(
        *(!a),
        [
            -1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8,
            0, 1, 2, 3, 4, 5, 6, 7
        ]
    );
}

#[simd_test]
fn not_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5,
            6, 7, 8,
        ],
    );
    assert_eq!(
        *(!a),
        [
            255, 254, 253, 252, 251, 250, 249, 248, 254, 253, 252, 251, 250, 249, 248, 247, 255,
            254, 253, 252, 251, 250, 249, 248, 254, 253, 252, 251, 250, 249, 248, 247
        ]
    );
}

#[simd_test]
fn not_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7, -1, -2,
            -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8, 0, 1,
            2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8,
        ],
    );
    assert_eq!(
        *(!a),
        [
            -1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8,
            0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7, -1, -2,
            -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7
        ]
    );
}

#[simd_test]
fn not_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5,
            6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2,
            3, 4, 5, 6, 7, 8,
        ],
    );
    assert_eq!(
        *(!a),
        [
            255, 254, 253, 252, 251, 250, 249, 248, 254, 253, 252, 251, 250, 249, 248, 247, 255,
            254, 253, 252, 251, 250, 249, 248, 254, 253, 252, 251, 250, 249, 248, 247, 255, 254,
            253, 252, 251, 250, 249, 248, 254, 253, 252, 251, 250, 249, 248, 247, 255, 254, 253,
            252, 251, 250, 249, 248, 254, 253, 252, 251, 250, 249, 248, 247
        ]
    );
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn not_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    assert_eq!(*simd.not_i64x2(a), [-2_i64, 1_i64]);
}

#[simd_test]
fn not_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    assert_eq!(*simd.not_i64x4(a), [-2_i64, 1_i64, -4_i64, 3_i64]);
}

#[simd_test]
fn not_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    assert_eq!(
        *simd.not_i64x8(a),
        [-2_i64, 1_i64, -4_i64, 3_i64, -6_i64, 5_i64, -8_i64, 7_i64]
    );
}

#[simd_test]
fn not_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    assert_eq!(
        *simd.not_u64x2(a),
        [18446744073709551614_u64, 18446744073709551613_u64]
    );
}

#[simd_test]
fn not_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    assert_eq!(
        *simd.not_u64x4(a),
        [
            18446744073709551614_u64,
            18446744073709551613_u64,
            18446744073709551612_u64,
            18446744073709551611_u64
        ]
    );
}

#[simd_test]
fn not_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    assert_eq!(
        *simd.not_u64x8(a),
        [
            18446744073709551614_u64,
            18446744073709551613_u64,
            18446744073709551612_u64,
            18446744073709551611_u64,
            18446744073709551610_u64,
            18446744073709551609_u64,
            18446744073709551608_u64,
            18446744073709551607_u64
        ]
    );
}

#[simd_test]
fn not_mask64x2<S: Simd>(simd: S) {
    let a = mask64x2::from_slice(simd, &[-1_i64, 0_i64]);
    assert_eq!(<[i64; 2]>::from(simd.not_mask64x2(a)), [0_i64, -1_i64]);
}

#[simd_test]
fn not_mask64x4<S: Simd>(simd: S) {
    let a = mask64x4::from_slice(simd, &[-1_i64, 0_i64, -1_i64, 0_i64]);
    assert_eq!(
        <[i64; 4]>::from(simd.not_mask64x4(a)),
        [0_i64, -1_i64, 0_i64, -1_i64]
    );
}

#[simd_test]
fn not_mask64x8<S: Simd>(simd: S) {
    let a = mask64x8::from_slice(
        simd,
        &[-1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64],
    );
    assert_eq!(
        <[i64; 8]>::from(simd.not_mask64x8(a)),
        [0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64]
    );
}
