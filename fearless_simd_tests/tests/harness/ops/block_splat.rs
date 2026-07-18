// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn block_splat_f32x16<S: Simd>(simd: S) {
    let block = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let a = f32x16::block_splat(block);
    assert_eq!(
        *a,
        [
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0
        ]
    );
}

#[simd_test]
fn block_splat_i8x64<S: Simd>(simd: S) {
    let block = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let a = i8x64::block_splat(block);
    assert_eq!(
        *a,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ]
    );
}

#[simd_test]
fn block_splat_u8x64<S: Simd>(simd: S) {
    let block = u8x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    let a = u8x64::block_splat(block);
    assert_eq!(
        *a,
        [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40,
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40, 50, 60, 70, 80,
            90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
            120, 130, 140, 150, 160
        ]
    );
}

#[simd_test]
fn block_splat_i16x32<S: Simd>(simd: S) {
    let block = i16x8::from_slice(simd, &[100, 200, 300, 400, 500, 600, 700, 800]);
    let a = i16x32::block_splat(block);
    assert_eq!(
        *a,
        [
            100, 200, 300, 400, 500, 600, 700, 800, 100, 200, 300, 400, 500, 600, 700, 800, 100,
            200, 300, 400, 500, 600, 700, 800, 100, 200, 300, 400, 500, 600, 700, 800
        ]
    );
}

#[simd_test]
fn block_splat_u16x32<S: Simd>(simd: S) {
    let block = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let a = u16x32::block_splat(block);
    assert_eq!(
        *a,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8
        ]
    );
}

#[simd_test]
fn block_splat_i32x16<S: Simd>(simd: S) {
    let block = i32x4::from_slice(simd, &[11, 22, 33, 44]);
    let a = i32x16::block_splat(block);
    assert_eq!(
        *a,
        [
            11, 22, 33, 44, 11, 22, 33, 44, 11, 22, 33, 44, 11, 22, 33, 44
        ]
    );
}

#[simd_test]
fn block_splat_u32x16<S: Simd>(simd: S) {
    let block = u32x4::from_slice(simd, &[0xDEAD, 0xBEEF, 0xCAFE, 0xBABE]);
    let a = u32x16::block_splat(block);
    assert_eq!(
        *a,
        [
            0xDEAD, 0xBEEF, 0xCAFE, 0xBABE, 0xDEAD, 0xBEEF, 0xCAFE, 0xBABE, 0xDEAD, 0xBEEF, 0xCAFE,
            0xBABE, 0xDEAD, 0xBEEF, 0xCAFE, 0xBABE
        ]
    );
}
