// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

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

// Generated gap-fill coverage rows.

#[simd_test]
fn block_splat_i8x16<S: Simd>(simd: S) {
    let block_values: [i8; 16] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let block = i8x16::from_slice(simd, &block_values);
    let result = i8x16::block_splat(block);
    let expected: [i8; 16] = core::array::from_fn(|i| block_values[i % 16]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_u8x16<S: Simd>(simd: S) {
    let block_values: [u8; 16] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let block = u8x16::from_slice(simd, &block_values);
    let result = u8x16::block_splat(block);
    let expected: [u8; 16] = core::array::from_fn(|i| block_values[i % 16]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_i8x32<S: Simd>(simd: S) {
    let block_values: [i8; 16] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let block = i8x16::from_slice(simd, &block_values);
    let result = i8x32::block_splat(block);
    let expected: [i8; 32] = core::array::from_fn(|i| block_values[i % 16]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_u8x32<S: Simd>(simd: S) {
    let block_values: [u8; 16] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let block = u8x16::from_slice(simd, &block_values);
    let result = u8x32::block_splat(block);
    let expected: [u8; 32] = core::array::from_fn(|i| block_values[i % 16]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_i16x8<S: Simd>(simd: S) {
    let block_values: [i16; 8] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let block = i16x8::from_slice(simd, &block_values);
    let result = i16x8::block_splat(block);
    let expected: [i16; 8] = core::array::from_fn(|i| block_values[i % 8]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_u16x8<S: Simd>(simd: S) {
    let block_values: [u16; 8] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let block = u16x8::from_slice(simd, &block_values);
    let result = u16x8::block_splat(block);
    let expected: [u16; 8] = core::array::from_fn(|i| block_values[i % 8]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_i16x16<S: Simd>(simd: S) {
    let block_values: [i16; 8] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let block = i16x8::from_slice(simd, &block_values);
    let result = i16x16::block_splat(block);
    let expected: [i16; 16] = core::array::from_fn(|i| block_values[i % 8]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_u16x16<S: Simd>(simd: S) {
    let block_values: [u16; 8] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let block = u16x8::from_slice(simd, &block_values);
    let result = u16x16::block_splat(block);
    let expected: [u16; 16] = core::array::from_fn(|i| block_values[i % 8]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_f32x4<S: Simd>(simd: S) {
    let block_values: [f32; 4] = core::array::from_fn(|i| i as f32 + 1.25_f32);
    let block = f32x4::from_slice(simd, &block_values);
    let result = f32x4::block_splat(block);
    let expected: [f32; 4] = core::array::from_fn(|i| block_values[i % 4]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_i32x4<S: Simd>(simd: S) {
    let block_values: [i32; 4] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let block = i32x4::from_slice(simd, &block_values);
    let result = i32x4::block_splat(block);
    let expected: [i32; 4] = core::array::from_fn(|i| block_values[i % 4]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_u32x4<S: Simd>(simd: S) {
    let block_values: [u32; 4] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let block = u32x4::from_slice(simd, &block_values);
    let result = u32x4::block_splat(block);
    let expected: [u32; 4] = core::array::from_fn(|i| block_values[i % 4]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_f32x8<S: Simd>(simd: S) {
    let block_values: [f32; 4] = core::array::from_fn(|i| i as f32 + 1.25_f32);
    let block = f32x4::from_slice(simd, &block_values);
    let result = f32x8::block_splat(block);
    let expected: [f32; 8] = core::array::from_fn(|i| block_values[i % 4]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_i32x8<S: Simd>(simd: S) {
    let block_values: [i32; 4] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let block = i32x4::from_slice(simd, &block_values);
    let result = i32x8::block_splat(block);
    let expected: [i32; 8] = core::array::from_fn(|i| block_values[i % 4]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_u32x8<S: Simd>(simd: S) {
    let block_values: [u32; 4] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let block = u32x4::from_slice(simd, &block_values);
    let result = u32x8::block_splat(block);
    let expected: [u32; 8] = core::array::from_fn(|i| block_values[i % 4]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_f64x2<S: Simd>(simd: S) {
    let block_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let block = f64x2::from_slice(simd, &block_values);
    let result = f64x2::block_splat(block);
    let expected: [f64; 2] = core::array::from_fn(|i| block_values[i % 2]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_i64x2<S: Simd>(simd: S) {
    let block_values: [i64; 2] = core::array::from_fn(|i| (i % 23) as i64 + 10_i64);
    let block = i64x2::from_slice(simd, &block_values);
    let result = i64x2::block_splat(block);
    let expected: [i64; 2] = core::array::from_fn(|i| block_values[i % 2]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_u64x2<S: Simd>(simd: S) {
    let block_values: [u64; 2] = core::array::from_fn(|i| (i % 23) as u64 + 10_u64);
    let block = u64x2::from_slice(simd, &block_values);
    let result = u64x2::block_splat(block);
    let expected: [u64; 2] = core::array::from_fn(|i| block_values[i % 2]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_f64x4<S: Simd>(simd: S) {
    let block_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let block = f64x2::from_slice(simd, &block_values);
    let result = f64x4::block_splat(block);
    let expected: [f64; 4] = core::array::from_fn(|i| block_values[i % 2]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_i64x4<S: Simd>(simd: S) {
    let block_values: [i64; 2] = core::array::from_fn(|i| (i % 23) as i64 + 10_i64);
    let block = i64x2::from_slice(simd, &block_values);
    let result = i64x4::block_splat(block);
    let expected: [i64; 4] = core::array::from_fn(|i| block_values[i % 2]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_u64x4<S: Simd>(simd: S) {
    let block_values: [u64; 2] = core::array::from_fn(|i| (i % 23) as u64 + 10_u64);
    let block = u64x2::from_slice(simd, &block_values);
    let result = u64x4::block_splat(block);
    let expected: [u64; 4] = core::array::from_fn(|i| block_values[i % 2]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_f64x8<S: Simd>(simd: S) {
    let block_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let block = f64x2::from_slice(simd, &block_values);
    let result = f64x8::block_splat(block);
    let expected: [f64; 8] = core::array::from_fn(|i| block_values[i % 2]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_i64x8<S: Simd>(simd: S) {
    let block_values: [i64; 2] = core::array::from_fn(|i| (i % 23) as i64 + 10_i64);
    let block = i64x2::from_slice(simd, &block_values);
    let result = i64x8::block_splat(block);
    let expected: [i64; 8] = core::array::from_fn(|i| block_values[i % 2]);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn block_splat_u64x8<S: Simd>(simd: S) {
    let block_values: [u64; 2] = core::array::from_fn(|i| (i % 23) as u64 + 10_u64);
    let block = u64x2::from_slice(simd, &block_values);
    let result = u64x8::block_splat(block);
    let expected: [u64; 8] = core::array::from_fn(|i| block_values[i % 2]);
    assert_eq!(result.as_slice(), expected.as_slice());
}
