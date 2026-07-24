// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn store_interleaved_128_f32x16<S: Simd>(simd: S) {
    let input = [
        0.0,
        f32::NAN,
        f32::INFINITY,
        -3.0,
        4.0,
        -0.0,
        6.0,
        f32::NEG_INFINITY,
        8.0,
        9.0,
        -10.0,
        11.0,
        f32::MIN,
        13.0,
        f32::MAX,
        15.0,
    ];
    let a = f32x16::from_slice(simd, &input);
    let mut dest = [0.0_f32; 16];
    simd.store_interleaved_128_f32x16(a, &mut dest);

    let expected = [
        0.0,
        4.0,
        8.0,
        f32::MIN,
        f32::NAN,
        -0.0,
        9.0,
        13.0,
        f32::INFINITY,
        6.0,
        -10.0,
        f32::MAX,
        -3.0,
        f32::NEG_INFINITY,
        11.0,
        15.0,
    ];

    // Note: f32::NAN != f32::NAN hence we compare the bit pattern.
    assert_eq!(dest.map(f32::to_bits), expected.map(f32::to_bits));
}

#[simd_test]
fn store_interleaved_128_u8x64<S: Simd>(simd: S) {
    let input: [u8; 64] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ];
    let a = u8x64::from_slice(simd, &input);
    let mut dest = [0_u8; 64];
    simd.store_interleaved_128_u8x64(a, &mut dest);

    let expected = [
        0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53,
        6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63,
    ];

    assert_eq!(dest, expected);
}

#[simd_test]
fn store_interleaved_128_u16x32<S: Simd>(simd: S) {
    let input: [u16; 32] = [
        0, 1, 2, 3, 4, 5, 6, 7, 100, 101, 102, 103, 104, 105, 106, 107, 200, 201, 202, 203, 204,
        205, 206, 207, 300, 301, 302, 303, 304, 305, 306, 307,
    ];
    let a = u16x32::from_slice(simd, &input);
    let mut dest = [0_u16; 32];
    simd.store_interleaved_128_u16x32(a, &mut dest);

    let expected = [
        0, 100, 200, 300, 1, 101, 201, 301, 2, 102, 202, 302, 3, 103, 203, 303, 4, 104, 204, 304,
        5, 105, 205, 305, 6, 106, 206, 306, 7, 107, 207, 307,
    ];

    assert_eq!(dest, expected);
}

#[simd_test]
fn store_interleaved_128_u32x16<S: Simd>(simd: S) {
    let input: [u32; 16] = [
        0,
        1,
        u32::MAX,
        3,
        1000,
        1001,
        1002,
        1003,
        2000,
        2001,
        2002,
        2003,
        u32::MIN,
        3001,
        3002,
        u32::MAX - 1,
    ];
    let a = u32x16::from_slice(simd, &input);
    let mut dest = [0_u32; 16];
    simd.store_interleaved_128_u32x16(a, &mut dest);

    let expected = [
        0,
        1000,
        2000,
        u32::MIN,
        1,
        1001,
        2001,
        3001,
        u32::MAX,
        1002,
        2002,
        3002,
        3,
        1003,
        2003,
        u32::MAX - 1,
    ];

    assert_eq!(dest, expected);
}

// Additional concrete rows for this operation.

#[simd_test]
fn store_interleaved_128_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let mut out = [0_u64; 8];
    simd.store_interleaved_128_u64x8(a, &mut out);
    assert_eq!(
        out,
        [1_u64, 3_u64, 5_u64, 7_u64, 2_u64, 4_u64, 6_u64, 8_u64]
    );
}
