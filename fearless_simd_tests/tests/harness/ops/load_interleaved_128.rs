// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn load_interleaved_128_f32x16<S: Simd>(simd: S) {
    let data = [
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
    let expected = [
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

    // Note: f32::NAN != f32::NAN hence we compare the bit pattern.
    let result = simd.load_interleaved_128_f32x16(&data);
    assert_eq!((*result).map(f32::to_bits), expected.map(f32::to_bits),);
}

#[simd_test]
fn load_interleaved_128_u32x16<S: Simd>(simd: S) {
    #[rustfmt::skip]
    let data: [u32; 16] = [
        1, 2, 3, 4,
        10, 20, 30, 40,
        100, 200, 300, 400,
        1000, 2000, 3000, 4000,
    ];
    assert_eq!(
        *simd.load_interleaved_128_u32x16(&data),
        [
            1, 10, 100, 1000, 2, 20, 200, 2000, 3, 30, 300, 3000, 4, 40, 400, 4000
        ]
    );
}

#[simd_test]
fn load_interleaved_128_u16x32<S: Simd>(simd: S) {
    #[rustfmt::skip]
    let data: [u16; 32] = [
        1, 2, 3, 4,
        5, 6, 7, 8,

        10, 20, 30, 40,
        50, 60, 70, 80,

        100, 200, 300, 400,
        500, 600, 700, 800,

        1000, 2000, 3000, 4000,
        5000, 6000, 7000, 8000,
    ];
    assert_eq!(
        *simd.load_interleaved_128_u16x32(&data),
        [
            1, 5, 10, 50, 100, 500, 1000, 5000, 2, 6, 20, 60, 200, 600, 2000, 6000, 3, 7, 30, 70,
            300, 700, 3000, 7000, 4, 8, 40, 80, 400, 800, 4000, 8000
        ]
    );
}

#[simd_test]
fn load_interleaved_128_u8x64<S: Simd>(simd: S) {
    #[rustfmt::skip]
    let data: [u8; 64] = [
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,

        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31,

        32, 33, 34, 35,
        36, 37, 38, 39,
        40, 41, 42, 43,
        44, 45, 46, 47,

        48, 49, 50, 51,
        52, 53, 54, 55,
        56, 57, 58, 59,
        60, 61, 62, 63,
    ];
    assert_eq!(
        *simd.load_interleaved_128_u8x64(&data),
        [
            0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25,
            29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50,
            54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
        ]
    );
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn load_interleaved_128_u64x8<S: Simd>(simd: S) {
    let data = [1_u64, 3_u64, 5_u64, 7_u64, 2_u64, 4_u64, 6_u64, 8_u64];
    let a = simd.load_interleaved_128_u64x8(&data);
    assert_eq!(*a, [1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64]);
}
