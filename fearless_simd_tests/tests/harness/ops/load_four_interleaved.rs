// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn load_four_interleaved_f32x4<S: Simd>(simd: S) {
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
        [0.0, f32::NAN, f32::INFINITY, -3.0],
        [4.0, -0.0, 6.0, f32::NEG_INFINITY],
        [8.0, 9.0, -10.0, 11.0],
        [f32::MIN, 13.0, f32::MAX, 15.0],
    ];

    // Note: f32::NAN != f32::NAN hence we compare the bit pattern.
    let result = simd.load_four_interleaved_f32x4(&data);
    assert_eq!(
        result.map(|vector| (*vector).map(f32::to_bits)),
        expected.map(|vector| vector.map(f32::to_bits)),
    );
}

#[simd_test]
fn load_four_interleaved_f64x2<S: Simd>(simd: S) {
    let data = [
        0.0,
        f64::NAN,
        f64::INFINITY,
        -3.0,
        -0.0,
        f64::MIN,
        f64::NEG_INFINITY,
        f64::MAX,
    ];
    let expected = [
        [0.0, -0.0],
        [f64::NAN, f64::MIN],
        [f64::INFINITY, f64::NEG_INFINITY],
        [-3.0, f64::MAX],
    ];

    // Note: f64::NAN != f64::NAN hence we compare the bit pattern.
    let result = simd.load_four_interleaved_f64x2(&data);
    assert_eq!(
        result.map(|vector| (*vector).map(f64::to_bits)),
        expected.map(|vector| vector.map(f64::to_bits)),
    );
}

#[simd_test]
fn load_four_interleaved_i8x16<S: Simd>(simd: S) {
    #[rustfmt::skip]
    let data: [i8; 64] = [
        i8::MIN, -31, -30, -29, -28, -27, -26, -25,
        -24, -23, -22, -21, -20, -19, -18, -17,
        -16, -15, -14, -13, -12, -11, -10, -9,
        -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, i8::MAX,
    ];
    assert_eq!(
        simd.load_four_interleaved_i8x16(&data)
            .map(|vector| *vector),
        [
            [
                i8::MIN,
                -28,
                -24,
                -20,
                -16,
                -12,
                -8,
                -4,
                0,
                4,
                8,
                12,
                16,
                20,
                24,
                28
            ],
            [
                -31, -27, -23, -19, -15, -11, -7, -3, 1, 5, 9, 13, 17, 21, 25, 29
            ],
            [
                -30, -26, -22, -18, -14, -10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30
            ],
            [
                -29,
                -25,
                -21,
                -17,
                -13,
                -9,
                -5,
                -1,
                3,
                7,
                11,
                15,
                19,
                23,
                27,
                i8::MAX
            ],
        ]
    );
}

#[simd_test]
fn load_four_interleaved_u8x16<S: Simd>(simd: S) {
    #[rustfmt::skip]
    let data: [u8; 64] = [
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, u8::MAX,
    ];
    assert_eq!(
        simd.load_four_interleaved_u8x16(&data)
            .map(|vector| *vector),
        [
            [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60],
            [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61],
            [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62],
            [
                3,
                7,
                11,
                15,
                19,
                23,
                27,
                31,
                35,
                39,
                43,
                47,
                51,
                55,
                59,
                u8::MAX
            ],
        ]
    );
}

#[simd_test]
fn load_four_interleaved_i16x8<S: Simd>(simd: S) {
    let data: [i16; 32] = [
        i16::MIN,
        -15,
        -14,
        -13,
        -12,
        -11,
        -10,
        -9,
        -8,
        -7,
        -6,
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        i16::MAX,
    ];
    assert_eq!(
        simd.load_four_interleaved_i16x8(&data)
            .map(|vector| *vector),
        [
            [i16::MIN, -12, -8, -4, 0, 4, 8, 12],
            [-15, -11, -7, -3, 1, 5, 9, 13],
            [-14, -10, -6, -2, 2, 6, 10, 14],
            [-13, -9, -5, -1, 3, 7, 11, i16::MAX],
        ]
    );
}

#[simd_test]
fn load_four_interleaved_u16x8<S: Simd>(simd: S) {
    #[rustfmt::skip]
    let data: [u16; 32] = [
        u16::MIN, 2, 3, 4, 5, 6, 7, 8,
        10, 20, 30, 40, 50, 60, 70, 80,
        100, 200, 300, 400, 500, 600, 700, 800,
        1000, 2000, 3000, 4000, 5000, 6000, 7000, u16::MAX,
    ];
    assert_eq!(
        simd.load_four_interleaved_u16x8(&data)
            .map(|vector| *vector),
        [
            [u16::MIN, 5, 10, 50, 100, 500, 1000, 5000],
            [2, 6, 20, 60, 200, 600, 2000, 6000],
            [3, 7, 30, 70, 300, 700, 3000, 7000],
            [4, 8, 40, 80, 400, 800, 4000, u16::MAX],
        ]
    );
}

#[simd_test]
fn load_four_interleaved_i32x4<S: Simd>(simd: S) {
    let data: [i32; 16] = [
        i32::MIN,
        -7,
        -6,
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        i32::MAX,
    ];
    assert_eq!(
        simd.load_four_interleaved_i32x4(&data)
            .map(|vector| *vector),
        [
            [i32::MIN, -4, 0, 4],
            [-7, -3, 1, 5],
            [-6, -2, 2, 6],
            [-5, -1, 3, i32::MAX]
        ]
    );
}

#[simd_test]
fn load_four_interleaved_u32x4<S: Simd>(simd: S) {
    #[rustfmt::skip]
    let data: [u32; 16] = [
        1, 2, 3, 4,
        10, 20, 30, 40,
        100, 200, 300, 400,
        1000, 2000, 3000, u32::MAX,
    ];
    assert_eq!(
        simd.load_four_interleaved_u32x4(&data)
            .map(|vector| *vector),
        [
            [1, 10, 100, 1000],
            [2, 20, 200, 2000],
            [3, 30, 300, 3000],
            [4, 40, 400, u32::MAX],
        ]
    );
}

#[simd_test]
fn load_four_interleaved_i64x2<S: Simd>(simd: S) {
    let data = [
        i64::MIN,
        -3_i64,
        -2_i64,
        -1_i64,
        0_i64,
        1_i64,
        2_i64,
        i64::MAX,
    ];
    assert_eq!(
        simd.load_four_interleaved_i64x2(&data)
            .map(|vector| *vector),
        [[i64::MIN, 0], [-3, 1], [-2, 2], [-1, i64::MAX]]
    );
}

#[simd_test]
fn load_four_interleaved_u64x2<S: Simd>(simd: S) {
    let data = [u64::MIN, 3_u64, 5_u64, 7_u64, 2_u64, 4_u64, 6_u64, u64::MAX];
    assert_eq!(
        simd.load_four_interleaved_u64x2(&data)
            .map(|vector| *vector),
        [[u64::MIN, 2], [3, 4], [5, 6], [7, u64::MAX]]
    );
}
