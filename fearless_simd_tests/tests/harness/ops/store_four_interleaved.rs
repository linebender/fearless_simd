// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn store_four_interleaved_f32x4<S: Simd>(simd: S) {
    let vectors = [
        f32x4::from_slice(simd, &[0.0, f32::NAN, f32::INFINITY, -3.0]),
        f32x4::from_slice(simd, &[4.0, -0.0, 6.0, f32::NEG_INFINITY]),
        f32x4::from_slice(simd, &[8.0, 9.0, -10.0, 11.0]),
        f32x4::from_slice(simd, &[f32::MIN, 13.0, f32::MAX, 15.0]),
    ];
    let mut dest = [0.0_f32; 16];
    f32x4::store_four_interleaved(vectors, &mut dest);

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
fn store_four_interleaved_f64x2<S: Simd>(simd: S) {
    let vectors = [
        f64x2::from_slice(simd, &[0.0, -0.0]),
        f64x2::from_slice(simd, &[f64::NAN, f64::MIN]),
        f64x2::from_slice(simd, &[f64::INFINITY, f64::NEG_INFINITY]),
        f64x2::from_slice(simd, &[-3.0, f64::MAX]),
    ];
    let mut dest = [0.0_f64; 8];
    f64x2::store_four_interleaved(vectors, &mut dest);

    let expected = [
        0.0,
        f64::NAN,
        f64::INFINITY,
        -3.0,
        -0.0,
        f64::MIN,
        f64::NEG_INFINITY,
        f64::MAX,
    ];

    // Note: f64::NAN != f64::NAN hence we compare the bit pattern.
    assert_eq!(dest.map(f64::to_bits), expected.map(f64::to_bits));
}

#[simd_test]
fn store_four_interleaved_i8x16<S: Simd>(simd: S) {
    let vectors = [
        i8x16::from_slice(
            simd,
            &[
                i8::MIN,
                -31,
                -30,
                -29,
                -28,
                -27,
                -26,
                -25,
                -24,
                -23,
                -22,
                -21,
                -20,
                -19,
                -18,
                -17,
            ],
        ),
        i8x16::from_slice(
            simd,
            &[
                -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
            ],
        ),
        i8x16::from_slice(
            simd,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ),
        i8x16::from_slice(
            simd,
            &[
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                i8::MAX,
            ],
        ),
    ];
    let mut dest = [0_i8; 64];
    i8x16::store_four_interleaved(vectors, &mut dest);

    let expected = [
        i8::MIN,
        -16,
        0,
        16,
        -31,
        -15,
        1,
        17,
        -30,
        -14,
        2,
        18,
        -29,
        -13,
        3,
        19,
        -28,
        -12,
        4,
        20,
        -27,
        -11,
        5,
        21,
        -26,
        -10,
        6,
        22,
        -25,
        -9,
        7,
        23,
        -24,
        -8,
        8,
        24,
        -23,
        -7,
        9,
        25,
        -22,
        -6,
        10,
        26,
        -21,
        -5,
        11,
        27,
        -20,
        -4,
        12,
        28,
        -19,
        -3,
        13,
        29,
        -18,
        -2,
        14,
        30,
        -17,
        -1,
        15,
        i8::MAX,
    ];

    assert_eq!(dest, expected);
}

#[simd_test]
fn store_four_interleaved_u8x16<S: Simd>(simd: S) {
    let vectors = [
        u8x16::from_slice(
            simd,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ),
        u8x16::from_slice(
            simd,
            &[
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ],
        ),
        u8x16::from_slice(
            simd,
            &[
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            ],
        ),
        u8x16::from_slice(
            simd,
            &[
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                u8::MAX,
            ],
        ),
    ];
    let mut dest = [0_u8; 64];
    u8x16::store_four_interleaved(vectors, &mut dest);

    let expected = [
        0,
        16,
        32,
        48,
        1,
        17,
        33,
        49,
        2,
        18,
        34,
        50,
        3,
        19,
        35,
        51,
        4,
        20,
        36,
        52,
        5,
        21,
        37,
        53,
        6,
        22,
        38,
        54,
        7,
        23,
        39,
        55,
        8,
        24,
        40,
        56,
        9,
        25,
        41,
        57,
        10,
        26,
        42,
        58,
        11,
        27,
        43,
        59,
        12,
        28,
        44,
        60,
        13,
        29,
        45,
        61,
        14,
        30,
        46,
        62,
        15,
        31,
        47,
        u8::MAX,
    ];

    assert_eq!(dest, expected);
}

#[simd_test]
fn store_four_interleaved_i16x8<S: Simd>(simd: S) {
    let vectors = [
        i16x8::from_slice(simd, &[i16::MIN, -15, -14, -13, -12, -11, -10, -9]),
        i16x8::from_slice(simd, &[-8, -7, -6, -5, -4, -3, -2, -1]),
        i16x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]),
        i16x8::from_slice(simd, &[8, 9, 10, 11, 12, 13, 14, i16::MAX]),
    ];
    let mut dest = [0_i16; 32];
    i16x8::store_four_interleaved(vectors, &mut dest);

    let expected = [
        i16::MIN,
        -8,
        0,
        8,
        -15,
        -7,
        1,
        9,
        -14,
        -6,
        2,
        10,
        -13,
        -5,
        3,
        11,
        -12,
        -4,
        4,
        12,
        -11,
        -3,
        5,
        13,
        -10,
        -2,
        6,
        14,
        -9,
        -1,
        7,
        i16::MAX,
    ];

    assert_eq!(dest, expected);
}

#[simd_test]
fn store_four_interleaved_u16x8<S: Simd>(simd: S) {
    let vectors = [
        u16x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]),
        u16x8::from_slice(simd, &[100, 101, 102, 103, 104, 105, 106, 107]),
        u16x8::from_slice(simd, &[200, 201, 202, 203, 204, 205, 206, 207]),
        u16x8::from_slice(simd, &[300, 301, 302, 303, 304, 305, 306, u16::MAX]),
    ];
    let mut dest = [0_u16; 32];
    u16x8::store_four_interleaved(vectors, &mut dest);

    let expected = [
        0,
        100,
        200,
        300,
        1,
        101,
        201,
        301,
        2,
        102,
        202,
        302,
        3,
        103,
        203,
        303,
        4,
        104,
        204,
        304,
        5,
        105,
        205,
        305,
        6,
        106,
        206,
        306,
        7,
        107,
        207,
        u16::MAX,
    ];

    assert_eq!(dest, expected);
}

#[simd_test]
fn store_four_interleaved_i32x4<S: Simd>(simd: S) {
    let vectors = [
        i32x4::from_slice(simd, &[i32::MIN, -7, -6, -5]),
        i32x4::from_slice(simd, &[-4, -3, -2, -1]),
        i32x4::from_slice(simd, &[0, 1, 2, 3]),
        i32x4::from_slice(simd, &[4, 5, 6, i32::MAX]),
    ];
    let mut dest = [0_i32; 16];
    i32x4::store_four_interleaved(vectors, &mut dest);
    assert_eq!(
        dest,
        [
            i32::MIN,
            -4,
            0,
            4,
            -7,
            -3,
            1,
            5,
            -6,
            -2,
            2,
            6,
            -5,
            -1,
            3,
            i32::MAX
        ]
    );
}

#[simd_test]
fn store_four_interleaved_u32x4<S: Simd>(simd: S) {
    let vectors = [
        u32x4::from_slice(simd, &[0, 1, u32::MAX, 3]),
        u32x4::from_slice(simd, &[1000, 1001, 1002, 1003]),
        u32x4::from_slice(simd, &[2000, 2001, 2002, 2003]),
        u32x4::from_slice(simd, &[u32::MIN, 3001, 3002, u32::MAX - 1]),
    ];
    let mut dest = [0_u32; 16];
    u32x4::store_four_interleaved(vectors, &mut dest);
    assert_eq!(
        dest,
        [
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
        ]
    );
}

#[simd_test]
fn store_four_interleaved_i64x2<S: Simd>(simd: S) {
    let vectors = [
        i64x2::from_slice(simd, &[i64::MIN, -3]),
        i64x2::from_slice(simd, &[-2, -1]),
        i64x2::from_slice(simd, &[0, 1]),
        i64x2::from_slice(simd, &[2, i64::MAX]),
    ];
    let mut dest = [0_i64; 8];
    i64x2::store_four_interleaved(vectors, &mut dest);
    assert_eq!(dest, [i64::MIN, -2, 0, 2, -3, -1, 1, i64::MAX]);
}

#[simd_test]
fn store_four_interleaved_u64x2<S: Simd>(simd: S) {
    let vectors = [
        u64x2::from_slice(simd, &[u64::MIN, 2]),
        u64x2::from_slice(simd, &[3, 4]),
        u64x2::from_slice(simd, &[5, 6]),
        u64x2::from_slice(simd, &[7, u64::MAX]),
    ];
    let mut dest = [0_u64; 8];
    u64x2::store_four_interleaved(vectors, &mut dest);
    assert_eq!(dest, [u64::MIN, 3, 5, 7, 2, 4, 6, u64::MAX]);
}
