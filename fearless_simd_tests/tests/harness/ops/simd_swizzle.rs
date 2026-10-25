// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn simd_swizzle_u8x16<S: Simd>(simd: S) {
    let value = u8x16::simd_from(
        simd,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let result = simd_swizzle!(
        value,
        [15, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14]
    );
    assert_eq!(
        *result,
        [16, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 16, 5, 10, 15]
    );
}

#[simd_test]
fn simd_swizzle_i8x16<S: Simd>(simd: S) {
    let value = i8x16::simd_from(
        simd,
        [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let result = simd_swizzle!(
        value,
        [15, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14]
    );
    assert_eq!(
        *result,
        [
            -16, 9, -14, 3, -8, 13, -2, 7, -12, 1, -6, 11, -16, 5, -10, 15
        ]
    );
}

#[simd_test]
fn simd_swizzle_u16x8<S: Simd>(simd: S) {
    let value = u16x8::simd_from(simd, [257, 514, 771, 1028, 1285, 1542, 1799, 2056]);
    let result = simd_swizzle!(value, [7, 0, 5, 2, 7, 4, 1, 6]);
    assert_eq!(*result, [2056, 257, 1542, 771, 2056, 1285, 514, 1799]);
}

#[simd_test]
fn simd_swizzle_i16x8<S: Simd>(simd: S) {
    let value = i16x8::simd_from(simd, [257, -514, 771, -1028, 1285, -1542, 1799, -2056]);
    let result = simd_swizzle!(value, [7, 0, 5, 2, 7, 4, 1, 6]);
    assert_eq!(*result, [-2056, 257, -1542, 771, -2056, 1285, -514, 1799]);
}

#[simd_test]
fn simd_swizzle_u32x4<S: Simd>(simd: S) {
    let value = u32x4::simd_from(simd, [65537, 131074, 196611, 262148]);
    let result = simd_swizzle!(value, [3, 0, 1, 2]);
    assert_eq!(*result, [262148, 65537, 131074, 196611]);
}

#[simd_test]
fn simd_swizzle_i32x4<S: Simd>(simd: S) {
    let value = i32x4::simd_from(simd, [65537, -131074, 196611, -262148]);
    let result = simd_swizzle!(value, [3, 0, 1, 2]);
    assert_eq!(*result, [-262148, 65537, -131074, 196611]);
}

#[simd_test]
fn simd_swizzle_u64x2<S: Simd>(simd: S) {
    let value = u64x2::simd_from(simd, [4294967297, 8589934594]);
    let result = simd_swizzle!(value, [1, 0]);
    assert_eq!(*result, [8589934594, 4294967297]);
}

#[simd_test]
fn simd_swizzle_i64x2<S: Simd>(simd: S) {
    let value = i64x2::simd_from(simd, [4294967297, -8589934594]);
    let result = simd_swizzle!(value, [1, 0]);
    assert_eq!(*result, [-8589934594, 4294967297]);
}

#[simd_test]
fn simd_swizzle_f32x4_bits<S: Simd>(simd: S) {
    let bits = u32x4::simd_from(simd, [0x80000000, 0x7fc12345, 0x7f812345, 0x0]);
    let value: f32x4<S> = bits.bitcast();
    let result: u32x4<S> = simd_swizzle!(value, [3, 0, 1, 2]).bitcast();
    assert_eq!(*result, [0x0, 0x80000000, 0x7fc12345, 0x7f812345]);
}

#[simd_test]
fn simd_swizzle_f64x2_bits<S: Simd>(simd: S) {
    let bits = u64x2::simd_from(simd, [0x8000000000000000, 0x7ff8123456789abc]);
    let value: f64x2<S> = bits.bitcast();
    let result: u64x2<S> = simd_swizzle!(value, [1, 0]).bitcast();
    assert_eq!(*result, [0x7ff8123456789abc, 0x8000000000000000]);
}

#[simd_test]
fn simd_swizzle_u8x32<S: Simd>(simd: S) {
    let value = u8x32::simd_from(
        simd,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let result = simd_swizzle!(
        value,
        [
            31, 8, 13, 18, 23, 28, 1, 6, 11, 16, 21, 26, 31, 4, 9, 14, 19, 24, 29, 2, 7, 12, 17,
            22, 27, 0, 5, 10, 15, 20, 25, 30
        ]
    );
    assert_eq!(
        *result,
        [
            32, 9, 14, 19, 24, 29, 2, 7, 12, 17, 22, 27, 32, 5, 10, 15, 20, 25, 30, 3, 8, 13, 18,
            23, 28, 1, 6, 11, 16, 21, 26, 31
        ]
    );
}

#[simd_test]
fn simd_swizzle_i8x32<S: Simd>(simd: S) {
    let value = i8x32::simd_from(
        simd,
        [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    let result = simd_swizzle!(
        value,
        [
            31, 8, 13, 18, 23, 28, 1, 6, 11, 16, 21, 26, 31, 4, 9, 14, 19, 24, 29, 2, 7, 12, 17,
            22, 27, 0, 5, 10, 15, 20, 25, 30
        ]
    );
    assert_eq!(
        *result,
        [
            -32, 9, -14, 19, -24, 29, -2, 7, -12, 17, -22, 27, -32, 5, -10, 15, -20, 25, -30, 3,
            -8, 13, -18, 23, -28, 1, -6, 11, -16, 21, -26, 31
        ]
    );
}

#[simd_test]
fn simd_swizzle_u16x16<S: Simd>(simd: S) {
    let value = u16x16::simd_from(
        simd,
        [
            257, 514, 771, 1028, 1285, 1542, 1799, 2056, 2313, 2570, 2827, 3084, 3341, 3598, 3855,
            4112,
        ],
    );
    let result = simd_swizzle!(
        value,
        [15, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14]
    );
    assert_eq!(
        *result,
        [
            4112, 2313, 3598, 771, 2056, 3341, 514, 1799, 3084, 257, 1542, 2827, 4112, 1285, 2570,
            3855
        ]
    );
}

#[simd_test]
fn simd_swizzle_i16x16<S: Simd>(simd: S) {
    let value = i16x16::simd_from(
        simd,
        [
            257, -514, 771, -1028, 1285, -1542, 1799, -2056, 2313, -2570, 2827, -3084, 3341, -3598,
            3855, -4112,
        ],
    );
    let result = simd_swizzle!(
        value,
        [15, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14]
    );
    assert_eq!(
        *result,
        [
            -4112, 2313, -3598, 771, -2056, 3341, -514, 1799, -3084, 257, -1542, 2827, -4112, 1285,
            -2570, 3855
        ]
    );
}

#[simd_test]
fn simd_swizzle_u32x8<S: Simd>(simd: S) {
    let value = u32x8::simd_from(
        simd,
        [
            65537, 131074, 196611, 262148, 327685, 393222, 458759, 524296,
        ],
    );
    let result = simd_swizzle!(value, [7, 0, 5, 2, 7, 4, 1, 6]);
    assert_eq!(
        *result,
        [
            524296, 65537, 393222, 196611, 524296, 327685, 131074, 458759
        ]
    );
}

#[simd_test]
fn simd_swizzle_i32x8<S: Simd>(simd: S) {
    let value = i32x8::simd_from(
        simd,
        [
            65537, -131074, 196611, -262148, 327685, -393222, 458759, -524296,
        ],
    );
    let result = simd_swizzle!(value, [7, 0, 5, 2, 7, 4, 1, 6]);
    assert_eq!(
        *result,
        [
            -524296, 65537, -393222, 196611, -524296, 327685, -131074, 458759
        ]
    );
}

#[simd_test]
fn simd_swizzle_u64x4<S: Simd>(simd: S) {
    let value = u64x4::simd_from(simd, [4294967297, 8589934594, 12884901891, 17179869188]);
    let result = simd_swizzle!(value, [3, 0, 1, 2]);
    assert_eq!(*result, [17179869188, 4294967297, 8589934594, 12884901891]);
}

#[simd_test]
fn simd_swizzle_i64x4<S: Simd>(simd: S) {
    let value = i64x4::simd_from(simd, [4294967297, -8589934594, 12884901891, -17179869188]);
    let result = simd_swizzle!(value, [3, 0, 1, 2]);
    assert_eq!(
        *result,
        [-17179869188, 4294967297, -8589934594, 12884901891]
    );
}

#[simd_test]
fn simd_swizzle_f32x8_bits<S: Simd>(simd: S) {
    let bits = u32x8::simd_from(
        simd,
        [
            0x80000000, 0x7fc12345, 0x7f812345, 0x0, 0x3f800000, 0xbf800000, 0x12345678, 0x7f800000,
        ],
    );
    let value: f32x8<S> = bits.bitcast();
    let result: u32x8<S> = simd_swizzle!(value, [7, 0, 5, 2, 7, 4, 1, 6]).bitcast();
    assert_eq!(
        *result,
        [
            0x7f800000, 0x80000000, 0xbf800000, 0x7f812345, 0x7f800000, 0x3f800000, 0x7fc12345,
            0x12345678
        ]
    );
}

#[simd_test]
fn simd_swizzle_f64x4_bits<S: Simd>(simd: S) {
    let bits = u64x4::simd_from(
        simd,
        [
            0x8000000000000000,
            0x7ff8123456789abc,
            0x7ff0123456789abc,
            0x0,
        ],
    );
    let value: f64x4<S> = bits.bitcast();
    let result: u64x4<S> = simd_swizzle!(value, [3, 0, 1, 2]).bitcast();
    assert_eq!(
        *result,
        [
            0x0,
            0x8000000000000000,
            0x7ff8123456789abc,
            0x7ff0123456789abc
        ]
    );
}

#[simd_test]
fn simd_swizzle_u8x64<S: Simd>(simd: S) {
    let value = u8x64::simd_from(
        simd,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ],
    );
    let result = simd_swizzle!(
        value,
        [
            63, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 4, 9, 14, 19, 24, 29, 34, 39, 44,
            49, 54, 59, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 1, 6, 11, 16, 21, 26, 31,
            36, 41, 46, 51, 56, 61, 2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62
        ]
    );
    assert_eq!(
        *result,
        [
            64, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 5, 10, 15, 20, 25, 30, 35, 40, 45,
            50, 55, 60, 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 2, 7, 12, 17, 22, 27, 32,
            37, 42, 47, 52, 57, 62, 3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63
        ]
    );
}

#[simd_test]
fn simd_swizzle_i8x64<S: Simd>(simd: S) {
    let value = i8x64::simd_from(
        simd,
        [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32, 33, -34, 35, -36, 37, -38, 39, -40,
            41, -42, 43, -44, 45, -46, 47, -48, 49, -50, 51, -52, 53, -54, 55, -56, 57, -58, 59,
            -60, 61, -62, 63, -64,
        ],
    );
    let result = simd_swizzle!(
        value,
        [
            63, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 4, 9, 14, 19, 24, 29, 34, 39, 44,
            49, 54, 59, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 1, 6, 11, 16, 21, 26, 31,
            36, 41, 46, 51, 56, 61, 2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62
        ]
    );
    assert_eq!(
        *result,
        [
            -64, 9, -14, 19, -24, 29, -34, 39, -44, 49, -54, 59, -64, 5, -10, 15, -20, 25, -30, 35,
            -40, 45, -50, 55, -60, 1, -6, 11, -16, 21, -26, 31, -36, 41, -46, 51, -56, 61, -2, 7,
            -12, 17, -22, 27, -32, 37, -42, 47, -52, 57, -62, 3, -8, 13, -18, 23, -28, 33, -38, 43,
            -48, 53, -58, 63
        ]
    );
}

#[simd_test]
fn simd_swizzle_u16x32<S: Simd>(simd: S) {
    let value = u16x32::simd_from(
        simd,
        [
            257, 514, 771, 1028, 1285, 1542, 1799, 2056, 2313, 2570, 2827, 3084, 3341, 3598, 3855,
            4112, 4369, 4626, 4883, 5140, 5397, 5654, 5911, 6168, 6425, 6682, 6939, 7196, 7453,
            7710, 7967, 8224,
        ],
    );
    let result = simd_swizzle!(
        value,
        [
            31, 8, 13, 18, 23, 28, 1, 6, 11, 16, 21, 26, 31, 4, 9, 14, 19, 24, 29, 2, 7, 12, 17,
            22, 27, 0, 5, 10, 15, 20, 25, 30
        ]
    );
    assert_eq!(
        *result,
        [
            8224, 2313, 3598, 4883, 6168, 7453, 514, 1799, 3084, 4369, 5654, 6939, 8224, 1285,
            2570, 3855, 5140, 6425, 7710, 771, 2056, 3341, 4626, 5911, 7196, 257, 1542, 2827, 4112,
            5397, 6682, 7967
        ]
    );
}

#[simd_test]
fn simd_swizzle_i16x32<S: Simd>(simd: S) {
    let value = i16x32::simd_from(
        simd,
        [
            257, -514, 771, -1028, 1285, -1542, 1799, -2056, 2313, -2570, 2827, -3084, 3341, -3598,
            3855, -4112, 4369, -4626, 4883, -5140, 5397, -5654, 5911, -6168, 6425, -6682, 6939,
            -7196, 7453, -7710, 7967, -8224,
        ],
    );
    let result = simd_swizzle!(
        value,
        [
            31, 8, 13, 18, 23, 28, 1, 6, 11, 16, 21, 26, 31, 4, 9, 14, 19, 24, 29, 2, 7, 12, 17,
            22, 27, 0, 5, 10, 15, 20, 25, 30
        ]
    );
    assert_eq!(
        *result,
        [
            -8224, 2313, -3598, 4883, -6168, 7453, -514, 1799, -3084, 4369, -5654, 6939, -8224,
            1285, -2570, 3855, -5140, 6425, -7710, 771, -2056, 3341, -4626, 5911, -7196, 257,
            -1542, 2827, -4112, 5397, -6682, 7967
        ]
    );
}

#[simd_test]
fn simd_swizzle_u32x16<S: Simd>(simd: S) {
    let value = u32x16::simd_from(
        simd,
        [
            65537, 131074, 196611, 262148, 327685, 393222, 458759, 524296, 589833, 655370, 720907,
            786444, 851981, 917518, 983055, 1048592,
        ],
    );
    let result = simd_swizzle!(
        value,
        [15, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14]
    );
    assert_eq!(
        *result,
        [
            1048592, 589833, 917518, 196611, 524296, 851981, 131074, 458759, 786444, 65537, 393222,
            720907, 1048592, 327685, 655370, 983055
        ]
    );
}

#[simd_test]
fn simd_swizzle_i32x16<S: Simd>(simd: S) {
    let value = i32x16::simd_from(
        simd,
        [
            65537, -131074, 196611, -262148, 327685, -393222, 458759, -524296, 589833, -655370,
            720907, -786444, 851981, -917518, 983055, -1048592,
        ],
    );
    let result = simd_swizzle!(
        value,
        [15, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14]
    );
    assert_eq!(
        *result,
        [
            -1048592, 589833, -917518, 196611, -524296, 851981, -131074, 458759, -786444, 65537,
            -393222, 720907, -1048592, 327685, -655370, 983055
        ]
    );
}

#[simd_test]
fn simd_swizzle_u64x8<S: Simd>(simd: S) {
    let value = u64x8::simd_from(
        simd,
        [
            4294967297,
            8589934594,
            12884901891,
            17179869188,
            21474836485,
            25769803782,
            30064771079,
            34359738376,
        ],
    );
    let result = simd_swizzle!(value, [7, 0, 5, 2, 7, 4, 1, 6]);
    assert_eq!(
        *result,
        [
            34359738376,
            4294967297,
            25769803782,
            12884901891,
            34359738376,
            21474836485,
            8589934594,
            30064771079
        ]
    );
}

#[simd_test]
fn simd_swizzle_i64x8<S: Simd>(simd: S) {
    let value = i64x8::simd_from(
        simd,
        [
            4294967297,
            -8589934594,
            12884901891,
            -17179869188,
            21474836485,
            -25769803782,
            30064771079,
            -34359738376,
        ],
    );
    let result = simd_swizzle!(value, [7, 0, 5, 2, 7, 4, 1, 6]);
    assert_eq!(
        *result,
        [
            -34359738376,
            4294967297,
            -25769803782,
            12884901891,
            -34359738376,
            21474836485,
            -8589934594,
            30064771079
        ]
    );
}

#[simd_test]
fn simd_swizzle_f32x16_bits<S: Simd>(simd: S) {
    let bits = u32x16::simd_from(
        simd,
        [
            0x80000000, 0x7fc12345, 0x7f812345, 0x0, 0x3f800000, 0xbf800000, 0x12345678,
            0x7f800000, 0x80000000, 0x7fc12345, 0x7f812345, 0x0, 0x3f800000, 0xbf800000,
            0x12345678, 0x7f800000,
        ],
    );
    let value: f32x16<S> = bits.bitcast();
    let result: u32x16<S> = simd_swizzle!(
        value,
        [15, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14]
    )
    .bitcast();
    assert_eq!(
        *result,
        [
            0x7f800000, 0x80000000, 0xbf800000, 0x7f812345, 0x7f800000, 0x3f800000, 0x7fc12345,
            0x12345678, 0x0, 0x80000000, 0xbf800000, 0x7f812345, 0x7f800000, 0x3f800000,
            0x7fc12345, 0x12345678
        ]
    );
}

#[simd_test]
fn simd_swizzle_f64x8_bits<S: Simd>(simd: S) {
    let bits = u64x8::simd_from(
        simd,
        [
            0x8000000000000000,
            0x7ff8123456789abc,
            0x7ff0123456789abc,
            0x0,
            0x3ff0000000000000,
            0xbff0000000000000,
            0x123456789abcdef0,
            0x7ff0000000000000,
        ],
    );
    let value: f64x8<S> = bits.bitcast();
    let result: u64x8<S> = simd_swizzle!(value, [7, 0, 5, 2, 7, 4, 1, 6]).bitcast();
    assert_eq!(
        *result,
        [
            0x7ff0000000000000,
            0x8000000000000000,
            0xbff0000000000000,
            0x7ff0123456789abc,
            0x7ff0000000000000,
            0x3ff0000000000000,
            0x7ff8123456789abc,
            0x123456789abcdef0
        ]
    );
}

#[simd_test]
fn simd_swizzle_identity<S: Simd>(simd: S) {
    let value = u32x4::simd_from(simd, [0x12345678, 0xabcdef01, 0x98765432, 0x10203040]);
    assert_eq!(*simd_swizzle!(value, [0, 1, 2, 3]), *value);
}

#[simd_test]
fn simd_swizzle_array_and_slice<S: Simd>(simd: S) {
    const ARRAY: [usize; 4] = [3, 2, 1, 0];
    const SLICE: &[usize] = &[2, 0, 3, 1];
    let value = u32x4::simd_from(simd, [10, 20, 30, 40]);
    assert_eq!(*simd_swizzle!(value, ARRAY), [40, 30, 20, 10]);
    assert_eq!(*simd_swizzle!(value, &ARRAY), [40, 30, 20, 10]);
    assert_eq!(*simd_swizzle!(value, SLICE,), [30, 10, 40, 20]);
    assert_eq!(*simd_swizzle!(value, &[1, 1, 1, 1]), [20, 20, 20, 20]);
}

#[simd_test]
fn simd_swizzle_native_width<S: Simd>(simd: S) {
    const PAIRS: [usize; 16] = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14];
    let value = S::f32s::from_fn(simd, |i| i as f32);
    let result = simd_swizzle!(value, PAIRS.split_at(S::f32s::LEN).0);
    for i in 0..S::f32s::LEN {
        assert_eq!(result[i], (i ^ 1) as f32);
    }
}

#[simd_test]
fn simd_swizzle_generic_vector<S: Simd>(simd: S) {
    #[inline(always)]
    fn pairwise<S: Simd, V: SimdBase<S>>(value: V) -> V {
        const PAIRS: [usize; 64] = [
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
            25, 24, 27, 26, 29, 28, 31, 30, 33, 32, 35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44,
            47, 46, 49, 48, 51, 50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62,
        ];
        simd_swizzle!(value, PAIRS.split_at(V::LEN).0)
    }
    let value = u64x8::simd_from(simd, [10, 20, 30, 40, 50, 60, 70, 80]);
    assert_eq!(*pairwise(value), [20, 10, 40, 30, 60, 50, 80, 70]);
    let bytes = S::u8s::from_fn(simd, |i| i as u8);
    let result = pairwise(bytes);
    for i in 0..S::u8s::LEN {
        assert_eq!(result[i], (i ^ 1) as u8);
    }
}

#[simd_test]
fn simd_swizzle_const_parameters<S: Simd>(simd: S) {
    const fn rotate(offset: usize) -> [usize; 4] {
        [
            offset % 4,
            (offset % 4 + 1) % 4,
            (offset % 4 + 2) % 4,
            (offset % 4 + 3) % 4,
        ]
    }
    #[inline(always)]
    fn apply<S: Simd, const OFFSET: usize>(value: u32x4<S>) -> u32x4<S> {
        simd_swizzle!(value, rotate(OFFSET))
    }
    let value = u32x4::simd_from(simd, [10, 20, 30, 40]);
    assert_eq!(*apply::<S, 3>(value), [40, 10, 20, 30]);
    assert_eq!(*apply::<S, 6>(value), [30, 40, 10, 20]);
}

#[simd_test]
fn simd_swizzle_associated_constant<S: Simd>(simd: S) {
    trait Pattern {
        const INDEX: [usize; 4];
    }
    struct Reverse;
    impl Pattern for Reverse {
        const INDEX: [usize; 4] = [3, 2, 1, 0];
    }
    #[inline(always)]
    fn apply<S: Simd, P: Pattern>(value: u32x4<S>) -> u32x4<S> {
        simd_swizzle!(value, P::INDEX)
    }
    let value = u32x4::simd_from(simd, [10, 20, 30, 40]);
    assert_eq!(*apply::<S, Reverse>(value), [40, 30, 20, 10]);
}

#[simd_test]
fn simd_swizzle_evaluates_vector_once<S: Simd>(simd: S) {
    let mut count = 0;
    let result = simd_swizzle!(
        {
            count += 1;
            u32x4::simd_from(simd, [10, 20, 30, 40])
        },
        [2, 0, 3, 1]
    );
    assert_eq!(count, 1);
    assert_eq!(*result, [30, 10, 40, 20]);
}
