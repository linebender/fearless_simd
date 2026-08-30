// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn reduce_max_f32x4<S: Simd>(simd: S) {
    let value = f32x4::from_slice(simd, &[3.0, f32::INFINITY, -2.0, f32::NEG_INFINITY]);
    assert_eq!(value.reduce_max(), f32::INFINITY);
    assert_eq!(
        f32x4::from_slice(simd, &[0.0, -0.0, 0.0, -0.0]).reduce_max(),
        0.0
    );
}

#[simd_test]
fn reduce_max_i8x16<S: Simd>(simd: S) {
    let value = i8x16::from_slice(
        simd,
        &[
            12,
            -7,
            44,
            i8::MAX,
            -99,
            0,
            63,
            -42,
            8,
            91,
            -1,
            37,
            -64,
            i8::MIN,
            5,
            72,
        ],
    );
    assert_eq!(value.reduce_max(), i8::MAX);
}

#[simd_test]
fn reduce_max_u8x16<S: Simd>(simd: S) {
    let value = u8x16::from_slice(
        simd,
        &[
            12,
            87,
            3,
            u8::MAX,
            144,
            0,
            63,
            201,
            8,
            91,
            1,
            37,
            164,
            222,
            5,
            72,
        ],
    );
    assert_eq!(value.reduce_max(), u8::MAX);
}

#[simd_test]
fn reduce_max_i16x8<S: Simd>(simd: S) {
    let value = i16x8::from_slice(simd, &[1200, -700, i16::MAX, 0, -9999, 42, i16::MIN, 7321]);
    assert_eq!(value.reduce_max(), i16::MAX);
}

#[simd_test]
fn reduce_max_u16x8<S: Simd>(simd: S) {
    let value = u16x8::from_slice(simd, &[1200, 700, u16::MAX, 81, 9999, u16::MIN, 42, 7321]);
    assert_eq!(value.reduce_max(), u16::MAX);
}

#[simd_test]
fn reduce_max_i32x4<S: Simd>(simd: S) {
    let value = i32x4::from_slice(simd, &[42, i32::MIN, i32::MAX, -700_000]);
    assert_eq!(value.reduce_max(), i32::MAX);
}

#[simd_test]
fn reduce_max_u32x4<S: Simd>(simd: S) {
    let value = u32x4::from_slice(simd, &[42, u32::MAX, 700_000, u32::MIN]);
    assert_eq!(value.reduce_max(), u32::MAX);
}

#[simd_test]
fn reduce_max_f64x2<S: Simd>(simd: S) {
    let value = f64x2::from_slice(simd, &[f64::NEG_INFINITY, f64::INFINITY]);
    assert_eq!(value.reduce_max(), f64::INFINITY);
    assert_eq!(f64x2::from_slice(simd, &[0.0, -0.0]).reduce_max(), 0.0);
}

#[simd_test]
fn reduce_max_i64x2<S: Simd>(simd: S) {
    let value = i64x2::from_slice(simd, &[i64::MIN, i64::MAX]);
    assert_eq!(value.reduce_max(), i64::MAX);
}

#[simd_test]
fn reduce_max_u64x2<S: Simd>(simd: S) {
    let value = u64x2::from_slice(simd, &[u64::MIN, u64::MAX]);
    assert_eq!(value.reduce_max(), u64::MAX);
}

#[simd_test]
fn reduce_max_f32x8<S: Simd>(simd: S) {
    let value = f32x8::from_slice(
        simd,
        &[
            3.0,
            -2.0,
            f32::NEG_INFINITY,
            9.0,
            1.0,
            f32::INFINITY,
            4.0,
            8.0,
        ],
    );
    assert_eq!(value.reduce_max(), f32::INFINITY);
    assert_eq!(
        f32x8::from_slice(simd, &[0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0]).reduce_max(),
        0.0
    );
}

#[simd_test]
fn reduce_max_i8x32<S: Simd>(simd: S) {
    let value = i8x32::from_slice(
        simd,
        &[
            12,
            -7,
            44,
            i8::MAX,
            -99,
            0,
            63,
            -42,
            8,
            91,
            -1,
            37,
            -64,
            i8::MIN,
            5,
            72,
            -18,
            26,
            100,
            -55,
            3,
            -120,
            84,
            16,
            -33,
            49,
            7,
            -76,
            110,
            67,
            -2,
            31,
        ],
    );
    assert_eq!(value.reduce_max(), i8::MAX);
}

#[simd_test]
fn reduce_max_u8x32<S: Simd>(simd: S) {
    let value = u8x32::from_slice(
        simd,
        &[
            12,
            87,
            3,
            u8::MAX,
            144,
            0,
            63,
            201,
            8,
            91,
            1,
            37,
            164,
            222,
            5,
            72,
            18,
            126,
            100,
            55,
            13,
            220,
            84,
            16,
            133,
            49,
            7,
            176,
            110,
            67,
            2,
            31,
        ],
    );
    assert_eq!(value.reduce_max(), u8::MAX);
}

#[simd_test]
fn reduce_max_i16x16<S: Simd>(simd: S) {
    let value = i16x16::from_slice(
        simd,
        &[
            1200,
            -700,
            i16::MAX,
            0,
            -9999,
            42,
            -16384,
            7321,
            81,
            -2222,
            19000,
            -5,
            640,
            -12000,
            i16::MIN,
            30001,
        ],
    );
    assert_eq!(value.reduce_max(), i16::MAX);
}

#[simd_test]
fn reduce_max_u16x16<S: Simd>(simd: S) {
    let value = u16x16::from_slice(
        simd,
        &[
            1200,
            700,
            u16::MAX,
            81,
            9999,
            32000,
            42,
            7321,
            18,
            u16::MIN,
            2222,
            19000,
            5,
            640,
            12000,
            30001,
        ],
    );
    assert_eq!(value.reduce_max(), u16::MAX);
}

#[simd_test]
fn reduce_max_i32x8<S: Simd>(simd: S) {
    let value = i32x8::from_slice(
        simd,
        &[42, -7, i32::MAX, -700_000, 81, i32::MIN, 19_000_000, -5],
    );
    assert_eq!(value.reduce_max(), i32::MAX);
}

#[simd_test]
fn reduce_max_u32x8<S: Simd>(simd: S) {
    let value = u32x8::from_slice(
        simd,
        &[
            42,
            u32::MAX,
            700_000,
            81,
            u32::MIN,
            19_000_000,
            5,
            3_000_000_000,
        ],
    );
    assert_eq!(value.reduce_max(), u32::MAX);
}

#[simd_test]
fn reduce_max_f64x4<S: Simd>(simd: S) {
    let value = f64x4::from_slice(simd, &[4.0, f64::NEG_INFINITY, -7.0, f64::INFINITY]);
    assert_eq!(value.reduce_max(), f64::INFINITY);
    assert_eq!(
        f64x4::from_slice(simd, &[0.0, -0.0, 0.0, -0.0]).reduce_max(),
        0.0
    );
}

#[simd_test]
fn reduce_max_i64x4<S: Simd>(simd: S) {
    let value = i64x4::from_slice(simd, &[42, i64::MAX, i64::MIN, -7_000_000_000]);
    assert_eq!(value.reduce_max(), i64::MAX);
}

#[simd_test]
fn reduce_max_u64x4<S: Simd>(simd: S) {
    let value = u64x4::from_slice(simd, &[42, u64::MIN, u64::MAX, 7_000_000_000]);
    assert_eq!(value.reduce_max(), u64::MAX);
}

#[simd_test]
fn reduce_max_f32x16<S: Simd>(simd: S) {
    let value = f32x16::from_fn(simd, |i| {
        if i == 13 {
            f32::INFINITY
        } else if i == 2 {
            f32::NEG_INFINITY
        } else {
            i as f32 - 4.0
        }
    });
    assert_eq!(value.reduce_max(), f32::INFINITY);
    let zeros = f32x16::from_fn(simd, |i| if i % 2 == 0 { 0.0 } else { -0.0 });
    assert_eq!(zeros.reduce_max(), 0.0);
}

#[simd_test]
fn reduce_max_i8x64<S: Simd>(simd: S) {
    let value = i8x64::from_slice(
        simd,
        &[
            12,
            -7,
            44,
            i8::MAX,
            -99,
            0,
            63,
            -42,
            8,
            91,
            -1,
            37,
            -64,
            115,
            5,
            72,
            -18,
            26,
            100,
            -55,
            3,
            -120,
            84,
            16,
            -33,
            49,
            7,
            -76,
            110,
            67,
            -2,
            31,
            54,
            -88,
            23,
            106,
            -15,
            39,
            -70,
            11,
            96,
            -4,
            58,
            -112,
            29,
            75,
            -47,
            1,
            82,
            -25,
            34,
            119,
            -61,
            14,
            69,
            -9,
            46,
            -101,
            20,
            89,
            -36,
            i8::MIN,
            6,
            77,
        ],
    );
    assert_eq!(value.reduce_max(), i8::MAX);
}

#[simd_test]
fn reduce_max_u8x64<S: Simd>(simd: S) {
    let value = u8x64::from_slice(
        simd,
        &[
            12,
            87,
            3,
            u8::MAX,
            144,
            90,
            63,
            201,
            8,
            91,
            1,
            37,
            164,
            222,
            5,
            72,
            18,
            126,
            100,
            55,
            13,
            220,
            84,
            16,
            133,
            49,
            7,
            176,
            110,
            67,
            2,
            31,
            154,
            u8::MIN,
            23,
            206,
            15,
            139,
            70,
            11,
            196,
            4,
            158,
            212,
            29,
            175,
            47,
            81,
            182,
            25,
            134,
            219,
            61,
            14,
            169,
            9,
            146,
            101,
            20,
            189,
            36,
            128,
            6,
            77,
        ],
    );
    assert_eq!(value.reduce_max(), u8::MAX);
}

#[simd_test]
fn reduce_max_i16x32<S: Simd>(simd: S) {
    let value = i16x32::from_slice(
        simd,
        &[
            1200,
            -700,
            i16::MAX,
            0,
            -9999,
            42,
            -16384,
            7321,
            81,
            -2222,
            19000,
            -5,
            640,
            -12000,
            30001,
            -301,
            5400,
            -8800,
            230,
            10600,
            -1500,
            390,
            -7000,
            1100,
            9600,
            -400,
            5800,
            i16::MIN,
            2900,
            750,
            -4700,
            100,
        ],
    );
    assert_eq!(value.reduce_max(), i16::MAX);
}

#[simd_test]
fn reduce_max_u16x32<S: Simd>(simd: S) {
    let value = u16x32::from_slice(
        simd,
        &[
            1200,
            700,
            u16::MAX,
            81,
            9999,
            32000,
            42,
            7321,
            18,
            2222,
            19000,
            5,
            640,
            12000,
            30001,
            301,
            5400,
            8800,
            u16::MIN,
            10600,
            1500,
            390,
            7000,
            1100,
            9600,
            400,
            5800,
            32768,
            2900,
            750,
            4700,
            100,
        ],
    );
    assert_eq!(value.reduce_max(), u16::MAX);
}

#[simd_test]
fn reduce_max_i32x16<S: Simd>(simd: S) {
    let value = i32x16::from_slice(
        simd,
        &[
            42,
            -7,
            i32::MAX,
            -700_000,
            81,
            19_000_000,
            -5,
            640,
            -12_000,
            30_001,
            -301,
            5_400_000,
            -8_800,
            230,
            10_600,
            i32::MIN,
        ],
    );
    assert_eq!(value.reduce_max(), i32::MAX);
}

#[simd_test]
fn reduce_max_u32x16<S: Simd>(simd: S) {
    let value = u32x16::from_slice(
        simd,
        &[
            42,
            u32::MAX,
            700_000,
            81,
            19_000_000,
            5,
            3_000_000_000,
            640,
            u32::MIN,
            30_001,
            301,
            5_400_000,
            8_800,
            230,
            10_600,
            1_500_000_000,
        ],
    );
    assert_eq!(value.reduce_max(), u32::MAX);
}

#[simd_test]
fn reduce_max_f64x8<S: Simd>(simd: S) {
    let value = f64x8::from_slice(
        simd,
        &[
            4.0,
            f64::NEG_INFINITY,
            -7.0,
            2.0,
            8.0,
            1.0,
            f64::INFINITY,
            5.0,
        ],
    );
    assert_eq!(value.reduce_max(), f64::INFINITY);
    assert_eq!(
        f64x8::from_slice(simd, &[0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0]).reduce_max(),
        0.0
    );
}

#[simd_test]
fn reduce_max_i64x8<S: Simd>(simd: S) {
    let value = i64x8::from_slice(
        simd,
        &[
            42,
            -7,
            i64::MAX,
            -7_000_000_000,
            81,
            19_000_000,
            i64::MIN,
            5_400_000_000,
        ],
    );
    assert_eq!(value.reduce_max(), i64::MAX);
}

#[simd_test]
fn reduce_max_u64x8<S: Simd>(simd: S) {
    let value = u64x8::from_slice(
        simd,
        &[
            42,
            u64::MAX,
            7_000_000_000,
            81,
            19_000_000,
            u64::MIN,
            5_400_000_000,
            12_345,
        ],
    );
    assert_eq!(value.reduce_max(), u64::MAX);
}
