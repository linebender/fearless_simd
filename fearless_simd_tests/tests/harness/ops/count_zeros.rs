// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported integer vector type.

#[simd_test]
fn count_zeros_i8x16<S: Simd>(simd: S) {
    let values: [i8; 16] = [
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
    ];
    let expected = values.map(|value| i8::try_from(value.count_zeros()).unwrap());
    let result: i8x16<S> = i8x16::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i8x32<S: Simd>(simd: S) {
    let values: [i8; 32] = [
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
    ];
    let expected = values.map(|value| i8::try_from(value.count_zeros()).unwrap());
    let result: i8x32<S> = i8x32::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i8x64<S: Simd>(simd: S) {
    let values: [i8; 64] = [
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
        0x55_i8,
        -0x56_i8,
        0_i8,
        -1_i8,
        i8::MIN,
        i8::MAX,
    ];
    let expected = values.map(|value| i8::try_from(value.count_zeros()).unwrap());
    let result: i8x64<S> = i8x64::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u8x16<S: Simd>(simd: S) {
    let values: [u8; 16] = [
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
    ];
    let expected = values.map(|value| u8::try_from(value.count_zeros()).unwrap());
    let result: u8x16<S> = u8x16::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u8x32<S: Simd>(simd: S) {
    let values: [u8; 32] = [
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
    ];
    let expected = values.map(|value| u8::try_from(value.count_zeros()).unwrap());
    let result: u8x32<S> = u8x32::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u8x64<S: Simd>(simd: S) {
    let values: [u8; 64] = [
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
        0x55_u8,
        0xaa_u8,
        0_u8,
        u8::MAX,
        1_u8 << 7,
        u8::MAX >> 1,
    ];
    let expected = values.map(|value| u8::try_from(value.count_zeros()).unwrap());
    let result: u8x64<S> = u8x64::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i16x8<S: Simd>(simd: S) {
    let values: [i16; 8] = [
        0_i16,
        -1_i16,
        i16::MIN,
        i16::MAX,
        0x5555_i16,
        -0x5556_i16,
        0_i16,
        -1_i16,
    ];
    let expected = values.map(|value| i16::try_from(value.count_zeros()).unwrap());
    let result: i16x8<S> = i16x8::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i16x16<S: Simd>(simd: S) {
    let values: [i16; 16] = [
        0_i16,
        -1_i16,
        i16::MIN,
        i16::MAX,
        0x5555_i16,
        -0x5556_i16,
        0_i16,
        -1_i16,
        i16::MIN,
        i16::MAX,
        0x5555_i16,
        -0x5556_i16,
        0_i16,
        -1_i16,
        i16::MIN,
        i16::MAX,
    ];
    let expected = values.map(|value| i16::try_from(value.count_zeros()).unwrap());
    let result: i16x16<S> = i16x16::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i16x32<S: Simd>(simd: S) {
    let values: [i16; 32] = [
        0_i16,
        -1_i16,
        i16::MIN,
        i16::MAX,
        0x5555_i16,
        -0x5556_i16,
        0_i16,
        -1_i16,
        i16::MIN,
        i16::MAX,
        0x5555_i16,
        -0x5556_i16,
        0_i16,
        -1_i16,
        i16::MIN,
        i16::MAX,
        0x5555_i16,
        -0x5556_i16,
        0_i16,
        -1_i16,
        i16::MIN,
        i16::MAX,
        0x5555_i16,
        -0x5556_i16,
        0_i16,
        -1_i16,
        i16::MIN,
        i16::MAX,
        0x5555_i16,
        -0x5556_i16,
        0_i16,
        -1_i16,
    ];
    let expected = values.map(|value| i16::try_from(value.count_zeros()).unwrap());
    let result: i16x32<S> = i16x32::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u16x8<S: Simd>(simd: S) {
    let values: [u16; 8] = [
        0_u16,
        u16::MAX,
        1_u16 << 15,
        u16::MAX >> 1,
        0x5555_u16,
        0xaaaa_u16,
        0_u16,
        u16::MAX,
    ];
    let expected = values.map(|value| u16::try_from(value.count_zeros()).unwrap());
    let result: u16x8<S> = u16x8::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u16x16<S: Simd>(simd: S) {
    let values: [u16; 16] = [
        0_u16,
        u16::MAX,
        1_u16 << 15,
        u16::MAX >> 1,
        0x5555_u16,
        0xaaaa_u16,
        0_u16,
        u16::MAX,
        1_u16 << 15,
        u16::MAX >> 1,
        0x5555_u16,
        0xaaaa_u16,
        0_u16,
        u16::MAX,
        1_u16 << 15,
        u16::MAX >> 1,
    ];
    let expected = values.map(|value| u16::try_from(value.count_zeros()).unwrap());
    let result: u16x16<S> = u16x16::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u16x32<S: Simd>(simd: S) {
    let values: [u16; 32] = [
        0_u16,
        u16::MAX,
        1_u16 << 15,
        u16::MAX >> 1,
        0x5555_u16,
        0xaaaa_u16,
        0_u16,
        u16::MAX,
        1_u16 << 15,
        u16::MAX >> 1,
        0x5555_u16,
        0xaaaa_u16,
        0_u16,
        u16::MAX,
        1_u16 << 15,
        u16::MAX >> 1,
        0x5555_u16,
        0xaaaa_u16,
        0_u16,
        u16::MAX,
        1_u16 << 15,
        u16::MAX >> 1,
        0x5555_u16,
        0xaaaa_u16,
        0_u16,
        u16::MAX,
        1_u16 << 15,
        u16::MAX >> 1,
        0x5555_u16,
        0xaaaa_u16,
        0_u16,
        u16::MAX,
    ];
    let expected = values.map(|value| u16::try_from(value.count_zeros()).unwrap());
    let result: u16x32<S> = u16x32::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i32x4<S: Simd>(simd: S) {
    let values: [i32; 4] = [0_i32, -1_i32, i32::MIN, i32::MAX];
    let expected = values.map(|value| value.count_zeros().cast_signed());
    let result: i32x4<S> = i32x4::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = [
        0_i32,
        -1_i32,
        i32::MIN,
        i32::MAX,
        0x5555_5555_i32,
        -0x5555_5556_i32,
        0_i32,
        -1_i32,
    ];
    let expected = values.map(|value| value.count_zeros().cast_signed());
    let result: i32x8<S> = i32x8::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i32x16<S: Simd>(simd: S) {
    let values: [i32; 16] = [
        0_i32,
        -1_i32,
        i32::MIN,
        i32::MAX,
        0x5555_5555_i32,
        -0x5555_5556_i32,
        0_i32,
        -1_i32,
        i32::MIN,
        i32::MAX,
        0x5555_5555_i32,
        -0x5555_5556_i32,
        0_i32,
        -1_i32,
        i32::MIN,
        i32::MAX,
    ];
    let expected = values.map(|value| value.count_zeros().cast_signed());
    let result: i32x16<S> = i32x16::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u32x4<S: Simd>(simd: S) {
    let values: [u32; 4] = [0_u32, u32::MAX, 1_u32 << 31, u32::MAX >> 1];
    let expected = values.map(|value| value.count_zeros());
    let result: u32x4<S> = u32x4::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u32x8<S: Simd>(simd: S) {
    let values: [u32; 8] = [
        0_u32,
        u32::MAX,
        1_u32 << 31,
        u32::MAX >> 1,
        0x5555_5555_u32,
        0xaaaa_aaaa_u32,
        0_u32,
        u32::MAX,
    ];
    let expected = values.map(|value| value.count_zeros());
    let result: u32x8<S> = u32x8::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u32x16<S: Simd>(simd: S) {
    let values: [u32; 16] = [
        0_u32,
        u32::MAX,
        1_u32 << 31,
        u32::MAX >> 1,
        0x5555_5555_u32,
        0xaaaa_aaaa_u32,
        0_u32,
        u32::MAX,
        1_u32 << 31,
        u32::MAX >> 1,
        0x5555_5555_u32,
        0xaaaa_aaaa_u32,
        0_u32,
        u32::MAX,
        1_u32 << 31,
        u32::MAX >> 1,
    ];
    let expected = values.map(|value| value.count_zeros());
    let result: u32x16<S> = u32x16::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i64x2<S: Simd>(simd: S) {
    let values: [i64; 2] = [0_i64, -1_i64];
    let expected = values.map(|value| i64::from(value.count_zeros()));
    let result: i64x2<S> = i64x2::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i64x4<S: Simd>(simd: S) {
    let values: [i64; 4] = [0_i64, -1_i64, i64::MIN, i64::MAX];
    let expected = values.map(|value| i64::from(value.count_zeros()));
    let result: i64x4<S> = i64x4::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_i64x8<S: Simd>(simd: S) {
    let values: [i64; 8] = [
        0_i64,
        -1_i64,
        i64::MIN,
        i64::MAX,
        0x5555_5555_5555_5555_i64,
        -0x5555_5555_5555_5556_i64,
        0_i64,
        -1_i64,
    ];
    let expected = values.map(|value| i64::from(value.count_zeros()));
    let result: i64x8<S> = i64x8::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u64x2<S: Simd>(simd: S) {
    let values: [u64; 2] = [0_u64, u64::MAX];
    let expected = values.map(|value| u64::from(value.count_zeros()));
    let result: u64x2<S> = u64x2::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u64x4<S: Simd>(simd: S) {
    let values: [u64; 4] = [0_u64, u64::MAX, 1_u64 << 63, u64::MAX >> 1];
    let expected = values.map(|value| u64::from(value.count_zeros()));
    let result: u64x4<S> = u64x4::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}

#[simd_test]
fn count_zeros_u64x8<S: Simd>(simd: S) {
    let values: [u64; 8] = [
        0_u64,
        u64::MAX,
        1_u64 << 63,
        u64::MAX >> 1,
        0x5555_5555_5555_5555_u64,
        0xaaaa_aaaa_aaaa_aaaa_u64,
        0_u64,
        u64::MAX,
    ];
    let expected = values.map(|value| u64::from(value.count_zeros()));
    let result: u64x8<S> = u64x8::from_slice(simd, &values).count_zeros();
    assert_eq!(*result, expected);
}
