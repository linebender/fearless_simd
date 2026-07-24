// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn from_fn_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_fn(simd, |i| i as f32 * 2.0);
    assert_eq!(
        *a,
        [
            0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0,
            30.0
        ]
    );
}

#[simd_test]
fn from_fn_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_fn(simd, |i| i.try_into().unwrap());
    assert_eq!(
        *a,
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
        ]
    );
}

#[simd_test]
fn from_fn_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_fn(simd, |i| (i * 2).try_into().unwrap());
    assert_eq!(
        *a,
        [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44,
            46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
            90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124,
            126
        ]
    );
}

#[simd_test]
fn from_fn_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_fn(simd, |i| i16::try_from(i).unwrap() * 100);
    assert_eq!(
        *a,
        [
            0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
            1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
            3000, 3100
        ]
    );
}

#[simd_test]
fn from_fn_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_fn(simd, |i| u16::try_from(i).unwrap() + 1000);
    assert_eq!(
        *a,
        [
            1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
            1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027,
            1028, 1029, 1030, 1031
        ]
    );
}

#[simd_test]
fn from_fn_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_fn(simd, |i| {
        let i: i32 = i.try_into().unwrap();
        i * i
    });
    assert_eq!(
        *a,
        [
            0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225
        ]
    );
}

#[simd_test]
fn from_fn_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_fn(simd, |i| 1_u32 << i);
    assert_eq!(
        *a,
        [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
        ]
    );
}

// Additional concrete rows for this operation.

#[simd_test]
fn from_fn_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_fn(simd, |i| i as i64 - 1);
    let expected: [i64; 2] = core::array::from_fn(|i| i as i64 - 1);
    assert_eq!(*a, expected);
}

#[simd_test]
fn from_fn_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_fn(simd, |i| i as i64 - 1);
    let expected: [i64; 4] = core::array::from_fn(|i| i as i64 - 1);
    assert_eq!(*a, expected);
}

#[simd_test]
fn from_fn_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_fn(simd, |i| i as i64 - 1);
    let expected: [i64; 8] = core::array::from_fn(|i| i as i64 - 1);
    assert_eq!(*a, expected);
}

#[simd_test]
fn from_fn_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_fn(simd, |i| i as u64 + 1);
    let expected: [u64; 2] = core::array::from_fn(|i| i as u64 + 1);
    assert_eq!(*a, expected);
}

#[simd_test]
fn from_fn_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_fn(simd, |i| i as u64 + 1);
    let expected: [u64; 4] = core::array::from_fn(|i| i as u64 + 1);
    assert_eq!(*a, expected);
}

#[simd_test]
fn from_fn_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_fn(simd, |i| i as u64 + 1);
    let expected: [u64; 8] = core::array::from_fn(|i| i as u64 + 1);
    assert_eq!(*a, expected);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn from_fn_i8x16<S: Simd>(simd: S) {
    let expected: [i8; 16] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let result = i8x16::from_fn(simd, |i| (i % 23) as i8 + 10_i8);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_u8x16<S: Simd>(simd: S) {
    let expected: [u8; 16] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let result = u8x16::from_fn(simd, |i| (i % 23) as u8 + 10_u8);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_i8x32<S: Simd>(simd: S) {
    let expected: [i8; 32] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let result = i8x32::from_fn(simd, |i| (i % 23) as i8 + 10_i8);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_u8x32<S: Simd>(simd: S) {
    let expected: [u8; 32] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let result = u8x32::from_fn(simd, |i| (i % 23) as u8 + 10_u8);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_i16x8<S: Simd>(simd: S) {
    let expected: [i16; 8] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let result = i16x8::from_fn(simd, |i| (i % 23) as i16 + 10_i16);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_u16x8<S: Simd>(simd: S) {
    let expected: [u16; 8] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let result = u16x8::from_fn(simd, |i| (i % 23) as u16 + 10_u16);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_i16x16<S: Simd>(simd: S) {
    let expected: [i16; 16] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let result = i16x16::from_fn(simd, |i| (i % 23) as i16 + 10_i16);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_u16x16<S: Simd>(simd: S) {
    let expected: [u16; 16] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let result = u16x16::from_fn(simd, |i| (i % 23) as u16 + 10_u16);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_f32x4<S: Simd>(simd: S) {
    let expected: [f32; 4] = core::array::from_fn(|i| i as f32 + 1.25_f32);
    let result = f32x4::from_fn(simd, |i| i as f32 + 1.25_f32);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_i32x4<S: Simd>(simd: S) {
    let expected: [i32; 4] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let result = i32x4::from_fn(simd, |i| (i % 23) as i32 + 10_i32);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_u32x4<S: Simd>(simd: S) {
    let expected: [u32; 4] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let result = u32x4::from_fn(simd, |i| (i % 23) as u32 + 10_u32);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_f32x8<S: Simd>(simd: S) {
    let expected: [f32; 8] = core::array::from_fn(|i| i as f32 + 1.25_f32);
    let result = f32x8::from_fn(simd, |i| i as f32 + 1.25_f32);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_i32x8<S: Simd>(simd: S) {
    let expected: [i32; 8] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let result = i32x8::from_fn(simd, |i| (i % 23) as i32 + 10_i32);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_u32x8<S: Simd>(simd: S) {
    let expected: [u32; 8] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let result = u32x8::from_fn(simd, |i| (i % 23) as u32 + 10_u32);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_f64x2<S: Simd>(simd: S) {
    let expected: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let result = f64x2::from_fn(simd, |i| i as f64 + 1.25_f64);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_f64x4<S: Simd>(simd: S) {
    let expected: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let result = f64x4::from_fn(simd, |i| i as f64 + 1.25_f64);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn from_fn_f64x8<S: Simd>(simd: S) {
    let expected: [f64; 8] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let result = f64x8::from_fn(simd, |i| i as f64 + 1.25_f64);
    assert_eq!(result.as_slice(), expected.as_slice());
}
