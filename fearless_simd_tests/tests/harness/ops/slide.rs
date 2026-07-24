// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn slide_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*a.slide::<0>(b), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(*a.slide::<1>(b), [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(*a.slide::<2>(b), [3.0, 4.0, 5.0, 6.0]);
    assert_eq!(*a.slide::<4>(b), [5.0, 6.0, 7.0, 8.0]);
}

#[simd_test]
fn slide_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::from_slice(simd, &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    assert_eq!(*a.slide::<0>(b), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*a.slide::<1>(b), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(
        *a.slide::<4>(b),
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
    assert_eq!(
        *a.slide::<7>(b),
        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    );
    assert_eq!(
        *a.slide::<8>(b),
        [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    );
}

#[simd_test]
fn slide_f32x16<S: Simd>(simd: S) {
    let a: Vec<f32> = (1_i16..=16).map(f32::from).collect();
    let b: Vec<f32> = (17_i16..=32).map(f32::from).collect();
    let a = f32x16::from_slice(simd, &a);
    let b = f32x16::from_slice(simd, &b);
    let expected_0: Vec<f32> = (1_i16..=16).map(f32::from).collect();
    let expected_1: Vec<f32> = (2_i16..=17).map(f32::from).collect();
    let expected_8: Vec<f32> = (9_i16..=24).map(f32::from).collect();
    let expected_15: Vec<f32> = (16_i16..=31).map(f32::from).collect();
    let expected_16: Vec<f32> = (17_i16..=32).map(f32::from).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide::<8>(b).as_slice(), &expected_8);
    assert_eq!(a.slide::<15>(b).as_slice(), &expected_15);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
}

#[simd_test]
fn slide_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.0, 2.0]);
    let b = f64x2::from_slice(simd, &[3.0, 4.0]);
    assert_eq!(*a.slide::<0>(b), [1.0, 2.0]);
    assert_eq!(*a.slide::<1>(b), [2.0, 3.0]);
    assert_eq!(*a.slide::<2>(b), [3.0, 4.0]);
}

#[simd_test]
fn slide_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*a.slide::<0>(b), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(*a.slide::<1>(b), [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(*a.slide::<2>(b), [3.0, 4.0, 5.0, 6.0]);
    assert_eq!(*a.slide::<3>(b), [4.0, 5.0, 6.0, 7.0]);
    assert_eq!(*a.slide::<4>(b), [5.0, 6.0, 7.0, 8.0]);
}

#[simd_test]
fn slide_f64x8<S: Simd>(simd: S) {
    let a: Vec<f64> = (1..=8).map(f64::from).collect();
    let b: Vec<f64> = (9..=16).map(f64::from).collect();
    let a = f64x8::from_slice(simd, &a);
    let b = f64x8::from_slice(simd, &b);
    let expected_0: Vec<f64> = (1..=8).map(f64::from).collect();
    let expected_1: Vec<f64> = (2..=9).map(f64::from).collect();
    let expected_4: Vec<f64> = (5..=12).map(f64::from).collect();
    let expected_7: Vec<f64> = (8..=15).map(f64::from).collect();
    let expected_8: Vec<f64> = (9..=16).map(f64::from).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide::<4>(b).as_slice(), &expected_4);
    assert_eq!(a.slide::<7>(b).as_slice(), &expected_7);
    assert_eq!(a.slide::<8>(b).as_slice(), &expected_8);
}

#[simd_test]
fn slide_i8x16<S: Simd>(simd: S) {
    let a: Vec<i8> = (1_i8..=16).collect();
    let b: Vec<i8> = (17_i8..=32).collect();
    let a = i8x16::from_slice(simd, &a);
    let b = i8x16::from_slice(simd, &b);
    let expected_0: Vec<i8> = (1_i8..=16).collect();
    let expected_1: Vec<i8> = (2_i8..=17).collect();
    let expected_8: Vec<i8> = (9_i8..=24).collect();
    let expected_15: Vec<i8> = (16_i8..=31).collect();
    let expected_16: Vec<i8> = (17_i8..=32).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide::<8>(b).as_slice(), &expected_8);
    assert_eq!(a.slide::<15>(b).as_slice(), &expected_15);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
}

#[simd_test]
fn slide_i8x32<S: Simd>(simd: S) {
    let a: Vec<i8> = (1_i8..=32).collect();
    let b: Vec<i8> = (33_i8..=64).collect();
    let a = i8x32::from_slice(simd, &a);
    let b = i8x32::from_slice(simd, &b);
    let expected_0: Vec<i8> = (1_i8..=32).collect();
    let expected_1: Vec<i8> = (2_i8..=33).collect();
    let expected_16: Vec<i8> = (17_i8..=48).collect();
    let expected_31: Vec<i8> = (32_i8..=63).collect();
    let expected_32: Vec<i8> = (33_i8..=64).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
    assert_eq!(a.slide::<31>(b).as_slice(), &expected_31);
    assert_eq!(a.slide::<32>(b).as_slice(), &expected_32);
}

#[simd_test]
fn slide_i8x64<S: Simd>(simd: S) {
    let a: Vec<i8> = (0_i8..=63).collect();
    let b: Vec<i8> = (64_i8..=127).collect();
    let a = i8x64::from_slice(simd, &a);
    let b = i8x64::from_slice(simd, &b);
    let expected_0: Vec<i8> = (0_i8..=63).collect();
    let expected_1: Vec<i8> = (1_i8..=64).collect();
    let expected_32: Vec<i8> = (32_i8..=95).collect();
    let expected_63: Vec<i8> = (63_i8..=126).collect();
    let expected_64: Vec<i8> = (64_i8..=127).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide::<32>(b).as_slice(), &expected_32);
    assert_eq!(a.slide::<63>(b).as_slice(), &expected_63);
    assert_eq!(a.slide::<64>(b).as_slice(), &expected_64);
}

#[simd_test]
fn slide_u8x16<S: Simd>(simd: S) {
    let a: Vec<u8> = (1..=16).collect();
    let b: Vec<u8> = (17..=32).collect();
    let a = u8x16::from_slice(simd, &a);
    let b = u8x16::from_slice(simd, &b);
    let expected_0: Vec<u8> = (1..=16).collect();
    let expected_8: Vec<u8> = (9..=24).collect();
    let expected_16: Vec<u8> = (17..=32).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<8>(b).as_slice(), &expected_8);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
}

#[simd_test]
fn slide_u8x32<S: Simd>(simd: S) {
    let a: Vec<u8> = (1..=32).collect();
    let b: Vec<u8> = (33..=64).collect();
    let a = u8x32::from_slice(simd, &a);
    let b = u8x32::from_slice(simd, &b);
    let expected_0: Vec<u8> = (1..=32).collect();
    let expected_16: Vec<u8> = (17..=48).collect();
    let expected_32: Vec<u8> = (33..=64).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
    assert_eq!(a.slide::<32>(b).as_slice(), &expected_32);
}

#[simd_test]
fn slide_u8x64<S: Simd>(simd: S) {
    let a: Vec<u8> = (1..=64).collect();
    let b: Vec<u8> = (65..=128).collect();
    let a = u8x64::from_slice(simd, &a);
    let b = u8x64::from_slice(simd, &b);
    let expected_0: Vec<u8> = (1..=64).collect();
    let expected_32: Vec<u8> = (33..=96).collect();
    let expected_64: Vec<u8> = (65..=128).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<32>(b).as_slice(), &expected_32);
    assert_eq!(a.slide::<64>(b).as_slice(), &expected_64);
}

#[simd_test]
fn slide_i16x8<S: Simd>(simd: S) {
    let a: Vec<i16> = (1_i16..=8).collect();
    let b: Vec<i16> = (9_i16..=16).collect();
    let a = i16x8::from_slice(simd, &a);
    let b = i16x8::from_slice(simd, &b);
    let expected_0: Vec<i16> = (1_i16..=8).collect();
    let expected_1: Vec<i16> = (2_i16..=9).collect();
    let expected_4: Vec<i16> = (5_i16..=12).collect();
    let expected_7: Vec<i16> = (8_i16..=15).collect();
    let expected_8: Vec<i16> = (9_i16..=16).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide::<4>(b).as_slice(), &expected_4);
    assert_eq!(a.slide::<7>(b).as_slice(), &expected_7);
    assert_eq!(a.slide::<8>(b).as_slice(), &expected_8);
}

#[simd_test]
fn slide_i16x16<S: Simd>(simd: S) {
    let a: Vec<i16> = (1_i16..=16).collect();
    let b: Vec<i16> = (17_i16..=32).collect();
    let a = i16x16::from_slice(simd, &a);
    let b = i16x16::from_slice(simd, &b);
    let expected_0: Vec<i16> = (1_i16..=16).collect();
    let expected_8: Vec<i16> = (9_i16..=24).collect();
    let expected_16: Vec<i16> = (17_i16..=32).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<8>(b).as_slice(), &expected_8);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
}

#[simd_test]
fn slide_i16x32<S: Simd>(simd: S) {
    let a: Vec<i16> = (1_i16..=32).collect();
    let b: Vec<i16> = (33_i16..=64).collect();
    let a = i16x32::from_slice(simd, &a);
    let b = i16x32::from_slice(simd, &b);
    let expected_0: Vec<i16> = (1_i16..=32).collect();
    let expected_16: Vec<i16> = (17_i16..=48).collect();
    let expected_32: Vec<i16> = (33_i16..=64).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
    assert_eq!(a.slide::<32>(b).as_slice(), &expected_32);
}

#[simd_test]
fn slide_u16x8<S: Simd>(simd: S) {
    let a: Vec<u16> = (1..=8).collect();
    let b: Vec<u16> = (9..=16).collect();
    let a = u16x8::from_slice(simd, &a);
    let b = u16x8::from_slice(simd, &b);
    let expected_0: Vec<u16> = (1..=8).collect();
    let expected_4: Vec<u16> = (5..=12).collect();
    let expected_8: Vec<u16> = (9..=16).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<4>(b).as_slice(), &expected_4);
    assert_eq!(a.slide::<8>(b).as_slice(), &expected_8);
}

#[simd_test]
fn slide_u16x16<S: Simd>(simd: S) {
    let a: Vec<u16> = (1..=16).collect();
    let b: Vec<u16> = (17..=32).collect();
    let a = u16x16::from_slice(simd, &a);
    let b = u16x16::from_slice(simd, &b);
    let expected_0: Vec<u16> = (1..=16).collect();
    let expected_8: Vec<u16> = (9..=24).collect();
    let expected_16: Vec<u16> = (17..=32).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<8>(b).as_slice(), &expected_8);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
}

#[simd_test]
fn slide_u16x32<S: Simd>(simd: S) {
    let a: Vec<u16> = (1..=32).collect();
    let b: Vec<u16> = (33..=64).collect();
    let a = u16x32::from_slice(simd, &a);
    let b = u16x32::from_slice(simd, &b);
    let expected_0: Vec<u16> = (1..=32).collect();
    let expected_16: Vec<u16> = (17..=48).collect();
    let expected_32: Vec<u16> = (33..=64).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
    assert_eq!(a.slide::<32>(b).as_slice(), &expected_32);
}

#[simd_test]
fn slide_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.slide::<0>(b), [1, 2, 3, 4]);
    assert_eq!(*a.slide::<1>(b), [2, 3, 4, 5]);
    assert_eq!(*a.slide::<2>(b), [3, 4, 5, 6]);
    assert_eq!(*a.slide::<4>(b), [5, 6, 7, 8]);
}

#[simd_test]
fn slide_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*a.slide::<0>(b), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*a.slide::<1>(b), [2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(*a.slide::<4>(b), [5, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(*a.slide::<7>(b), [8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(*a.slide::<8>(b), [9, 10, 11, 12, 13, 14, 15, 16]);
}

#[simd_test]
fn slide_i32x16<S: Simd>(simd: S) {
    let a: Vec<i32> = (1..=16).collect();
    let b: Vec<i32> = (17..=32).collect();
    let a = i32x16::from_slice(simd, &a);
    let b = i32x16::from_slice(simd, &b);
    let expected_0: Vec<i32> = (1..=16).collect();
    let expected_8: Vec<i32> = (9..=24).collect();
    let expected_16: Vec<i32> = (17..=32).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<8>(b).as_slice(), &expected_8);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
}

#[simd_test]
fn slide_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.slide::<0>(b), [1, 2, 3, 4]);
    assert_eq!(*a.slide::<2>(b), [3, 4, 5, 6]);
    assert_eq!(*a.slide::<4>(b), [5, 6, 7, 8]);
}

#[simd_test]
fn slide_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*a.slide::<0>(b), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*a.slide::<4>(b), [5, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(*a.slide::<8>(b), [9, 10, 11, 12, 13, 14, 15, 16]);
}

#[simd_test]
fn slide_u32x16<S: Simd>(simd: S) {
    let a: Vec<u32> = (1..=16).collect();
    let b: Vec<u32> = (17..=32).collect();
    let a = u32x16::from_slice(simd, &a);
    let b = u32x16::from_slice(simd, &b);
    let expected_0: Vec<u32> = (1..=16).collect();
    let expected_8: Vec<u32> = (9..=24).collect();
    let expected_16: Vec<u32> = (17..=32).collect();
    assert_eq!(a.slide::<0>(b).as_slice(), &expected_0);
    assert_eq!(a.slide::<8>(b).as_slice(), &expected_8);
    assert_eq!(a.slide::<16>(b).as_slice(), &expected_16);
}

#[simd_test]
fn slide_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1, 2]);
    let b = i64x2::from_slice(simd, &[3, 4]);
    assert_eq!(*a.slide::<1>(b), [2, 3]);
}

#[simd_test]
fn slide_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i64x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.slide::<1>(b), [2, 3, 4, 5]);
}

#[simd_test]
fn slide_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*a.slide::<1>(b), [2, 3, 4, 5, 6, 7, 8, 9]);
}

#[simd_test]
fn slide_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1, 2]);
    let b = u64x2::from_slice(simd, &[3, 4]);
    assert_eq!(*a.slide::<1>(b), [2, 3]);
}

#[simd_test]
fn slide_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u64x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.slide::<1>(b), [2, 3, 4, 5]);
}

#[simd_test]
fn slide_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*a.slide::<1>(b), [2, 3, 4, 5, 6, 7, 8, 9]);
}
