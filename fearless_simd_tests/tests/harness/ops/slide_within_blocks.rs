// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn slide_within_blocks_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*a.slide_within_blocks::<0>(b), *a);
    assert_eq!(*a.slide_within_blocks::<1>(b), *a.slide::<1>(b));
    assert_eq!(*a.slide_within_blocks::<4>(b), *b);
}

#[simd_test]
fn slide_within_blocks_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::from_slice(simd, &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    assert_eq!(*a.slide_within_blocks::<0>(b), *a);
    assert_eq!(
        *a.slide_within_blocks::<1>(b),
        [2.0, 3.0, 4.0, 9.0, 6.0, 7.0, 8.0, 13.0]
    );
    assert_eq!(*a.slide_within_blocks::<4>(b), *b);
}

#[simd_test]
fn slide_within_blocks_f32x16<S: Simd>(simd: S) {
    let a: Vec<f32> = (1_i16..=16).map(f32::from).collect();
    let b: Vec<f32> = (17_i16..=32).map(f32::from).collect();
    let a = f32x16::from_slice(simd, &a);
    let b = f32x16::from_slice(simd, &b);

    assert_eq!(*a.slide_within_blocks::<0>(b), *a);

    let expected_1: [f32; 16] = [
        2.0, 3.0, 4.0, 17.0, 6.0, 7.0, 8.0, 21.0, 10.0, 11.0, 12.0, 25.0, 14.0, 15.0, 16.0, 29.0,
    ];
    assert_eq!(*a.slide_within_blocks::<1>(b), expected_1);

    assert_eq!(*a.slide_within_blocks::<4>(b), *b);
}

#[simd_test]
fn slide_within_blocks_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.0, 2.0]);
    let b = f64x2::from_slice(simd, &[3.0, 4.0]);
    assert_eq!(*a.slide_within_blocks::<0>(b), *a);
    assert_eq!(*a.slide_within_blocks::<1>(b), *a.slide::<1>(b));
    assert_eq!(*a.slide_within_blocks::<2>(b), *b);
}

#[simd_test]
fn slide_within_blocks_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*a.slide_within_blocks::<0>(b), *a);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2.0, 5.0, 4.0, 7.0]);
    assert_eq!(*a.slide_within_blocks::<2>(b), *b);
}

#[simd_test]
fn slide_within_blocks_f64x8<S: Simd>(simd: S) {
    let a: Vec<f64> = (1..=8).map(f64::from).collect();
    let b: Vec<f64> = (9..=16).map(f64::from).collect();
    let a = f64x8::from_slice(simd, &a);
    let b = f64x8::from_slice(simd, &b);
    assert_eq!(*a.slide_within_blocks::<0>(b), *a);
    assert_eq!(
        *a.slide_within_blocks::<1>(b),
        [2.0, 9.0, 4.0, 11.0, 6.0, 13.0, 8.0, 15.0]
    );
    assert_eq!(*a.slide_within_blocks::<2>(b), *b);
}

#[simd_test]
fn slide_within_blocks_i8x16<S: Simd>(simd: S) {
    let a: Vec<i8> = (1_i8..=16).collect();
    let b: Vec<i8> = (17_i8..=32).collect();
    let a = i8x16::from_slice(simd, &a);
    let b = i8x16::from_slice(simd, &b);
    assert_eq!(*a.slide_within_blocks::<0>(b), *a);
    assert_eq!(*a.slide_within_blocks::<1>(b), *a.slide::<1>(b));
    assert_eq!(*a.slide_within_blocks::<16>(b), *b);
}

#[simd_test]
fn slide_within_blocks_i8x32<S: Simd>(simd: S) {
    let a: Vec<i8> = (1_i8..=32).collect();
    let b: Vec<i8> = (33_i8..=64).collect();
    let a = i8x32::from_slice(simd, &a);
    let b = i8x32::from_slice(simd, &b);

    assert_eq!(*a.slide_within_blocks::<0>(b), *a);

    let expected_1: [i8; 32] = [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 33, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 49,
    ];
    assert_eq!(*a.slide_within_blocks::<1>(b), expected_1);

    assert_eq!(*a.slide_within_blocks::<16>(b), *b);
}

#[simd_test]
fn slide_within_blocks_i8x64<S: Simd>(simd: S) {
    let a: Vec<i8> = (0_i8..=63).collect();
    let b: Vec<i8> = (64_i8..=127).collect();
    let a = i8x64::from_slice(simd, &a);
    let b = i8x64::from_slice(simd, &b);

    assert_eq!(*a.slide_within_blocks::<0>(b), *a);

    let expected_1: [i8; 64] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 64, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 80, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 96,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 112,
    ];
    assert_eq!(*a.slide_within_blocks::<1>(b), expected_1);

    assert_eq!(*a.slide_within_blocks::<16>(b), *b);
}

#[simd_test]
fn slide_within_blocks_u8x16<S: Simd>(simd: S) {
    let a: Vec<u8> = (1..=16).collect();
    let b: Vec<u8> = (17..=32).collect();
    let a = u8x16::from_slice(simd, &a);
    let b = u8x16::from_slice(simd, &b);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    assert_eq!(
        a.slide_within_blocks::<1>(b).as_slice(),
        a.slide::<1>(b).as_slice()
    );
    assert_eq!(a.slide_within_blocks::<16>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_u8x32<S: Simd>(simd: S) {
    let a: Vec<u8> = (1..=32).collect();
    let b: Vec<u8> = (33..=64).collect();
    let a = u8x32::from_slice(simd, &a);
    let b = u8x32::from_slice(simd, &b);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    let expected_1: [u8; 32] = [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 33, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 49,
    ];
    assert_eq!(a.slide_within_blocks::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide_within_blocks::<16>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_u8x64<S: Simd>(simd: S) {
    let a: Vec<u8> = (1..=64).collect();
    let b: Vec<u8> = (65..=128).collect();
    let a = u8x64::from_slice(simd, &a);
    let b = u8x64::from_slice(simd, &b);

    assert_eq!(*a.slide_within_blocks::<0>(b), *a);

    let expected_1: [u8; 64] = [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 65, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 81, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 97,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 113,
    ];
    assert_eq!(*a.slide_within_blocks::<1>(b), expected_1);

    assert_eq!(*a.slide_within_blocks::<16>(b), *b);
}

#[simd_test]
fn slide_within_blocks_i16x8<S: Simd>(simd: S) {
    let a: Vec<i16> = (1_i16..=8).collect();
    let b: Vec<i16> = (9_i16..=16).collect();
    let a = i16x8::from_slice(simd, &a);
    let b = i16x8::from_slice(simd, &b);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    assert_eq!(
        a.slide_within_blocks::<1>(b).as_slice(),
        a.slide::<1>(b).as_slice()
    );
    assert_eq!(a.slide_within_blocks::<8>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_i16x16<S: Simd>(simd: S) {
    let a: Vec<i16> = (1_i16..=16).collect();
    let b: Vec<i16> = (17_i16..=32).collect();
    let a = i16x16::from_slice(simd, &a);
    let b = i16x16::from_slice(simd, &b);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    let expected_1: [i16; 16] = [2, 3, 4, 5, 6, 7, 8, 17, 10, 11, 12, 13, 14, 15, 16, 25];
    assert_eq!(a.slide_within_blocks::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide_within_blocks::<8>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_i16x32<S: Simd>(simd: S) {
    let a: Vec<i16> = (1_i16..=32).collect();
    let b: Vec<i16> = (33_i16..=64).collect();
    let a = i16x32::from_slice(simd, &a);
    let b = i16x32::from_slice(simd, &b);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    let expected_1: [i16; 32] = [
        2, 3, 4, 5, 6, 7, 8, 33, 10, 11, 12, 13, 14, 15, 16, 41, 18, 19, 20, 21, 22, 23, 24, 49,
        26, 27, 28, 29, 30, 31, 32, 57,
    ];
    assert_eq!(a.slide_within_blocks::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide_within_blocks::<8>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_u16x8<S: Simd>(simd: S) {
    let a: Vec<u16> = (1..=8).collect();
    let b: Vec<u16> = (9..=16).collect();
    let a = u16x8::from_slice(simd, &a);
    let b = u16x8::from_slice(simd, &b);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    assert_eq!(
        a.slide_within_blocks::<1>(b).as_slice(),
        a.slide::<1>(b).as_slice()
    );
    assert_eq!(a.slide_within_blocks::<8>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_u16x16<S: Simd>(simd: S) {
    let a: Vec<u16> = (1..=16).collect();
    let b: Vec<u16> = (17..=32).collect();
    let a = u16x16::from_slice(simd, &a);
    let b = u16x16::from_slice(simd, &b);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    let expected_1: [u16; 16] = [2, 3, 4, 5, 6, 7, 8, 17, 10, 11, 12, 13, 14, 15, 16, 25];
    assert_eq!(a.slide_within_blocks::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide_within_blocks::<8>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_u16x32<S: Simd>(simd: S) {
    let a: Vec<u16> = (1..=32).collect();
    let b: Vec<u16> = (33..=64).collect();
    let a = u16x32::from_slice(simd, &a);
    let b = u16x32::from_slice(simd, &b);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    let expected_1: [u16; 32] = [
        2, 3, 4, 5, 6, 7, 8, 33, 10, 11, 12, 13, 14, 15, 16, 41, 18, 19, 20, 21, 22, 23, 24, 49,
        26, 27, 28, 29, 30, 31, 32, 57,
    ];
    assert_eq!(a.slide_within_blocks::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide_within_blocks::<8>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    assert_eq!(
        a.slide_within_blocks::<1>(b).as_slice(),
        a.slide::<1>(b).as_slice()
    );
    assert_eq!(a.slide_within_blocks::<4>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    assert_eq!(
        a.slide_within_blocks::<1>(b).as_slice(),
        &[2, 3, 4, 9, 6, 7, 8, 13]
    );
    assert_eq!(a.slide_within_blocks::<4>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_i32x16<S: Simd>(simd: S) {
    let a: Vec<i32> = (1..=16).collect();
    let b: Vec<i32> = (17..=32).collect();
    let a = i32x16::from_slice(simd, &a);
    let b = i32x16::from_slice(simd, &b);

    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());

    let expected_1: [i32; 16] = [2, 3, 4, 17, 6, 7, 8, 21, 10, 11, 12, 25, 14, 15, 16, 29];
    assert_eq!(a.slide_within_blocks::<1>(b).as_slice(), &expected_1);

    assert_eq!(a.slide_within_blocks::<4>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    assert_eq!(
        a.slide_within_blocks::<1>(b).as_slice(),
        a.slide::<1>(b).as_slice()
    );
    assert_eq!(a.slide_within_blocks::<4>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    assert_eq!(
        a.slide_within_blocks::<1>(b).as_slice(),
        &[2, 3, 4, 9, 6, 7, 8, 13]
    );
    assert_eq!(a.slide_within_blocks::<4>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_u32x16<S: Simd>(simd: S) {
    let a: Vec<u32> = (1..=16).collect();
    let b: Vec<u32> = (17..=32).collect();
    let a = u32x16::from_slice(simd, &a);
    let b = u32x16::from_slice(simd, &b);
    assert_eq!(a.slide_within_blocks::<0>(b).as_slice(), a.as_slice());
    let expected_1: [u32; 16] = [2, 3, 4, 17, 6, 7, 8, 21, 10, 11, 12, 25, 14, 15, 16, 29];
    assert_eq!(a.slide_within_blocks::<1>(b).as_slice(), &expected_1);
    assert_eq!(a.slide_within_blocks::<4>(b).as_slice(), b.as_slice());
}

#[simd_test]
fn slide_within_blocks_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1, 2]);
    let b = i64x2::from_slice(simd, &[3, 4]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 3]);
    assert_eq!(*a.slide_within_blocks::<2>(b), [3, 4]);
}

#[simd_test]
fn slide_within_blocks_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i64x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2, 3, 4]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 5, 4, 7]);
    assert_eq!(*a.slide_within_blocks::<2>(b), [5, 6, 7, 8]);
}

#[simd_test]
fn slide_within_blocks_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 9, 4, 11, 6, 13, 8, 15]);
    assert_eq!(
        *a.slide_within_blocks::<2>(b),
        [9, 10, 11, 12, 13, 14, 15, 16]
    );
}

#[simd_test]
fn slide_within_blocks_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1, 2]);
    let b = u64x2::from_slice(simd, &[3, 4]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 3]);
    assert_eq!(*a.slide_within_blocks::<2>(b), [3, 4]);
}

#[simd_test]
fn slide_within_blocks_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u64x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2, 3, 4]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 5, 4, 7]);
    assert_eq!(*a.slide_within_blocks::<2>(b), [5, 6, 7, 8]);
}

#[simd_test]
fn slide_within_blocks_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u64x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*a.slide_within_blocks::<0>(b), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*a.slide_within_blocks::<1>(b), [2, 9, 4, 11, 6, 13, 8, 15]);
    assert_eq!(
        *a.slide_within_blocks::<2>(b),
        [9, 10, 11, 12, 13, 14, 15, 16]
    );
}
