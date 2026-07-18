// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn rotate_elements_left_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        *simd.rotate_elements_left_f32x4::<1>(a),
        [2.0, 3.0, 4.0, 1.0]
    );
}

#[simd_test]
fn rotate_elements_left_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            1_i8, 2_i8, 3_i8, 4_i8, 5_i8, 6_i8, 7_i8, 8_i8, 9_i8, 10_i8, 11_i8, 12_i8, 13_i8,
            14_i8, 15_i8, 16_i8,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_i8x16::<1>(a),
        [
            2_i8, 3_i8, 4_i8, 5_i8, 6_i8, 7_i8, 8_i8, 9_i8, 10_i8, 11_i8, 12_i8, 13_i8, 14_i8,
            15_i8, 16_i8, 1_i8
        ]
    );
}

#[simd_test]
fn rotate_elements_left_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[
            1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8, 9_u8, 10_u8, 11_u8, 12_u8, 13_u8,
            14_u8, 15_u8, 16_u8,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_u8x16::<1>(a),
        [
            2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8, 9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8,
            15_u8, 16_u8, 1_u8
        ]
    );
}

#[simd_test]
fn rotate_elements_left_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(
        simd,
        &[1_i16, 2_i16, 3_i16, 4_i16, 5_i16, 6_i16, 7_i16, 8_i16],
    );
    assert_eq!(
        *simd.rotate_elements_left_i16x8::<1>(a),
        [2_i16, 3_i16, 4_i16, 5_i16, 6_i16, 7_i16, 8_i16, 1_i16]
    );
}

#[simd_test]
fn rotate_elements_left_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(
        simd,
        &[1_u16, 2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16],
    );
    assert_eq!(
        *simd.rotate_elements_left_u16x8::<1>(a),
        [2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16, 1_u16]
    );
}

#[simd_test]
fn rotate_elements_left_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1_i32, 2_i32, 3_i32, 4_i32]);
    assert_eq!(
        *simd.rotate_elements_left_i32x4::<1>(a),
        [2_i32, 3_i32, 4_i32, 1_i32]
    );
}

#[simd_test]
fn rotate_elements_left_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1_u32, 2_u32, 3_u32, 4_u32]);
    assert_eq!(
        *simd.rotate_elements_left_u32x4::<1>(a),
        [2_u32, 3_u32, 4_u32, 1_u32]
    );
}

#[simd_test]
fn rotate_elements_left_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.0, 2.0]);
    assert_eq!(*simd.rotate_elements_left_f64x2::<1>(a), [2.0, 1.0]);
}

#[simd_test]
fn rotate_elements_left_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(
        *simd.rotate_elements_left_f32x8::<1>(a),
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0]
    );
}

#[simd_test]
fn rotate_elements_left_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1_i8, 2_i8, 3_i8, 4_i8, 5_i8, 6_i8, 7_i8, 8_i8, 9_i8, 10_i8, 11_i8, 12_i8, 13_i8,
            14_i8, 15_i8, 16_i8, 17_i8, 18_i8, 19_i8, 20_i8, 21_i8, 22_i8, 23_i8, 24_i8, 25_i8,
            26_i8, 27_i8, 28_i8, 29_i8, 30_i8, 31_i8, 32_i8,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_i8x32::<1>(a),
        [
            2_i8, 3_i8, 4_i8, 5_i8, 6_i8, 7_i8, 8_i8, 9_i8, 10_i8, 11_i8, 12_i8, 13_i8, 14_i8,
            15_i8, 16_i8, 17_i8, 18_i8, 19_i8, 20_i8, 21_i8, 22_i8, 23_i8, 24_i8, 25_i8, 26_i8,
            27_i8, 28_i8, 29_i8, 30_i8, 31_i8, 32_i8, 1_i8
        ]
    );
}

#[simd_test]
fn rotate_elements_left_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8, 9_u8, 10_u8, 11_u8, 12_u8, 13_u8,
            14_u8, 15_u8, 16_u8, 17_u8, 18_u8, 19_u8, 20_u8, 21_u8, 22_u8, 23_u8, 24_u8, 25_u8,
            26_u8, 27_u8, 28_u8, 29_u8, 30_u8, 31_u8, 32_u8,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_u8x32::<1>(a),
        [
            2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8, 9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8,
            15_u8, 16_u8, 17_u8, 18_u8, 19_u8, 20_u8, 21_u8, 22_u8, 23_u8, 24_u8, 25_u8, 26_u8,
            27_u8, 28_u8, 29_u8, 30_u8, 31_u8, 32_u8, 1_u8
        ]
    );
}

#[simd_test]
fn rotate_elements_left_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            1_i16, 2_i16, 3_i16, 4_i16, 5_i16, 6_i16, 7_i16, 8_i16, 9_i16, 10_i16, 11_i16, 12_i16,
            13_i16, 14_i16, 15_i16, 16_i16,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_i16x16::<1>(a),
        [
            2_i16, 3_i16, 4_i16, 5_i16, 6_i16, 7_i16, 8_i16, 9_i16, 10_i16, 11_i16, 12_i16, 13_i16,
            14_i16, 15_i16, 16_i16, 1_i16
        ]
    );
}

#[simd_test]
fn rotate_elements_left_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[
            1_u16, 2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16, 9_u16, 10_u16, 11_u16, 12_u16,
            13_u16, 14_u16, 15_u16, 16_u16,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_u16x16::<1>(a),
        [
            2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16, 9_u16, 10_u16, 11_u16, 12_u16, 13_u16,
            14_u16, 15_u16, 16_u16, 1_u16
        ]
    );
}

#[simd_test]
fn rotate_elements_left_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(
        simd,
        &[1_i32, 2_i32, 3_i32, 4_i32, 5_i32, 6_i32, 7_i32, 8_i32],
    );
    assert_eq!(
        *simd.rotate_elements_left_i32x8::<1>(a),
        [2_i32, 3_i32, 4_i32, 5_i32, 6_i32, 7_i32, 8_i32, 1_i32]
    );
}

#[simd_test]
fn rotate_elements_left_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(
        simd,
        &[1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32],
    );
    assert_eq!(
        *simd.rotate_elements_left_u32x8::<1>(a),
        [2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32, 1_u32]
    );
}

#[simd_test]
fn rotate_elements_left_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        *simd.rotate_elements_left_f64x4::<1>(a),
        [2.0, 3.0, 4.0, 1.0]
    );
}

#[simd_test]
fn rotate_elements_left_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_f32x16::<1>(a),
        [
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1.0
        ]
    );
}

#[simd_test]
fn rotate_elements_left_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            1_i8, 2_i8, 3_i8, 4_i8, 5_i8, 6_i8, 7_i8, 8_i8, 9_i8, 10_i8, 11_i8, 12_i8, 13_i8,
            14_i8, 15_i8, 16_i8, 17_i8, 18_i8, 19_i8, 20_i8, 21_i8, 22_i8, 23_i8, 24_i8, 25_i8,
            26_i8, 27_i8, 28_i8, 29_i8, 30_i8, 31_i8, 32_i8, 33_i8, 34_i8, 35_i8, 36_i8, 37_i8,
            38_i8, 39_i8, 40_i8, 41_i8, 42_i8, 43_i8, 44_i8, 45_i8, 46_i8, 47_i8, 48_i8, 49_i8,
            50_i8, 51_i8, 52_i8, 53_i8, 54_i8, 55_i8, 56_i8, 57_i8, 58_i8, 59_i8, 60_i8, 61_i8,
            62_i8, 63_i8, 64_i8,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_i8x64::<1>(a),
        [
            2_i8, 3_i8, 4_i8, 5_i8, 6_i8, 7_i8, 8_i8, 9_i8, 10_i8, 11_i8, 12_i8, 13_i8, 14_i8,
            15_i8, 16_i8, 17_i8, 18_i8, 19_i8, 20_i8, 21_i8, 22_i8, 23_i8, 24_i8, 25_i8, 26_i8,
            27_i8, 28_i8, 29_i8, 30_i8, 31_i8, 32_i8, 33_i8, 34_i8, 35_i8, 36_i8, 37_i8, 38_i8,
            39_i8, 40_i8, 41_i8, 42_i8, 43_i8, 44_i8, 45_i8, 46_i8, 47_i8, 48_i8, 49_i8, 50_i8,
            51_i8, 52_i8, 53_i8, 54_i8, 55_i8, 56_i8, 57_i8, 58_i8, 59_i8, 60_i8, 61_i8, 62_i8,
            63_i8, 64_i8, 1_i8
        ]
    );
}

#[simd_test]
fn rotate_elements_left_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            1_u8, 2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8, 9_u8, 10_u8, 11_u8, 12_u8, 13_u8,
            14_u8, 15_u8, 16_u8, 17_u8, 18_u8, 19_u8, 20_u8, 21_u8, 22_u8, 23_u8, 24_u8, 25_u8,
            26_u8, 27_u8, 28_u8, 29_u8, 30_u8, 31_u8, 32_u8, 33_u8, 34_u8, 35_u8, 36_u8, 37_u8,
            38_u8, 39_u8, 40_u8, 41_u8, 42_u8, 43_u8, 44_u8, 45_u8, 46_u8, 47_u8, 48_u8, 49_u8,
            50_u8, 51_u8, 52_u8, 53_u8, 54_u8, 55_u8, 56_u8, 57_u8, 58_u8, 59_u8, 60_u8, 61_u8,
            62_u8, 63_u8, 64_u8,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_u8x64::<1>(a),
        [
            2_u8, 3_u8, 4_u8, 5_u8, 6_u8, 7_u8, 8_u8, 9_u8, 10_u8, 11_u8, 12_u8, 13_u8, 14_u8,
            15_u8, 16_u8, 17_u8, 18_u8, 19_u8, 20_u8, 21_u8, 22_u8, 23_u8, 24_u8, 25_u8, 26_u8,
            27_u8, 28_u8, 29_u8, 30_u8, 31_u8, 32_u8, 33_u8, 34_u8, 35_u8, 36_u8, 37_u8, 38_u8,
            39_u8, 40_u8, 41_u8, 42_u8, 43_u8, 44_u8, 45_u8, 46_u8, 47_u8, 48_u8, 49_u8, 50_u8,
            51_u8, 52_u8, 53_u8, 54_u8, 55_u8, 56_u8, 57_u8, 58_u8, 59_u8, 60_u8, 61_u8, 62_u8,
            63_u8, 64_u8, 1_u8
        ]
    );
}

#[simd_test]
fn rotate_elements_left_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1_i16, 2_i16, 3_i16, 4_i16, 5_i16, 6_i16, 7_i16, 8_i16, 9_i16, 10_i16, 11_i16, 12_i16,
            13_i16, 14_i16, 15_i16, 16_i16, 17_i16, 18_i16, 19_i16, 20_i16, 21_i16, 22_i16, 23_i16,
            24_i16, 25_i16, 26_i16, 27_i16, 28_i16, 29_i16, 30_i16, 31_i16, 32_i16,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_i16x32::<1>(a),
        [
            2_i16, 3_i16, 4_i16, 5_i16, 6_i16, 7_i16, 8_i16, 9_i16, 10_i16, 11_i16, 12_i16, 13_i16,
            14_i16, 15_i16, 16_i16, 17_i16, 18_i16, 19_i16, 20_i16, 21_i16, 22_i16, 23_i16, 24_i16,
            25_i16, 26_i16, 27_i16, 28_i16, 29_i16, 30_i16, 31_i16, 32_i16, 1_i16
        ]
    );
}

#[simd_test]
fn rotate_elements_left_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            1_u16, 2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16, 9_u16, 10_u16, 11_u16, 12_u16,
            13_u16, 14_u16, 15_u16, 16_u16, 17_u16, 18_u16, 19_u16, 20_u16, 21_u16, 22_u16, 23_u16,
            24_u16, 25_u16, 26_u16, 27_u16, 28_u16, 29_u16, 30_u16, 31_u16, 32_u16,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_u16x32::<1>(a),
        [
            2_u16, 3_u16, 4_u16, 5_u16, 6_u16, 7_u16, 8_u16, 9_u16, 10_u16, 11_u16, 12_u16, 13_u16,
            14_u16, 15_u16, 16_u16, 17_u16, 18_u16, 19_u16, 20_u16, 21_u16, 22_u16, 23_u16, 24_u16,
            25_u16, 26_u16, 27_u16, 28_u16, 29_u16, 30_u16, 31_u16, 32_u16, 1_u16
        ]
    );
}

#[simd_test]
fn rotate_elements_left_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[
            1_i32, 2_i32, 3_i32, 4_i32, 5_i32, 6_i32, 7_i32, 8_i32, 9_i32, 10_i32, 11_i32, 12_i32,
            13_i32, 14_i32, 15_i32, 16_i32,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_i32x16::<1>(a),
        [
            2_i32, 3_i32, 4_i32, 5_i32, 6_i32, 7_i32, 8_i32, 9_i32, 10_i32, 11_i32, 12_i32, 13_i32,
            14_i32, 15_i32, 16_i32, 1_i32
        ]
    );
}

#[simd_test]
fn rotate_elements_left_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32, 9_u32, 10_u32, 11_u32, 12_u32,
            13_u32, 14_u32, 15_u32, 16_u32,
        ],
    );
    assert_eq!(
        *simd.rotate_elements_left_u32x16::<1>(a),
        [
            2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32, 9_u32, 10_u32, 11_u32, 12_u32, 13_u32,
            14_u32, 15_u32, 16_u32, 1_u32
        ]
    );
}

#[simd_test]
fn rotate_elements_left_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(
        *simd.rotate_elements_left_f64x8::<1>(a),
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0]
    );
}
