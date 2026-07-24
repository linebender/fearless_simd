// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn simd_gt_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 3.0, 2.0, 1.0]);
    let b = f32x4::from_slice(simd, &[1.0, 2.0, 2.0, 4.0]);
    assert_eq!(<[i32; 4]>::from(a.simd_gt(b)), [-1, -1, 0, 0]);
}

#[simd_test]
fn simd_gt_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[2, 2, 2, 5, 0, 0, 0, 0, 5, 25, 25, 45, 45, 65, 65, 85],
    );
    let b = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, -1, -2, -3, -4, 10, 20, 30, 40, 50, 60, 70, 80],
    );
    assert_eq!(
        <[i8; 16]>::from(a.simd_gt(b)),
        [-1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1]
    );
}

#[simd_test]
fn simd_gt_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[2, 2, 2, 5, 0, 0, 0, 0]);
    let b = i16x8::from_slice(simd, &[1, 2, 3, 4, -1, -2, -3, -4]);
    assert_eq!(
        <[i16; 8]>::from(a.simd_gt(b)),
        [-1, 0, 0, -1, -1, -1, -1, -1]
    );
}

#[simd_test]
fn simd_gt_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[2, 2, 2, 5, 40000, 150, 50000, 350]);
    let b = u16x8::from_slice(simd, &[1, 2, 3, 4, 100, 200, 300, 400]);
    assert_eq!(<[i16; 8]>::from(a.simd_gt(b)), [-1, 0, 0, -1, -1, 0, -1, 0]);
}

#[simd_test]
fn simd_gt_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[2, 2, 0, 0]);
    let b = i32x4::from_slice(simd, &[1, 2, -3, -4]);
    assert_eq!(<[i32; 4]>::from(a.simd_gt(b)), [-1, 0, -1, -1]);
}

#[simd_test]
fn simd_gt_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[2, 2, 3000000000, 150]);
    let b = u32x4::from_slice(simd, &[1, 2, 100, 200]);
    assert_eq!(<[i32; 4]>::from(a.simd_gt(b)), [-1, 0, -1, 0]);
}

#[simd_test]
fn simd_gt_u8x16<S: Simd>(simd: S) {
    let vals = u8x16::from_slice(
        simd,
        &[
            0, 12, 34, 50, 220, 180, 127, 128, 255, 50, 33, 126, 0, 0, 0, 0,
        ],
    );
    let mask = vals.simd_gt(u8x16::splat(simd, 128));

    assert_eq!(
        <[i8; 16]>::from(mask),
        [0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]
    );
}

#[simd_test]
fn simd_gt_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::from_slice(simd, &[1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 8.0, 7.0]);
    assert_eq!(<[i32; 8]>::from(a.simd_gt(b)), [-1, -1, 0, 0, 0, 0, 0, -1]);
}

#[simd_test]
fn simd_gt_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            2, 2, 2, 5, 0, 0, 0, 0, 5, 25, 25, 45, 45, 65, 65, 85, 2, 2, 2, 5, 0, 0, 0, 0, 5, 25,
            25, 45, 45, 65, 65, 85,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, -1, -2, -3, -4, 10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, -1, -2, -3, -4,
            10, 20, 30, 40, 50, 60, 70, 80,
        ],
    );
    assert_eq!(
        <[i8; 32]>::from(a.simd_gt(b)),
        [
            -1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1,
            0, -1, 0, -1, 0, -1, 0, -1
        ]
    );
}

#[simd_test]
fn simd_gt_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            2, 2, 2, 5, 4, 7, 6, 9, 5, 25, 25, 45, 45, 65, 65, 85, 2, 2, 2, 5, 4, 7, 6, 9, 5, 25,
            25, 45, 45, 65, 65, 85,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20,
            30, 40, 50, 60, 70, 80,
        ],
    );
    assert_eq!(
        <[i8; 32]>::from(a.simd_gt(b)),
        [
            -1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1
        ]
    );
}

#[simd_test]
fn simd_gt_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[2, 2, 2, 5, 0, 0, 0, 0, 5, 25, 25, 45, 45, 65, 65, 85],
    );
    let b = i16x16::from_slice(
        simd,
        &[1, 2, 3, 4, -1, -2, -3, -4, 10, 20, 30, 40, 50, 60, 70, 80],
    );
    assert_eq!(
        <[i16; 16]>::from(a.simd_gt(b)),
        [-1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1]
    );
}

#[simd_test]
fn simd_gt_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[2, 2, 2, 5, 4, 7, 6, 9, 5, 25, 25, 45, 45, 65, 65, 85],
    );
    let b = u16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80],
    );
    assert_eq!(
        <[i16; 16]>::from(a.simd_gt(b)),
        [-1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]
    );
}

#[simd_test]
fn simd_gt_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[2, 2, 2, 5, 0, 0, 0, 0]);
    let b = i32x8::from_slice(simd, &[1, 2, 3, 4, -1, -2, -3, -4]);
    assert_eq!(
        <[i32; 8]>::from(a.simd_gt(b)),
        [-1, 0, 0, -1, -1, -1, -1, -1]
    );
}

#[simd_test]
fn simd_gt_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[2, 2, 2, 5, 4, 7, 6, 9]);
    let b = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(<[i32; 8]>::from(a.simd_gt(b)), [-1, 0, 0, -1, 0, -1, 0, -1]);
}

#[simd_test]
fn simd_gt_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0, 4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 8.0, 7.0, 1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 8.0, 7.0,
        ],
    );
    assert_eq!(
        <[i32; 16]>::from(a.simd_gt(b)),
        [-1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1]
    );
}

#[simd_test]
fn simd_gt_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            2, 2, 2, 5, 0, 0, 0, 0, 5, 25, 25, 45, 45, 65, 65, 85, 2, 2, 2, 5, 0, 0, 0, 0, 5, 25,
            25, 45, 45, 65, 65, 85, 2, 2, 2, 5, 0, 0, 0, 0, 5, 25, 25, 45, 45, 65, 65, 85, 2, 2, 2,
            5, 0, 0, 0, 0, 5, 25, 25, 45, 45, 65, 65, 85,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, -1, -2, -3, -4, 10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, -1, -2, -3, -4,
            10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, -1, -2, -3, -4, 10, 20, 30, 40, 50, 60, 70,
            80, 1, 2, 3, 4, -1, -2, -3, -4, 10, 20, 30, 40, 50, 60, 70, 80,
        ],
    );
    assert_eq!(
        <[i8; 64]>::from(a.simd_gt(b)),
        [
            -1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1,
            0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            -1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1
        ]
    );
}

#[simd_test]
fn simd_gt_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            2, 2, 2, 5, 4, 7, 6, 9, 5, 25, 25, 45, 45, 65, 65, 85, 2, 2, 2, 5, 4, 7, 6, 9, 5, 25,
            25, 45, 45, 65, 65, 85, 2, 2, 2, 5, 4, 7, 6, 9, 5, 25, 25, 45, 45, 65, 65, 85, 2, 2, 2,
            5, 4, 7, 6, 9, 5, 25, 25, 45, 45, 65, 65, 85,
        ],
    );
    let b = u8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20,
            30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80, 1, 2,
            3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80,
        ],
    );
    assert_eq!(
        <[i8; 64]>::from(a.simd_gt(b)),
        [
            -1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0,
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1
        ]
    );
}

#[simd_test]
fn simd_gt_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            2, 2, 2, 5, 0, 0, 0, 0, 5, 25, 25, 45, 45, 65, 65, 85, 2, 2, 2, 5, 0, 0, 0, 0, 5, 25,
            25, 45, 45, 65, 65, 85,
        ],
    );
    let b = i16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, -1, -2, -3, -4, 10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, -1, -2, -3, -4,
            10, 20, 30, 40, 50, 60, 70, 80,
        ],
    );
    assert_eq!(
        <[i16; 32]>::from(a.simd_gt(b)),
        [
            -1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1,
            0, -1, 0, -1, 0, -1, 0, -1
        ]
    );
}

#[simd_test]
fn simd_gt_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            2, 2, 2, 5, 4, 7, 6, 9, 5, 25, 25, 45, 45, 65, 65, 85, 2, 2, 2, 5, 4, 7, 6, 9, 5, 25,
            25, 45, 45, 65, 65, 85,
        ],
    );
    let b = u16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20,
            30, 40, 50, 60, 70, 80,
        ],
    );
    assert_eq!(
        <[i16; 32]>::from(a.simd_gt(b)),
        [
            -1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1
        ]
    );
}

#[simd_test]
fn simd_gt_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(simd, &[2, 2, 2, 5, 0, 0, 0, 0, 2, 2, 2, 5, 0, 0, 0, 0]);
    let b = i32x16::from_slice(
        simd,
        &[1, 2, 3, 4, -1, -2, -3, -4, 1, 2, 3, 4, -1, -2, -3, -4],
    );
    assert_eq!(
        <[i32; 16]>::from(a.simd_gt(b)),
        [-1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1]
    );
}

#[simd_test]
fn simd_gt_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(simd, &[2, 2, 2, 5, 4, 7, 6, 9, 2, 2, 2, 5, 4, 7, 6, 9]);
    let b = u32x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        <[i32; 16]>::from(a.simd_gt(b)),
        [-1, 0, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1]
    );
}

#[simd_test]
fn simd_gt_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f64x8::from_slice(simd, &[1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 8.0, 7.0]);
    assert_eq!(<[i64; 8]>::from(a.simd_gt(b)), [-1, -1, 0, 0, 0, 0, 0, -1]);
}

// Additional concrete rows for this operation.

#[simd_test]
fn simd_gt_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[0_i64, 0_i64]);
    assert_eq!(<[i64; 2]>::from(simd.simd_gt_i64x2(a, b)), [-1_i64, 0_i64]);
}

#[simd_test]
fn simd_gt_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[0_i64, 0_i64, 0_i64, 0_i64]);
    assert_eq!(
        <[i64; 4]>::from(simd.simd_gt_i64x4(a, b)),
        [-1_i64, 0_i64, -1_i64, 0_i64]
    );
}

#[simd_test]
fn simd_gt_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64],
    );
    assert_eq!(
        <[i64; 8]>::from(simd.simd_gt_i64x8(a, b)),
        [-1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64]
    );
}

#[simd_test]
fn simd_gt_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[0_u64, 0_u64]);
    assert_eq!(<[i64; 2]>::from(simd.simd_gt_u64x2(a, b)), [-1_i64, -1_i64]);
}

#[simd_test]
fn simd_gt_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[0_u64, 0_u64, 0_u64, 0_u64]);
    assert_eq!(
        <[i64; 4]>::from(simd.simd_gt_u64x4(a, b)),
        [-1_i64, -1_i64, -1_i64, -1_i64]
    );
}

#[simd_test]
fn simd_gt_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64, 0_u64],
    );
    assert_eq!(
        <[i64; 8]>::from(simd.simd_gt_u64x8(a, b)),
        [
            -1_i64, -1_i64, -1_i64, -1_i64, -1_i64, -1_i64, -1_i64, -1_i64
        ]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn simd_gt_f64x2<S: Simd>(simd: S) {
    let a_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 2] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let expected: [i64; 2] = core::array::from_fn(|i| {
        if a_values[i] > b_values[i] {
            -1_i64
        } else {
            0_i64
        }
    });
    let result = simd.simd_gt_f64x2(a, b);
    assert_eq!(<[i64; 2]>::from(result), expected);
}

#[simd_test]
fn simd_gt_f64x4<S: Simd>(simd: S) {
    let a_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 4] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x4::from_slice(simd, &a_values);
    let b = f64x4::from_slice(simd, &b_values);
    let expected: [i64; 4] = core::array::from_fn(|i| {
        if a_values[i] > b_values[i] {
            -1_i64
        } else {
            0_i64
        }
    });
    let result = simd.simd_gt_f64x4(a, b);
    assert_eq!(<[i64; 4]>::from(result), expected);
}
