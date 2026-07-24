// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn mul_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 100],
    );
    let b = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2],
    );

    assert_eq!(
        *(a * b),
        [
            0, 2, 6, 12, 20, 30, 70, 120, 180, 250, 74, 164, 8, 188, 132, 200
        ]
    );
}

#[simd_test]
fn mul_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            0, 1, -2, 3, -4, 5, 10, -15, 20, -25, 30, 35, -40, 50, -60, 100,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[1, 2, 3, -4, 5, -6, 7, 8, 9, 10, -11, 12, 13, -14, 15, 2],
    );

    assert_eq!(
        *(a * b),
        [
            0, 2, -6, -12, -20, -30, 70, -120, -76, 6, -74, -92, -8, 68, 124, -56
        ]
    );
}

#[simd_test]
fn mul_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[0, 5, 10, 30, 500, 0, 0, 0]);
    let b = u16x8::from_slice(simd, &[5, 8, 13, 21, 230, 0, 0, 0]);

    assert_eq!(*(a * b), [0, 40, 130, 630, 49464, 0, 0, 0]);
}

#[simd_test]
fn mul_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 5464, 23234, 456456]);
    let b = u32x4::from_slice(simd, &[23, 34, 565, 34234]);

    assert_eq!(*(a * b), [23, 185776, 13127210, 2741412816]);
}

#[simd_test]
fn mul_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, 0.0, 13.34, 234234.0]);
    let b = f32x4::from_slice(simd, &[-8.1, 7.9, -9.8, 3243.6]);

    assert_eq!(*(a * b), [83.43001, 0.0, -130.73201, 759761400.0]);
}

#[simd_test]
fn mul_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::from_slice(simd, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(*(a * b), [2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0]);
}

#[simd_test]
fn mul_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::from_slice(simd, &[2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(*(a * b), [2, 6, 12, 20, 30, 42, 56, 72]);
}

#[simd_test]
fn mul_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u32x8::from_slice(simd, &[2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(*(a * b), [2, 6, 12, 20, 30, 42, 56, 72]);
}

#[simd_test]
fn mul_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ],
    );
    assert_eq!(
        *(a * b),
        [
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
            32.0
        ]
    );
}

#[simd_test]
fn mul_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i32x16::from_slice(simd, &[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
    assert_eq!(
        *(a * b),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    );
}

#[simd_test]
fn mul_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u32x16::from_slice(simd, &[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
    assert_eq!(
        *(a * b),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    );
}

#[simd_test]
fn mul_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f64x8::from_slice(simd, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(*(a * b), [2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0]);
}

// Additional concrete rows for this operation.

#[simd_test]
fn mul_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[3_i64, 3_i64]);
    assert_eq!(*simd.mul_i64x2(a, b), [3_i64, -6_i64]);
}

#[simd_test]
fn mul_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[3_i64, 3_i64, 3_i64, 3_i64]);
    assert_eq!(*simd.mul_i64x4(a, b), [3_i64, -6_i64, 9_i64, -12_i64]);
}

#[simd_test]
fn mul_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64],
    );
    assert_eq!(
        *simd.mul_i64x8(a, b),
        [
            3_i64, -6_i64, 9_i64, -12_i64, 15_i64, -18_i64, 21_i64, -24_i64
        ]
    );
}

#[simd_test]
fn mul_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[2_u64, 2_u64]);
    assert_eq!(*simd.mul_u64x2(a, b), [2_u64, 4_u64]);
}

#[simd_test]
fn mul_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[2_u64, 2_u64, 2_u64, 2_u64]);
    assert_eq!(*simd.mul_u64x4(a, b), [2_u64, 4_u64, 6_u64, 8_u64]);
}

#[simd_test]
fn mul_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[2_u64, 2_u64, 2_u64, 2_u64, 2_u64, 2_u64, 2_u64, 2_u64],
    );
    assert_eq!(
        *simd.mul_u64x8(a, b),
        [2_u64, 4_u64, 6_u64, 8_u64, 10_u64, 12_u64, 14_u64, 16_u64]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn mul_i8x32<S: Simd>(simd: S) {
    let a_values: [i8; 32] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let b_values: [i8; 32] = core::array::from_fn(|i| (i % 7) as i8 + 1_i8);
    let a = i8x32::from_slice(simd, &a_values);
    let b = i8x32::from_slice(simd, &b_values);
    let expected: [i8; 32] = core::array::from_fn(|i| a_values[i].wrapping_mul(b_values[i]));
    let result = simd.mul_i8x32(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_u8x32<S: Simd>(simd: S) {
    let a_values: [u8; 32] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let b_values: [u8; 32] = core::array::from_fn(|i| (i % 7) as u8 + 1_u8);
    let a = u8x32::from_slice(simd, &a_values);
    let b = u8x32::from_slice(simd, &b_values);
    let expected: [u8; 32] = core::array::from_fn(|i| a_values[i] * b_values[i]);
    let result = simd.mul_u8x32(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_i8x64<S: Simd>(simd: S) {
    let a_values: [i8; 64] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let b_values: [i8; 64] = core::array::from_fn(|i| (i % 7) as i8 + 1_i8);
    let a = i8x64::from_slice(simd, &a_values);
    let b = i8x64::from_slice(simd, &b_values);
    let expected: [i8; 64] = core::array::from_fn(|i| a_values[i].wrapping_mul(b_values[i]));
    let result = simd.mul_i8x64(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_u8x64<S: Simd>(simd: S) {
    let a_values: [u8; 64] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let b_values: [u8; 64] = core::array::from_fn(|i| (i % 7) as u8 + 1_u8);
    let a = u8x64::from_slice(simd, &a_values);
    let b = u8x64::from_slice(simd, &b_values);
    let expected: [u8; 64] = core::array::from_fn(|i| a_values[i] * b_values[i]);
    let result = simd.mul_u8x64(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_i16x8<S: Simd>(simd: S) {
    let a_values: [i16; 8] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let b_values: [i16; 8] = core::array::from_fn(|i| (i % 7) as i16 + 1_i16);
    let a = i16x8::from_slice(simd, &a_values);
    let b = i16x8::from_slice(simd, &b_values);
    let expected: [i16; 8] = core::array::from_fn(|i| a_values[i] * b_values[i]);
    let result = simd.mul_i16x8(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_i16x16<S: Simd>(simd: S) {
    let a_values: [i16; 16] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let b_values: [i16; 16] = core::array::from_fn(|i| (i % 7) as i16 + 1_i16);
    let a = i16x16::from_slice(simd, &a_values);
    let b = i16x16::from_slice(simd, &b_values);
    let expected: [i16; 16] = core::array::from_fn(|i| a_values[i] * b_values[i]);
    let result = simd.mul_i16x16(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_u16x16<S: Simd>(simd: S) {
    let a_values: [u16; 16] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let b_values: [u16; 16] = core::array::from_fn(|i| (i % 7) as u16 + 1_u16);
    let a = u16x16::from_slice(simd, &a_values);
    let b = u16x16::from_slice(simd, &b_values);
    let expected: [u16; 16] = core::array::from_fn(|i| a_values[i] * b_values[i]);
    let result = simd.mul_u16x16(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_i16x32<S: Simd>(simd: S) {
    let a_values: [i16; 32] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let b_values: [i16; 32] = core::array::from_fn(|i| (i % 7) as i16 + 1_i16);
    let a = i16x32::from_slice(simd, &a_values);
    let b = i16x32::from_slice(simd, &b_values);
    let expected: [i16; 32] = core::array::from_fn(|i| a_values[i] * b_values[i]);
    let result = simd.mul_i16x32(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_u16x32<S: Simd>(simd: S) {
    let a_values: [u16; 32] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let b_values: [u16; 32] = core::array::from_fn(|i| (i % 7) as u16 + 1_u16);
    let a = u16x32::from_slice(simd, &a_values);
    let b = u16x32::from_slice(simd, &b_values);
    let expected: [u16; 32] = core::array::from_fn(|i| a_values[i] * b_values[i]);
    let result = simd.mul_u16x32(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_i32x4<S: Simd>(simd: S) {
    let a_values: [i32; 4] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let b_values: [i32; 4] = core::array::from_fn(|i| (i % 7) as i32 + 1_i32);
    let a = i32x4::from_slice(simd, &a_values);
    let b = i32x4::from_slice(simd, &b_values);
    let expected: [i32; 4] = core::array::from_fn(|i| a_values[i] * b_values[i]);
    let result = simd.mul_i32x4(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_f64x2<S: Simd>(simd: S) {
    let a_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 2] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x2::from_slice(simd, &a_values);
    let b = f64x2::from_slice(simd, &b_values);
    let expected: [f64; 2] = core::array::from_fn(|i| a_values[i] * b_values[i]);
    let result = simd.mul_f64x2(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn mul_f64x4<S: Simd>(simd: S) {
    let a_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let b_values: [f64; 4] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let a = f64x4::from_slice(simd, &a_values);
    let b = f64x4::from_slice(simd, &b_values);
    let expected: [f64; 4] = core::array::from_fn(|i| a_values[i] * b_values[i]);
    let result = simd.mul_f64x4(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}
