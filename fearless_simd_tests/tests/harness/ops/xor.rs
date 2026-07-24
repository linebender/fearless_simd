// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn xor_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, 0, 0, 0, 0]);
    let b = i8x16::from_slice(
        simd,
        &[-1, -1, 0, 0, 5, 4, 7, 6, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    assert_eq!(
        *(a ^ b),
        [-1, -2, 2, 3, 1, 1, 1, 1, 0, -1, 0, -1, -1, 0, -1, 0]
    );
}

#[simd_test]
fn xor_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 0, 0, 0, 0]);
    let b = u8x16::from_slice(simd, &[1, 1, 0, 0, 5, 4, 7, 6, 1, 0, 1, 0, 1, 0, 1, 0]);
    assert_eq!(*(a ^ b), [1, 0, 2, 3, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]);
}

#[simd_test]
fn xor_mask8x16<S: Simd>(simd: S) {
    let a = mask8x16::from_slice(
        simd,
        &[0, -1, -1, 0, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0],
    );
    let b = mask8x16::from_slice(
        simd,
        &[-1, -1, 0, 0, -1, -1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    assert_eq!(
        <[i8; 16]>::from(a ^ b),
        [-1, 0, -1, 0, 0, -1, 0, -1, 0, -1, -1, 0, 0, -1, -1, 0]
    );
}

#[simd_test]
fn xor_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1,
            -1, 0, 0, 0, 0,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            -1, -1, 0, 0, 5, 4, 7, 6, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, 5, 4, 7, 6, -1, 0,
            -1, 0, -1, 0, -1, 0,
        ],
    );
    assert_eq!(
        *(a ^ b),
        [
            -1, -2, 2, 3, 1, 1, 1, 1, 0, -1, 0, -1, -1, 0, -1, 0, -1, -2, 2, 3, 1, 1, 1, 1, 0, -1,
            0, -1, -1, 0, -1, 0
        ]
    );
}

#[simd_test]
fn xor_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 0,
            0, 0, 0,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            1, 1, 0, 0, 5, 4, 7, 6, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 5, 4, 7, 6, 1, 0, 1, 0, 1,
            0, 1, 0,
        ],
    );
    assert_eq!(
        *(a ^ b),
        [
            1, 0, 2, 3, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 2, 3, 1, 1, 1, 1, 0, 1, 0, 1, 1,
            0, 1, 0
        ]
    );
}

#[simd_test]
fn xor_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1,
            -1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5,
            6, 7, -1, -1, -1, -1, 0, 0, 0, 0,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            -1, -1, 0, 0, 5, 4, 7, 6, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, 5, 4, 7, 6, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, -1, 0, 0, 5, 4, 7, 6, -1, 0, -1, 0, -1, 0, -1, 0, -1, -1, 0,
            0, 5, 4, 7, 6, -1, 0, -1, 0, -1, 0, -1, 0,
        ],
    );
    assert_eq!(
        *(a ^ b),
        [
            -1, -2, 2, 3, 1, 1, 1, 1, 0, -1, 0, -1, -1, 0, -1, 0, -1, -2, 2, 3, 1, 1, 1, 1, 0, -1,
            0, -1, -1, 0, -1, 0, -1, -2, 2, 3, 1, 1, 1, 1, 0, -1, 0, -1, -1, 0, -1, 0, -1, -2, 2,
            3, 1, 1, 1, 1, 0, -1, 0, -1, -1, 0, -1, 0
        ]
    );
}

#[simd_test]
fn xor_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 1, 1,
            1, 1, 0, 0, 0, 0,
        ],
    );
    let b = u8x64::from_slice(
        simd,
        &[
            1, 1, 0, 0, 5, 4, 7, 6, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 5, 4, 7, 6, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 1, 0, 0, 5, 4, 7, 6, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 5, 4, 7, 6, 1, 0,
            1, 0, 1, 0, 1, 0,
        ],
    );
    assert_eq!(
        *(a ^ b),
        [
            1, 0, 2, 3, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 2, 3, 1, 1, 1, 1, 0, 1, 0, 1, 1,
            0, 1, 0, 1, 0, 2, 3, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 2, 3, 1, 1, 1, 1, 0, 1,
            0, 1, 1, 0, 1, 0
        ]
    );
}

// Additional concrete rows for this operation.

#[simd_test]
fn xor_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[3_i64, 3_i64]);
    assert_eq!(*simd.xor_i64x2(a, b), [2_i64, -3_i64]);
}

#[simd_test]
fn xor_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[3_i64, 3_i64, 3_i64, 3_i64]);
    assert_eq!(*simd.xor_i64x4(a, b), [2_i64, -3_i64, 0_i64, -1_i64]);
}

#[simd_test]
fn xor_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64, 3_i64],
    );
    assert_eq!(
        *simd.xor_i64x8(a, b),
        [2_i64, -3_i64, 0_i64, -1_i64, 6_i64, -7_i64, 4_i64, -5_i64]
    );
}

#[simd_test]
fn xor_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[3_u64, 3_u64]);
    assert_eq!(*simd.xor_u64x2(a, b), [2_u64, 1_u64]);
}

#[simd_test]
fn xor_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[3_u64, 3_u64, 3_u64, 3_u64]);
    assert_eq!(*simd.xor_u64x4(a, b), [2_u64, 1_u64, 0_u64, 7_u64]);
}

#[simd_test]
fn xor_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[3_u64, 3_u64, 3_u64, 3_u64, 3_u64, 3_u64, 3_u64, 3_u64],
    );
    assert_eq!(
        *simd.xor_u64x8(a, b),
        [2_u64, 1_u64, 0_u64, 7_u64, 6_u64, 5_u64, 4_u64, 11_u64]
    );
}

#[simd_test]
fn xor_mask64x2<S: Simd>(simd: S) {
    let a = mask64x2::from_slice(simd, &[-1_i64, 0_i64]);
    let b = mask64x2::from_slice(simd, &[0_i64, -1_i64]);
    assert_eq!(<[i64; 2]>::from(simd.xor_mask64x2(a, b)), [-1_i64, -1_i64]);
}

#[simd_test]
fn xor_mask64x4<S: Simd>(simd: S) {
    let a = mask64x4::from_slice(simd, &[-1_i64, 0_i64, -1_i64, 0_i64]);
    let b = mask64x4::from_slice(simd, &[0_i64, -1_i64, 0_i64, -1_i64]);
    assert_eq!(
        <[i64; 4]>::from(simd.xor_mask64x4(a, b)),
        [-1_i64, -1_i64, -1_i64, -1_i64]
    );
}

#[simd_test]
fn xor_mask64x8<S: Simd>(simd: S) {
    let a = mask64x8::from_slice(
        simd,
        &[-1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64],
    );
    let b = mask64x8::from_slice(
        simd,
        &[0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64],
    );
    assert_eq!(
        <[i64; 8]>::from(simd.xor_mask64x8(a, b)),
        [
            -1_i64, -1_i64, -1_i64, -1_i64, -1_i64, -1_i64, -1_i64, -1_i64
        ]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn xor_i16x8<S: Simd>(simd: S) {
    let a_values: [i16; 8] = core::array::from_fn(|i| (i % 5) as i16 + 1_i16);
    let b_values: [i16; 8] = core::array::from_fn(|i| (i % 3) as i16 + 3_i16);
    let a = i16x8::from_slice(simd, &a_values);
    let b = i16x8::from_slice(simd, &b_values);
    let expected: [i16; 8] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_i16x8(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_u16x8<S: Simd>(simd: S) {
    let a_values: [u16; 8] = core::array::from_fn(|i| (i % 5) as u16 + 1_u16);
    let b_values: [u16; 8] = core::array::from_fn(|i| (i % 3) as u16 + 3_u16);
    let a = u16x8::from_slice(simd, &a_values);
    let b = u16x8::from_slice(simd, &b_values);
    let expected: [u16; 8] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_u16x8(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_i16x16<S: Simd>(simd: S) {
    let a_values: [i16; 16] = core::array::from_fn(|i| (i % 5) as i16 + 1_i16);
    let b_values: [i16; 16] = core::array::from_fn(|i| (i % 3) as i16 + 3_i16);
    let a = i16x16::from_slice(simd, &a_values);
    let b = i16x16::from_slice(simd, &b_values);
    let expected: [i16; 16] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_i16x16(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_u16x16<S: Simd>(simd: S) {
    let a_values: [u16; 16] = core::array::from_fn(|i| (i % 5) as u16 + 1_u16);
    let b_values: [u16; 16] = core::array::from_fn(|i| (i % 3) as u16 + 3_u16);
    let a = u16x16::from_slice(simd, &a_values);
    let b = u16x16::from_slice(simd, &b_values);
    let expected: [u16; 16] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_u16x16(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_i16x32<S: Simd>(simd: S) {
    let a_values: [i16; 32] = core::array::from_fn(|i| (i % 5) as i16 + 1_i16);
    let b_values: [i16; 32] = core::array::from_fn(|i| (i % 3) as i16 + 3_i16);
    let a = i16x32::from_slice(simd, &a_values);
    let b = i16x32::from_slice(simd, &b_values);
    let expected: [i16; 32] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_i16x32(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_u16x32<S: Simd>(simd: S) {
    let a_values: [u16; 32] = core::array::from_fn(|i| (i % 5) as u16 + 1_u16);
    let b_values: [u16; 32] = core::array::from_fn(|i| (i % 3) as u16 + 3_u16);
    let a = u16x32::from_slice(simd, &a_values);
    let b = u16x32::from_slice(simd, &b_values);
    let expected: [u16; 32] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_u16x32(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_i32x4<S: Simd>(simd: S) {
    let a_values: [i32; 4] = core::array::from_fn(|i| (i % 5) as i32 + 1_i32);
    let b_values: [i32; 4] = core::array::from_fn(|i| (i % 3) as i32 + 3_i32);
    let a = i32x4::from_slice(simd, &a_values);
    let b = i32x4::from_slice(simd, &b_values);
    let expected: [i32; 4] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_i32x4(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_u32x4<S: Simd>(simd: S) {
    let a_values: [u32; 4] = core::array::from_fn(|i| (i % 5) as u32 + 1_u32);
    let b_values: [u32; 4] = core::array::from_fn(|i| (i % 3) as u32 + 3_u32);
    let a = u32x4::from_slice(simd, &a_values);
    let b = u32x4::from_slice(simd, &b_values);
    let expected: [u32; 4] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_u32x4(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_i32x8<S: Simd>(simd: S) {
    let a_values: [i32; 8] = core::array::from_fn(|i| (i % 5) as i32 + 1_i32);
    let b_values: [i32; 8] = core::array::from_fn(|i| (i % 3) as i32 + 3_i32);
    let a = i32x8::from_slice(simd, &a_values);
    let b = i32x8::from_slice(simd, &b_values);
    let expected: [i32; 8] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_i32x8(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_u32x8<S: Simd>(simd: S) {
    let a_values: [u32; 8] = core::array::from_fn(|i| (i % 5) as u32 + 1_u32);
    let b_values: [u32; 8] = core::array::from_fn(|i| (i % 3) as u32 + 3_u32);
    let a = u32x8::from_slice(simd, &a_values);
    let b = u32x8::from_slice(simd, &b_values);
    let expected: [u32; 8] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_u32x8(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_i32x16<S: Simd>(simd: S) {
    let a_values: [i32; 16] = core::array::from_fn(|i| (i % 5) as i32 + 1_i32);
    let b_values: [i32; 16] = core::array::from_fn(|i| (i % 3) as i32 + 3_i32);
    let a = i32x16::from_slice(simd, &a_values);
    let b = i32x16::from_slice(simd, &b_values);
    let expected: [i32; 16] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_i32x16(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_u32x16<S: Simd>(simd: S) {
    let a_values: [u32; 16] = core::array::from_fn(|i| (i % 5) as u32 + 1_u32);
    let b_values: [u32; 16] = core::array::from_fn(|i| (i % 3) as u32 + 3_u32);
    let a = u32x16::from_slice(simd, &a_values);
    let b = u32x16::from_slice(simd, &b_values);
    let expected: [u32; 16] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_u32x16(a, b);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn xor_mask8x32<S: Simd>(simd: S) {
    let a_values: [i8; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let b_values: [i8; 32] = core::array::from_fn(|i| if i % 3 == 0 { -1_i8 } else { 0_i8 });
    let a = mask8x32::from_slice(simd, &a_values);
    let b = mask8x32::from_slice(simd, &b_values);
    let expected: [i8; 32] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_mask8x32(a, b);
    assert_eq!(<[i8; 32]>::from(result), expected);
}

#[simd_test]
fn xor_mask8x64<S: Simd>(simd: S) {
    let a_values: [i8; 64] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let b_values: [i8; 64] = core::array::from_fn(|i| if i % 3 == 0 { -1_i8 } else { 0_i8 });
    let a = mask8x64::from_slice(simd, &a_values);
    let b = mask8x64::from_slice(simd, &b_values);
    let expected: [i8; 64] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_mask8x64(a, b);
    assert_eq!(<[i8; 64]>::from(result), expected);
}

#[simd_test]
fn xor_mask16x8<S: Simd>(simd: S) {
    let a_values: [i16; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let b_values: [i16; 8] = core::array::from_fn(|i| if i % 3 == 0 { -1_i16 } else { 0_i16 });
    let a = mask16x8::from_slice(simd, &a_values);
    let b = mask16x8::from_slice(simd, &b_values);
    let expected: [i16; 8] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_mask16x8(a, b);
    assert_eq!(<[i16; 8]>::from(result), expected);
}

#[simd_test]
fn xor_mask16x16<S: Simd>(simd: S) {
    let a_values: [i16; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let b_values: [i16; 16] = core::array::from_fn(|i| if i % 3 == 0 { -1_i16 } else { 0_i16 });
    let a = mask16x16::from_slice(simd, &a_values);
    let b = mask16x16::from_slice(simd, &b_values);
    let expected: [i16; 16] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_mask16x16(a, b);
    assert_eq!(<[i16; 16]>::from(result), expected);
}

#[simd_test]
fn xor_mask16x32<S: Simd>(simd: S) {
    let a_values: [i16; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let b_values: [i16; 32] = core::array::from_fn(|i| if i % 3 == 0 { -1_i16 } else { 0_i16 });
    let a = mask16x32::from_slice(simd, &a_values);
    let b = mask16x32::from_slice(simd, &b_values);
    let expected: [i16; 32] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_mask16x32(a, b);
    assert_eq!(<[i16; 32]>::from(result), expected);
}

#[simd_test]
fn xor_mask32x4<S: Simd>(simd: S) {
    let a_values: [i32; 4] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let b_values: [i32; 4] = core::array::from_fn(|i| if i % 3 == 0 { -1_i32 } else { 0_i32 });
    let a = mask32x4::from_slice(simd, &a_values);
    let b = mask32x4::from_slice(simd, &b_values);
    let expected: [i32; 4] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_mask32x4(a, b);
    assert_eq!(<[i32; 4]>::from(result), expected);
}

#[simd_test]
fn xor_mask32x8<S: Simd>(simd: S) {
    let a_values: [i32; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let b_values: [i32; 8] = core::array::from_fn(|i| if i % 3 == 0 { -1_i32 } else { 0_i32 });
    let a = mask32x8::from_slice(simd, &a_values);
    let b = mask32x8::from_slice(simd, &b_values);
    let expected: [i32; 8] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_mask32x8(a, b);
    assert_eq!(<[i32; 8]>::from(result), expected);
}

#[simd_test]
fn xor_mask32x16<S: Simd>(simd: S) {
    let a_values: [i32; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let b_values: [i32; 16] = core::array::from_fn(|i| if i % 3 == 0 { -1_i32 } else { 0_i32 });
    let a = mask32x16::from_slice(simd, &a_values);
    let b = mask32x16::from_slice(simd, &b_values);
    let expected: [i32; 16] = core::array::from_fn(|i| a_values[i] ^ b_values[i]);
    let result = simd.xor_mask32x16(a, b);
    assert_eq!(<[i32; 16]>::from(result), expected);
}
