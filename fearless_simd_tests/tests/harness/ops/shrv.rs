// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn shrv_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[i32::MIN, -65536, 65536, i32::MAX]);
    assert_eq!(
        *(a >> i32x4::splat(simd, 8)),
        [-8388608, -256, 256, 8388607]
    );
}

#[simd_test]
fn shrv_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[u32::MAX, 2147483648, 65536, 256]);
    assert_eq!(*(a >> u32x4::splat(simd, 8)), [16777215, 8388608, 256, 1]);
}

#[simd_test]
fn shrv_u32x4_varied<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[u32::MAX; 4]);
    const SHIFTS: [u32; 4] = [0, 1, 2, 3];
    assert_eq!(
        *(a >> u32x4::from_slice(simd, &SHIFTS)),
        SHIFTS.map(|x| u32::MAX >> x)
    );
}

#[simd_test]
fn shrv_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            -128, -64, -33, -1, 127, 64, 33, 1, -2, -4, -8, -16, 0, 2, 4, 8,
        ],
    );
    let shifts = i8x16::from_slice(simd, &[1, 2, 3, 7, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3]);
    assert_eq!(
        *(a >> shifts),
        [-64, -16, -5, -1, 63, 16, 4, 1, -1, -1, -1, -1, 0, 1, 1, 1]
    );
}

#[simd_test]
fn shrv_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[255, 128, 64, 32, 16, 8, 4, 2, 1, 3, 5, 7, 15, 31, 63, 127],
    );
    let shifts = u8x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
    assert_eq!(
        *(a >> shifts),
        [127, 32, 8, 2, 0, 0, 0, 1, 1, 1, 1, 0, 0, 3, 15, 63]
    );
}

#[simd_test]
fn shrv_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[-32768, -16384, -1025, -1, 32767, 16384, 1025, 1]);
    let shifts = i16x8::from_slice(simd, &[1, 2, 3, 15, 1, 2, 3, 0]);
    assert_eq!(
        *(a >> shifts),
        [-16384, -4096, -129, -1, 16383, 4096, 128, 1]
    );
}

#[simd_test]
fn shrv_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[65535, 32768, 16384, 8192, 1, 255, 1024, 4096]);
    let shifts = u16x8::from_slice(simd, &[1, 2, 3, 4, 0, 4, 5, 12]);
    assert_eq!(*(a >> shifts), [32767, 8192, 2048, 512, 1, 15, 32, 1]);
}

#[simd_test]
fn shrv_i8x64<S: Simd>(simd: S) {
    const A: [i8; 16] = [
        -128, -64, -33, -1, 127, 64, 33, 1, -2, -4, -8, -16, 0, 2, 4, 8,
    ];
    const SHIFTS: [i8; 16] = [1, 2, 3, 7, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3];
    const EXPECTED: [i8; 16] = [-64, -16, -5, -1, 63, 16, 4, 1, -1, -1, -1, -1, 0, 1, 1, 1];
    let a_vals: [i8; 64] = core::array::from_fn(|i| A[i % 16]);
    let shift_vals: [i8; 64] = core::array::from_fn(|i| SHIFTS[i % 16]);
    let expected: [i8; 64] = core::array::from_fn(|i| EXPECTED[i % 16]);
    let a = i8x64::from_slice(simd, &a_vals);
    let shifts = i8x64::from_slice(simd, &shift_vals);
    assert_eq!(*(a >> shifts), expected);
}

#[simd_test]
fn shrv_u8x64<S: Simd>(simd: S) {
    const A: [u8; 16] = [255, 128, 64, 32, 16, 8, 4, 2, 1, 3, 5, 7, 15, 31, 63, 127];
    const SHIFTS: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 1, 0, 1, 2, 3, 4, 3, 2, 1];
    const EXPECTED: [u8; 16] = [127, 32, 8, 2, 0, 0, 0, 1, 1, 1, 1, 0, 0, 3, 15, 63];
    let a_vals: [u8; 64] = core::array::from_fn(|i| A[i % 16]);
    let shift_vals: [u8; 64] = core::array::from_fn(|i| SHIFTS[i % 16]);
    let expected: [u8; 64] = core::array::from_fn(|i| EXPECTED[i % 16]);
    let a = u8x64::from_slice(simd, &a_vals);
    let shifts = u8x64::from_slice(simd, &shift_vals);
    assert_eq!(*(a >> shifts), expected);
}

#[simd_test]
fn shrv_i16x32<S: Simd>(simd: S) {
    const A: [i16; 8] = [-32768, -16384, -1025, -1, 32767, 16384, 1025, 1];
    const SHIFTS: [i16; 8] = [1, 2, 3, 15, 1, 2, 3, 0];
    const EXPECTED: [i16; 8] = [-16384, -4096, -129, -1, 16383, 4096, 128, 1];
    let a_vals: [i16; 32] = core::array::from_fn(|i| A[i % 8]);
    let shift_vals: [i16; 32] = core::array::from_fn(|i| SHIFTS[i % 8]);
    let expected: [i16; 32] = core::array::from_fn(|i| EXPECTED[i % 8]);
    let a = i16x32::from_slice(simd, &a_vals);
    let shifts = i16x32::from_slice(simd, &shift_vals);
    assert_eq!(*(a >> shifts), expected);
}

#[simd_test]
fn shrv_u16x32<S: Simd>(simd: S) {
    const A: [u16; 8] = [65535, 32768, 16384, 8192, 1, 255, 1024, 4096];
    const SHIFTS: [u16; 8] = [1, 2, 3, 4, 0, 4, 5, 12];
    const EXPECTED: [u16; 8] = [32767, 8192, 2048, 512, 1, 15, 32, 1];
    let a_vals: [u16; 32] = core::array::from_fn(|i| A[i % 8]);
    let shift_vals: [u16; 32] = core::array::from_fn(|i| SHIFTS[i % 8]);
    let expected: [u16; 32] = core::array::from_fn(|i| EXPECTED[i % 8]);
    let a = u16x32::from_slice(simd, &a_vals);
    let shifts = u16x32::from_slice(simd, &shift_vals);
    assert_eq!(*(a >> shifts), expected);
}

#[simd_test]
fn shrv_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
        ],
    );
    assert_eq!(
        *(a >> i32x16::splat(simd, 8)),
        [
            -8388608, -256, 256, 8388607, -8388608, -256, 256, 8388607, -8388608, -256, 256,
            8388607, -8388608, -256, 256, 8388607
        ]
    );
}

#[simd_test]
fn shrv_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            u32::MAX,
            2147483648,
            65536,
            256,
            u32::MAX,
            2147483648,
            65536,
            256,
            u32::MAX,
            2147483648,
            65536,
            256,
            u32::MAX,
            2147483648,
            65536,
            256,
        ],
    );
    assert_eq!(
        *(a >> u32x16::splat(simd, 8)),
        [
            16777215, 8388608, 256, 1, 16777215, 8388608, 256, 1, 16777215, 8388608, 256, 1,
            16777215, 8388608, 256, 1
        ]
    );
}

#[simd_test]
fn shrv_u32x16_varied<S: Simd>(simd: S) {
    let a = u32x16::splat(simd, u32::MAX);
    const SHIFTS: [u32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    assert_eq!(
        *(a >> u32x16::from_slice(simd, &SHIFTS)),
        SHIFTS.map(|x| u32::MAX >> x)
    );
}

// Additional concrete rows for this operation.

#[simd_test]
fn shrv_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[0_i64, 1_i64]);
    assert_eq!(*simd.shrv_i64x2(a, b), [1_i64, -1_i64]);
}

#[simd_test]
fn shrv_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[0_i64, 1_i64, 2_i64, 3_i64]);
    assert_eq!(*simd.shrv_i64x4(a, b), [1_i64, -1_i64, 0_i64, -1_i64]);
}

#[simd_test]
fn shrv_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[0_i64, 1_i64, 2_i64, 3_i64, 0_i64, 1_i64, 2_i64, 3_i64],
    );
    assert_eq!(
        *simd.shrv_i64x8(a, b),
        [1_i64, -1_i64, 0_i64, -1_i64, 5_i64, -3_i64, 1_i64, -1_i64]
    );
}

#[simd_test]
fn shrv_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[0_u64, 1_u64]);
    assert_eq!(*simd.shrv_u64x2(a, b), [1_u64, 1_u64]);
}

#[simd_test]
fn shrv_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[0_u64, 1_u64, 2_u64, 3_u64]);
    assert_eq!(*simd.shrv_u64x4(a, b), [1_u64, 1_u64, 0_u64, 0_u64]);
}

#[simd_test]
fn shrv_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[0_u64, 1_u64, 2_u64, 3_u64, 0_u64, 1_u64, 2_u64, 3_u64],
    );
    assert_eq!(
        *simd.shrv_u64x8(a, b),
        [1_u64, 1_u64, 0_u64, 0_u64, 5_u64, 3_u64, 1_u64, 1_u64]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn shrv_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            -128, -64, -33, -1, 127, 64, 33, 1, -2, -4, -8, -16, 0, 2, 4, 8, -128, -64, -33, -1,
            127, 64, 33, 1, -2, -4, -8, -16, 0, 2, 4, 8,
        ],
    );
    let shifts = i8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4,
            5, 6, 7,
        ],
    );
    assert_eq!(
        *simd.shrv_i8x32(a, shifts),
        [
            -128, -32, -9, -1, 7, 2, 0, 0, -2, -2, -2, -2, 0, 0, 0, 0, -128, -32, -9, -1, 7, 2, 0,
            0, -2, -2, -2, -2, 0, 0, 0, 0,
        ]
    );
}

#[simd_test]
fn shrv_u8x32<S: Simd>(simd: S) {
    let values: [u8; 32] = core::array::from_fn(|i| (i % 31) as u8 + 1_u8);
    let shift_values: [u8; 32] = core::array::from_fn(|i| (i % 3) as u8);
    let a = u8x32::from_slice(simd, &values);
    let shifts = u8x32::from_slice(simd, &shift_values);
    let expected: [u8; 32] = core::array::from_fn(|i| values[i] >> shift_values[i]);
    let result = simd.shrv_u8x32(a, shifts);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shrv_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            -32768, -16384, -1025, -1, 32767, 16384, 1025, 1, -32768, -16384, -1025, -1, 32767,
            16384, 1025, 1,
        ],
    );
    let shifts = i16x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(
        *simd.shrv_i16x16(a, shifts),
        [
            -32768, -8192, -257, -1, 2047, 512, 16, 0, -32768, -8192, -257, -1, 2047, 512, 16, 0,
        ]
    );
}

#[simd_test]
fn shrv_u16x16<S: Simd>(simd: S) {
    let values: [u16; 16] = core::array::from_fn(|i| (i % 31) as u16 + 1_u16);
    let shift_values: [u16; 16] = core::array::from_fn(|i| (i % 3) as u16);
    let a = u16x16::from_slice(simd, &values);
    let shifts = u16x16::from_slice(simd, &shift_values);
    let expected: [u16; 16] = core::array::from_fn(|i| values[i] >> shift_values[i]);
    let result = simd.shrv_u16x16(a, shifts);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn shrv_i32x8<S: Simd>(simd: S) {
    const MIN: i32 = i32::MIN;
    const MAX: i32 = i32::MAX;
    let a = i32x8::from_slice(
        simd,
        &[MIN, -1073741824, -65537, -1, MAX, 1073741824, 65537, 1],
    );
    let shifts = i32x8::from_slice(simd, &[0, 1, 2, 31, 4, 5, 16, 31]);
    assert_eq!(
        *simd.shrv_i32x8(a, shifts),
        [MIN, -536870912, -16385, -1, 134217727, 33554432, 1, 0,]
    );
}

#[simd_test]
fn shrv_u32x8<S: Simd>(simd: S) {
    let values: [u32; 8] = core::array::from_fn(|i| (i % 31) as u32 + 1_u32);
    let shift_values: [u32; 8] = core::array::from_fn(|i| (i % 3) as u32);
    let a = u32x8::from_slice(simd, &values);
    let shifts = u32x8::from_slice(simd, &shift_values);
    let expected: [u32; 8] = core::array::from_fn(|i| values[i] >> shift_values[i]);
    let result = simd.shrv_u32x8(a, shifts);
    assert_eq!(result.as_slice(), expected.as_slice());
}
