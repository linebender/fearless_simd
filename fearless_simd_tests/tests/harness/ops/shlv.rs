// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn shlv_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0xFFFFFFFF, 0xFFFF, 0xFF, 0]);
    assert_eq!(
        *(a << u32x4::splat(simd, 4)),
        [0xFFFFFFF0, 0xFFFF0, 0xFF0, 0]
    );
}

#[simd_test]
fn shlv_u32x4_varied<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[u32::MAX; 4]);
    const SHIFTS: [u32; 4] = [0, 1, 2, 3];
    assert_eq!(
        *(a << u32x4::from_slice(simd, &SHIFTS)),
        SHIFTS.map(|x| u32::MAX << x)
    );
}

#[simd_test]
fn shlv_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[64, 65, -64, -65, 1, 2, 3, 4, -1, -2, -3, -4, 15, 16, 31, 32],
    );
    let shifts = i8x16::from_slice(simd, &[1, 2, 1, 2, 0, 1, 2, 3, 1, 2, 3, 4, 3, 2, 1, 0]);
    assert_eq!(
        *(a << shifts),
        [
            -128, 4, -128, -4, 1, 4, 12, 32, -2, -8, -24, -64, 120, 64, 62, 32
        ]
    );
}

#[simd_test]
fn shlv_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[255, 128, 64, 32, 16, 8, 4, 2, 1, 3, 5, 7, 15, 31, 63, 127],
    );
    let shifts = u8x16::from_slice(simd, &[4, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 3, 2, 1]);
    assert_eq!(
        *(a << shifts),
        [240, 0, 0, 0, 0, 0, 0, 0, 1, 6, 20, 56, 240, 248, 252, 254]
    );
}

#[simd_test]
fn shlv_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[16384, 8192, -16384, -8192, 1, -1, 255, -256]);
    let shifts = i16x8::from_slice(simd, &[1, 2, 1, 2, 15, 1, 4, 3]);
    assert_eq!(
        *(a << shifts),
        [-32768, -32768, -32768, -32768, -32768, -2, 4080, -2048]
    );
}

#[simd_test]
fn shlv_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[65535, 32768, 16384, 8192, 1, 255, 1024, 4096]);
    let shifts = u16x8::from_slice(simd, &[4, 1, 2, 3, 15, 4, 5, 0]);
    assert_eq!(*(a << shifts), [65520, 0, 0, 0, 32768, 4080, 32768, 4096]);
}

#[simd_test]
fn shlv_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            255, 128, 64, 32, 16, 8, 4, 2, 1, 3, 5, 7, 15, 31, 63, 127, 255, 128, 64, 32, 16, 8, 4,
            2, 1, 3, 5, 7, 15, 31, 63, 127,
        ],
    );
    let shifts = u8x32::from_slice(
        simd,
        &[
            4, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 3, 2, 1, 4, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4,
            3, 2, 1,
        ],
    );
    assert_eq!(
        *(a << shifts),
        [
            240, 0, 0, 0, 0, 0, 0, 0, 1, 6, 20, 56, 240, 248, 252, 254, 240, 0, 0, 0, 0, 0, 0, 0,
            1, 6, 20, 56, 240, 248, 252, 254
        ]
    );
}

#[simd_test]
fn shlv_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[
            65535, 32768, 16384, 8192, 1, 255, 1024, 4096, 65535, 32768, 16384, 8192, 1, 255, 1024,
            4096,
        ],
    );
    let shifts = u16x16::from_slice(simd, &[4, 1, 2, 3, 15, 4, 5, 0, 4, 1, 2, 3, 15, 4, 5, 0]);
    assert_eq!(
        *(a << shifts),
        [
            65520, 0, 0, 0, 32768, 4080, 32768, 4096, 65520, 0, 0, 0, 32768, 4080, 32768, 4096
        ]
    );
}

#[simd_test]
fn shlv_i8x64<S: Simd>(simd: S) {
    const A: [i8; 16] = [64, 65, -64, -65, 1, 2, 3, 4, -1, -2, -3, -4, 15, 16, 31, 32];
    const SHIFTS: [i8; 16] = [1, 2, 1, 2, 0, 1, 2, 3, 1, 2, 3, 4, 3, 2, 1, 0];
    const EXPECTED: [i8; 16] = [
        -128, 4, -128, -4, 1, 4, 12, 32, -2, -8, -24, -64, 120, 64, 62, 32,
    ];
    let a_vals: [i8; 64] = core::array::from_fn(|i| A[i % 16]);
    let shift_vals: [i8; 64] = core::array::from_fn(|i| SHIFTS[i % 16]);
    let expected: [i8; 64] = core::array::from_fn(|i| EXPECTED[i % 16]);
    let a = i8x64::from_slice(simd, &a_vals);
    let shifts = i8x64::from_slice(simd, &shift_vals);
    assert_eq!(*(a << shifts), expected);
}

#[simd_test]
fn shlv_u8x64<S: Simd>(simd: S) {
    const A: [u8; 16] = [255, 128, 64, 32, 16, 8, 4, 2, 1, 3, 5, 7, 15, 31, 63, 127];
    const SHIFTS: [u8; 16] = [4, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 3, 2, 1];
    const EXPECTED: [u8; 16] = [240, 0, 0, 0, 0, 0, 0, 0, 1, 6, 20, 56, 240, 248, 252, 254];
    let a_vals: [u8; 64] = core::array::from_fn(|i| A[i % 16]);
    let shift_vals: [u8; 64] = core::array::from_fn(|i| SHIFTS[i % 16]);
    let expected: [u8; 64] = core::array::from_fn(|i| EXPECTED[i % 16]);
    let a = u8x64::from_slice(simd, &a_vals);
    let shifts = u8x64::from_slice(simd, &shift_vals);
    assert_eq!(*(a << shifts), expected);
}

#[simd_test]
fn shlv_i16x32<S: Simd>(simd: S) {
    const A: [i16; 8] = [16384, 8192, -16384, -8192, 1, -1, 255, -256];
    const SHIFTS: [i16; 8] = [1, 2, 1, 2, 15, 1, 4, 3];
    const EXPECTED: [i16; 8] = [-32768, -32768, -32768, -32768, -32768, -2, 4080, -2048];
    let a_vals: [i16; 32] = core::array::from_fn(|i| A[i % 8]);
    let shift_vals: [i16; 32] = core::array::from_fn(|i| SHIFTS[i % 8]);
    let expected: [i16; 32] = core::array::from_fn(|i| EXPECTED[i % 8]);
    let a = i16x32::from_slice(simd, &a_vals);
    let shifts = i16x32::from_slice(simd, &shift_vals);
    assert_eq!(*(a << shifts), expected);
}

#[simd_test]
fn shlv_u16x32<S: Simd>(simd: S) {
    const A: [u16; 8] = [65535, 32768, 16384, 8192, 1, 255, 1024, 4096];
    const SHIFTS: [u16; 8] = [4, 1, 2, 3, 15, 4, 5, 0];
    const EXPECTED: [u16; 8] = [65520, 0, 0, 0, 32768, 4080, 32768, 4096];
    let a_vals: [u16; 32] = core::array::from_fn(|i| A[i % 8]);
    let shift_vals: [u16; 32] = core::array::from_fn(|i| SHIFTS[i % 8]);
    let expected: [u16; 32] = core::array::from_fn(|i| EXPECTED[i % 8]);
    let a = u16x32::from_slice(simd, &a_vals);
    let shifts = u16x32::from_slice(simd, &shift_vals);
    assert_eq!(*(a << shifts), expected);
}

#[simd_test]
fn shlv_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            0xFFFFFFFF, 0xFFFF, 0xFF, 0, 0xFFFFFFFF, 0xFFFF, 0xFF, 0, 0xFFFFFFFF, 0xFFFF, 0xFF, 0,
            0xFFFFFFFF, 0xFFFF, 0xFF, 0,
        ],
    );
    assert_eq!(
        *(a << u32x16::splat(simd, 4)),
        [
            0xFFFFFFF0, 0xFFFF0, 0xFF0, 0, 0xFFFFFFF0, 0xFFFF0, 0xFF0, 0, 0xFFFFFFF0, 0xFFFF0,
            0xFF0, 0, 0xFFFFFFF0, 0xFFFF0, 0xFF0, 0
        ]
    );
}

#[simd_test]
fn shlv_u32x16_varied<S: Simd>(simd: S) {
    let a = u32x16::splat(simd, u32::MAX);
    const SHIFTS: [u32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    assert_eq!(
        *(a << u32x16::from_slice(simd, &SHIFTS)),
        SHIFTS.map(|x| u32::MAX << x)
    );
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn shlv_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let b = i64x2::from_slice(simd, &[0_i64, 1_i64]);
    assert_eq!(*simd.shlv_i64x2(a, b), [1_i64, -4_i64]);
}

#[simd_test]
fn shlv_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let b = i64x4::from_slice(simd, &[0_i64, 1_i64, 2_i64, 3_i64]);
    assert_eq!(*simd.shlv_i64x4(a, b), [1_i64, -4_i64, 12_i64, -32_i64]);
}

#[simd_test]
fn shlv_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let b = i64x8::from_slice(
        simd,
        &[0_i64, 1_i64, 2_i64, 3_i64, 0_i64, 1_i64, 2_i64, 3_i64],
    );
    assert_eq!(
        *simd.shlv_i64x8(a, b),
        [
            1_i64, -4_i64, 12_i64, -32_i64, 5_i64, -12_i64, 28_i64, -64_i64
        ]
    );
}

#[simd_test]
fn shlv_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let b = u64x2::from_slice(simd, &[0_u64, 1_u64]);
    assert_eq!(*simd.shlv_u64x2(a, b), [1_u64, 4_u64]);
}

#[simd_test]
fn shlv_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let b = u64x4::from_slice(simd, &[0_u64, 1_u64, 2_u64, 3_u64]);
    assert_eq!(*simd.shlv_u64x4(a, b), [1_u64, 4_u64, 12_u64, 32_u64]);
}

#[simd_test]
fn shlv_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let b = u64x8::from_slice(
        simd,
        &[0_u64, 1_u64, 2_u64, 3_u64, 0_u64, 1_u64, 2_u64, 3_u64],
    );
    assert_eq!(
        *simd.shlv_u64x8(a, b),
        [1_u64, 4_u64, 12_u64, 32_u64, 5_u64, 12_u64, 28_u64, 64_u64]
    );
}
