// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn splat_f32x4<S: Simd>(simd: S) {
    let a = f32x4::splat(simd, 4.2);
    assert_eq!(*a, [4.2, 4.2, 4.2, 4.2]);
}

#[simd_test]
fn mask_trait_splat_mask32x4<S: Simd>(simd: S) {
    let t = mask32x4::splat(simd, true);
    assert_eq!(<[i32; 4]>::from(t), [-1; 4]);

    let f = mask32x4::splat(simd, false);
    assert_eq!(<[i32; 4]>::from(f), [0; 4]);
}

#[simd_test]
fn splat_native_mask<S: Simd>(simd: S) {
    let all_true = S::mask32s::splat(simd, true);
    assert!(all_true.all_true());

    let all_false = S::mask32s::splat(simd, false);
    assert!(all_false.all_false());
}

#[simd_test]
fn splat_f32x8<S: Simd>(simd: S) {
    let a = f32x8::splat(simd, 4.2);
    assert_eq!(*a, [4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2]);
}

#[simd_test]
fn splat_f32x16<S: Simd>(simd: S) {
    let a = f32x16::splat(simd, 4.2);
    assert_eq!(
        *a,
        [
            4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2
        ]
    );
}

#[simd_test]
fn splat_f64x8<S: Simd>(simd: S) {
    let a = f64x8::splat(simd, 4.2);
    assert_eq!(*a, [4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2]);
}

// Additional concrete rows for this operation.

#[simd_test]
fn splat_i64x2<S: Simd>(simd: S) {
    let a = i64x2::splat(simd, 1_i64);
    assert_eq!(*a, [1_i64, 1_i64]);
}

#[simd_test]
fn splat_i64x4<S: Simd>(simd: S) {
    let a = i64x4::splat(simd, 1_i64);
    assert_eq!(*a, [1_i64, 1_i64, 1_i64, 1_i64]);
}

#[simd_test]
fn splat_i64x8<S: Simd>(simd: S) {
    let a = i64x8::splat(simd, 1_i64);
    assert_eq!(*a, [1_i64, 1_i64, 1_i64, 1_i64, 1_i64, 1_i64, 1_i64, 1_i64]);
}

#[simd_test]
fn splat_u64x2<S: Simd>(simd: S) {
    let a = u64x2::splat(simd, 1_u64);
    assert_eq!(*a, [1_u64, 1_u64]);
}

#[simd_test]
fn splat_u64x4<S: Simd>(simd: S) {
    let a = u64x4::splat(simd, 1_u64);
    assert_eq!(*a, [1_u64, 1_u64, 1_u64, 1_u64]);
}

#[simd_test]
fn splat_u64x8<S: Simd>(simd: S) {
    let a = u64x8::splat(simd, 1_u64);
    assert_eq!(*a, [1_u64, 1_u64, 1_u64, 1_u64, 1_u64, 1_u64, 1_u64, 1_u64]);
}

#[simd_test]
fn splat_mask64x2<S: Simd>(simd: S) {
    let a = simd.splat_mask64x2(true);
    assert_eq!(<[i64; 2]>::from(a), [-1; 2]);
}

#[simd_test]
fn splat_mask64x4<S: Simd>(simd: S) {
    let a = simd.splat_mask64x4(true);
    assert_eq!(<[i64; 4]>::from(a), [-1; 4]);
}

#[simd_test]
fn splat_mask64x8<S: Simd>(simd: S) {
    let a = simd.splat_mask64x8(true);
    assert_eq!(<[i64; 8]>::from(a), [-1; 8]);
}

// Generated gap-fill coverage rows.

#[simd_test]
fn splat_i8x16<S: Simd>(simd: S) {
    let result = simd.splat_i8x16(3_i8);
    let expected: [i8; 16] = [3_i8; 16];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_u8x16<S: Simd>(simd: S) {
    let result = simd.splat_u8x16(3_u8);
    let expected: [u8; 16] = [3_u8; 16];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_i8x32<S: Simd>(simd: S) {
    let result = simd.splat_i8x32(3_i8);
    let expected: [i8; 32] = [3_i8; 32];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_u8x32<S: Simd>(simd: S) {
    let result = simd.splat_u8x32(3_u8);
    let expected: [u8; 32] = [3_u8; 32];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_i8x64<S: Simd>(simd: S) {
    let result = simd.splat_i8x64(3_i8);
    let expected: [i8; 64] = [3_i8; 64];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_u8x64<S: Simd>(simd: S) {
    let result = simd.splat_u8x64(3_u8);
    let expected: [u8; 64] = [3_u8; 64];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_i16x8<S: Simd>(simd: S) {
    let result = simd.splat_i16x8(3_i16);
    let expected: [i16; 8] = [3_i16; 8];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_u16x8<S: Simd>(simd: S) {
    let result = simd.splat_u16x8(3_u16);
    let expected: [u16; 8] = [3_u16; 8];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_i16x16<S: Simd>(simd: S) {
    let result = simd.splat_i16x16(3_i16);
    let expected: [i16; 16] = [3_i16; 16];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_u16x16<S: Simd>(simd: S) {
    let result = simd.splat_u16x16(3_u16);
    let expected: [u16; 16] = [3_u16; 16];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_i16x32<S: Simd>(simd: S) {
    let result = simd.splat_i16x32(3_i16);
    let expected: [i16; 32] = [3_i16; 32];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_u16x32<S: Simd>(simd: S) {
    let result = simd.splat_u16x32(3_u16);
    let expected: [u16; 32] = [3_u16; 32];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_i32x4<S: Simd>(simd: S) {
    let result = simd.splat_i32x4(3_i32);
    let expected: [i32; 4] = [3_i32; 4];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_u32x4<S: Simd>(simd: S) {
    let result = simd.splat_u32x4(3_u32);
    let expected: [u32; 4] = [3_u32; 4];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_i32x8<S: Simd>(simd: S) {
    let result = simd.splat_i32x8(3_i32);
    let expected: [i32; 8] = [3_i32; 8];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_u32x8<S: Simd>(simd: S) {
    let result = simd.splat_u32x8(3_u32);
    let expected: [u32; 8] = [3_u32; 8];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_i32x16<S: Simd>(simd: S) {
    let result = simd.splat_i32x16(3_i32);
    let expected: [i32; 16] = [3_i32; 16];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_u32x16<S: Simd>(simd: S) {
    let result = simd.splat_u32x16(3_u32);
    let expected: [u32; 16] = [3_u32; 16];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_f64x2<S: Simd>(simd: S) {
    let result = simd.splat_f64x2(3.5_f64);
    let expected: [f64; 2] = [3.5_f64; 2];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_f64x4<S: Simd>(simd: S) {
    let result = simd.splat_f64x4(3.5_f64);
    let expected: [f64; 4] = [3.5_f64; 4];
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn splat_mask8x16<S: Simd>(simd: S) {
    let result = simd.splat_mask8x16(true);
    let expected: [i8; 16] = [-1_i8; 16];
    assert_eq!(<[i8; 16]>::from(result), expected);
}

#[simd_test]
fn splat_mask8x32<S: Simd>(simd: S) {
    let result = simd.splat_mask8x32(true);
    let expected: [i8; 32] = [-1_i8; 32];
    assert_eq!(<[i8; 32]>::from(result), expected);
}

#[simd_test]
fn splat_mask8x64<S: Simd>(simd: S) {
    let result = simd.splat_mask8x64(true);
    let expected: [i8; 64] = [-1_i8; 64];
    assert_eq!(<[i8; 64]>::from(result), expected);
}

#[simd_test]
fn splat_mask16x8<S: Simd>(simd: S) {
    let result = simd.splat_mask16x8(true);
    let expected: [i16; 8] = [-1_i16; 8];
    assert_eq!(<[i16; 8]>::from(result), expected);
}

#[simd_test]
fn splat_mask16x16<S: Simd>(simd: S) {
    let result = simd.splat_mask16x16(true);
    let expected: [i16; 16] = [-1_i16; 16];
    assert_eq!(<[i16; 16]>::from(result), expected);
}

#[simd_test]
fn splat_mask16x32<S: Simd>(simd: S) {
    let result = simd.splat_mask16x32(true);
    let expected: [i16; 32] = [-1_i16; 32];
    assert_eq!(<[i16; 32]>::from(result), expected);
}

#[simd_test]
fn splat_mask32x8<S: Simd>(simd: S) {
    let result = simd.splat_mask32x8(true);
    let expected: [i32; 8] = [-1_i32; 8];
    assert_eq!(<[i32; 8]>::from(result), expected);
}

#[simd_test]
fn splat_mask32x16<S: Simd>(simd: S) {
    let result = simd.splat_mask32x16(true);
    let expected: [i32; 16] = [-1_i32; 16];
    assert_eq!(<[i32; 16]>::from(result), expected);
}
