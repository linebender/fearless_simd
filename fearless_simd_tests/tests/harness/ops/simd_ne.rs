// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

const I8_MIN: i8 = i8::MIN;
const I8_MAX: i8 = i8::MAX;
const U8_MIN: u8 = u8::MIN;
const U8_MAX: u8 = u8::MAX;

const I16_MIN: i16 = i16::MIN;
const I16_MAX: i16 = i16::MAX;
const U16_MIN: u16 = u16::MIN;
const U16_MAX: u16 = u16::MAX;

const I32_MIN: i32 = i32::MIN;
const I32_MAX: i32 = i32::MAX;
const U32_MIN: u32 = u32::MIN;
const U32_MAX: u32 = u32::MAX;

const I64_MIN: i64 = i64::MIN;
const I64_MAX: i64 = i64::MAX;
const U64_MIN: u64 = u64::MIN;
const U64_MAX: u64 = u64::MAX;

#[simd_test]
fn simd_ne_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i8; 16]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i8; 16]>::from(simd.simd_ne_i8x16(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i8; 16]>::from(a.simd_ne(a)), [0; 16]);
    let min = i8x16::from_slice(simd, &[I8_MIN; 16]);
    let max = i8x16::from_slice(simd, &[I8_MAX; 16]);
    assert_eq!(<[i8; 16]>::from(min.simd_ne(I8_MIN)), [0; 16]);
    assert_eq!(<[i8; 16]>::from(max.simd_ne(I8_MIN)), [-1; 16]);
}

#[simd_test]
fn simd_ne_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[
            U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1,
        ],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i8; 16]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i8; 16]>::from(simd.simd_ne_u8x16(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i8; 16]>::from(a.simd_ne(a)), [0; 16]);
    let min = u8x16::from_slice(simd, &[U8_MIN; 16]);
    let max = u8x16::from_slice(simd, &[U8_MAX; 16]);
    assert_eq!(<[i8; 16]>::from(min.simd_ne(U8_MIN)), [0; 16]);
    assert_eq!(<[i8; 16]>::from(max.simd_ne(U8_MIN)), [-1; 16]);
}

#[simd_test]
fn simd_ne_mask8x16<S: Simd>(simd: S) {
    let a = mask8x16::from_slice(
        simd,
        &[0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    );
    let b = mask8x16::from_slice(
        simd,
        &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    assert_eq!(<[i8; 16]>::from(a.simd_ne(b)), [-1; 16]);
    assert_eq!(<[i8; 16]>::from(simd.simd_ne_mask8x16(a, b)), [-1; 16]);
    assert_eq!(<[i8; 16]>::from(a.simd_ne(a)), [0; 16]);
    let all_false = mask8x16::from_slice(simd, &[0; 16]);
    assert_eq!(
        <[i8; 16]>::from(a.simd_ne(all_false)),
        [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]
    );
    assert_eq!(a.simd_ne(b).to_bitmask(), 65535_u64);
}

#[simd_test]
fn simd_ne_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[I16_MIN, I16_MAX, 0, 1, I16_MIN, I16_MAX, 0, 1]);
    let b = i16x8::from_slice(simd, &[I16_MAX, I16_MAX, 1, 0, I16_MAX, I16_MAX, 1, 0]);
    assert_eq!(
        <[i16; 8]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i16; 8]>::from(simd.simd_ne_i16x8(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i16; 8]>::from(a.simd_ne(a)), [0; 8]);
    let min = i16x8::from_slice(simd, &[I16_MIN; 8]);
    let max = i16x8::from_slice(simd, &[I16_MAX; 8]);
    assert_eq!(<[i16; 8]>::from(min.simd_ne(I16_MIN)), [0; 8]);
    assert_eq!(<[i16; 8]>::from(max.simd_ne(I16_MIN)), [-1; 8]);
}

#[simd_test]
fn simd_ne_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[U16_MIN, U16_MAX, 0, 1, U16_MIN, U16_MAX, 0, 1]);
    let b = u16x8::from_slice(simd, &[U16_MAX, U16_MAX, 1, 0, U16_MAX, U16_MAX, 1, 0]);
    assert_eq!(
        <[i16; 8]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i16; 8]>::from(simd.simd_ne_u16x8(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i16; 8]>::from(a.simd_ne(a)), [0; 8]);
    let min = u16x8::from_slice(simd, &[U16_MIN; 8]);
    let max = u16x8::from_slice(simd, &[U16_MAX; 8]);
    assert_eq!(<[i16; 8]>::from(min.simd_ne(U16_MIN)), [0; 8]);
    assert_eq!(<[i16; 8]>::from(max.simd_ne(U16_MIN)), [-1; 8]);
}

#[simd_test]
fn simd_ne_mask16x8<S: Simd>(simd: S) {
    let a = mask16x8::from_slice(simd, &[0, -1, 0, -1, 0, -1, 0, -1]);
    let b = mask16x8::from_slice(simd, &[-1, 0, -1, 0, -1, 0, -1, 0]);
    assert_eq!(<[i16; 8]>::from(a.simd_ne(b)), [-1; 8]);
    assert_eq!(<[i16; 8]>::from(simd.simd_ne_mask16x8(a, b)), [-1; 8]);
    assert_eq!(<[i16; 8]>::from(a.simd_ne(a)), [0; 8]);
    let all_false = mask16x8::from_slice(simd, &[0; 8]);
    assert_eq!(
        <[i16; 8]>::from(a.simd_ne(all_false)),
        [0, -1, 0, -1, 0, -1, 0, -1]
    );
    assert_eq!(a.simd_ne(b).to_bitmask(), 255_u64);
}

#[simd_test]
fn simd_ne_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[I32_MIN, I32_MAX, 0, 1]);
    let b = i32x4::from_slice(simd, &[I32_MAX, I32_MAX, 1, 0]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(b)), [-1, 0, -1, -1]);
    assert_eq!(<[i32; 4]>::from(simd.simd_ne_i32x4(a, b)), [-1, 0, -1, -1]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(a)), [0; 4]);
    let min = i32x4::from_slice(simd, &[I32_MIN; 4]);
    let max = i32x4::from_slice(simd, &[I32_MAX; 4]);
    assert_eq!(<[i32; 4]>::from(min.simd_ne(I32_MIN)), [0; 4]);
    assert_eq!(<[i32; 4]>::from(max.simd_ne(I32_MIN)), [-1; 4]);
}

#[simd_test]
fn simd_ne_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[U32_MIN, U32_MAX, 0, 1]);
    let b = u32x4::from_slice(simd, &[U32_MAX, U32_MAX, 1, 0]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(b)), [-1, 0, -1, -1]);
    assert_eq!(<[i32; 4]>::from(simd.simd_ne_u32x4(a, b)), [-1, 0, -1, -1]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(a)), [0; 4]);
    let min = u32x4::from_slice(simd, &[U32_MIN; 4]);
    let max = u32x4::from_slice(simd, &[U32_MAX; 4]);
    assert_eq!(<[i32; 4]>::from(min.simd_ne(U32_MIN)), [0; 4]);
    assert_eq!(<[i32; 4]>::from(max.simd_ne(U32_MIN)), [-1; 4]);
}

#[simd_test]
fn simd_ne_mask32x4<S: Simd>(simd: S) {
    let a = mask32x4::from_slice(simd, &[0, -1, 0, -1]);
    let b = mask32x4::from_slice(simd, &[-1, 0, -1, 0]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(b)), [-1; 4]);
    assert_eq!(<[i32; 4]>::from(simd.simd_ne_mask32x4(a, b)), [-1; 4]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(a)), [0; 4]);
    let all_false = mask32x4::from_slice(simd, &[0; 4]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(all_false)), [0, -1, 0, -1]);
    assert_eq!(a.simd_ne(b).to_bitmask(), 15_u64);
}

#[simd_test]
fn simd_ne_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[I64_MIN, I64_MAX]);
    let b = i64x2::from_slice(simd, &[I64_MAX, I64_MAX]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(b)), [-1, 0]);
    assert_eq!(<[i64; 2]>::from(simd.simd_ne_i64x2(a, b)), [-1, 0]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(a)), [0; 2]);
    let min = i64x2::from_slice(simd, &[I64_MIN; 2]);
    let max = i64x2::from_slice(simd, &[I64_MAX; 2]);
    assert_eq!(<[i64; 2]>::from(min.simd_ne(I64_MIN)), [0; 2]);
    assert_eq!(<[i64; 2]>::from(max.simd_ne(I64_MIN)), [-1; 2]);
}

#[simd_test]
fn simd_ne_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[U64_MIN, U64_MAX]);
    let b = u64x2::from_slice(simd, &[U64_MAX, U64_MAX]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(b)), [-1, 0]);
    assert_eq!(<[i64; 2]>::from(simd.simd_ne_u64x2(a, b)), [-1, 0]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(a)), [0; 2]);
    let min = u64x2::from_slice(simd, &[U64_MIN; 2]);
    let max = u64x2::from_slice(simd, &[U64_MAX; 2]);
    assert_eq!(<[i64; 2]>::from(min.simd_ne(U64_MIN)), [0; 2]);
    assert_eq!(<[i64; 2]>::from(max.simd_ne(U64_MIN)), [-1; 2]);
}

#[simd_test]
fn simd_ne_mask64x2<S: Simd>(simd: S) {
    let a = mask64x2::from_slice(simd, &[0, -1]);
    let b = mask64x2::from_slice(simd, &[-1, 0]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(b)), [-1; 2]);
    assert_eq!(<[i64; 2]>::from(simd.simd_ne_mask64x2(a, b)), [-1; 2]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(a)), [0; 2]);
    let all_false = mask64x2::from_slice(simd, &[0; 2]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(all_false)), [0, -1]);
    assert_eq!(a.simd_ne(b).to_bitmask(), 3_u64);
}

#[simd_test]
fn simd_ne_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, -3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[1.0, -2.0, 3.0, 4.0]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(b)), [0, -1, -1, 0]);
    assert_eq!(<[i32; 4]>::from(simd.simd_ne_f32x4(a, b)), [0, -1, -1, 0]);

    let a = f32x4::from_slice(simd, &[0.0, f32::INFINITY, -0.0, f32::NEG_INFINITY]);
    let b = f32x4::from_slice(simd, &[-0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(b)), [0, -1, 0, -1]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(a)), [0; 4]);
    let zeros = f32x4::from_slice(simd, &[0.0; 4]);
    assert_eq!(<[i32; 4]>::from(zeros.simd_ne(-0.0)), [0; 4]);

    let a = f32x4::from_slice(simd, &[f32::NAN, 1.0, f32::NAN, 1.0]);
    let b = f32x4::from_slice(simd, &[1.0, f32::NAN, f32::NAN, 1.0]);
    assert_eq!(<[i32; 4]>::from(a.simd_ne(b)), [-1, -1, -1, 0]);
    let nan = f32x4::from_slice(simd, &[f32::NAN; 4]);
    assert_eq!(<[i32; 4]>::from(nan.simd_ne(nan)), [-1; 4]);
    assert_eq!(<[i32; 4]>::from(nan.simd_ne(1.0)), [-1; 4]);
    assert_eq!(<[i32; 4]>::from(zeros.simd_ne(f32::NAN)), [-1; 4]);

    let snan = f32x4::from_slice(simd, &[f32::from_bits(0x7f80_0001); 4]);
    assert_eq!(<[i32; 4]>::from(snan.simd_ne(snan)), [-1; 4]);
    assert_eq!(<[i32; 4]>::from(snan.simd_ne(0.0)), [-1; 4]);
    assert_eq!(<[i32; 4]>::from(zeros.simd_ne(snan)), [-1; 4]);
}

#[simd_test]
fn simd_ne_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.0, 2.0]);
    let b = f64x2::from_slice(simd, &[1.0, -2.0]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(b)), [0, -1]);
    assert_eq!(<[i64; 2]>::from(simd.simd_ne_f64x2(a, b)), [0, -1]);

    let a = f64x2::from_slice(simd, &[0.0, f64::INFINITY]);
    let b = f64x2::from_slice(simd, &[-0.0, f64::NEG_INFINITY]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(b)), [0, -1]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(a)), [0; 2]);
    let zeros = f64x2::from_slice(simd, &[0.0; 2]);
    assert_eq!(<[i64; 2]>::from(zeros.simd_ne(-0.0)), [0; 2]);

    let a = f64x2::from_slice(simd, &[f64::NAN, 1.0]);
    let b = f64x2::from_slice(simd, &[1.0, f64::NAN]);
    assert_eq!(<[i64; 2]>::from(a.simd_ne(b)), [-1, -1]);
    let nan = f64x2::from_slice(simd, &[f64::NAN; 2]);
    assert_eq!(<[i64; 2]>::from(nan.simd_ne(nan)), [-1; 2]);
    assert_eq!(<[i64; 2]>::from(nan.simd_ne(1.0)), [-1; 2]);
    assert_eq!(<[i64; 2]>::from(zeros.simd_ne(f64::NAN)), [-1; 2]);

    let snan = f64x2::from_slice(simd, &[f64::from_bits(0x7ff0_0000_0000_0001); 2]);
    assert_eq!(<[i64; 2]>::from(snan.simd_ne(snan)), [-1; 2]);
    assert_eq!(<[i64; 2]>::from(snan.simd_ne(0.0)), [-1; 2]);
    assert_eq!(<[i64; 2]>::from(zeros.simd_ne(snan)), [-1; 2]);
}

#[simd_test]
fn simd_ne_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1,
            I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0,
            I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i8; 32]>::from(a.simd_ne(b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(
        <[i8; 32]>::from(simd.simd_ne_i8x32(a, b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(<[i8; 32]>::from(a.simd_ne(a)), [0; 32]);
    let min = i8x32::from_slice(simd, &[I8_MIN; 32]);
    let max = i8x32::from_slice(simd, &[I8_MAX; 32]);
    assert_eq!(<[i8; 32]>::from(min.simd_ne(I8_MIN)), [0; 32]);
    assert_eq!(<[i8; 32]>::from(max.simd_ne(I8_MIN)), [-1; 32]);
}

#[simd_test]
fn simd_ne_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1,
            U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0,
            U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i8; 32]>::from(a.simd_ne(b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(
        <[i8; 32]>::from(simd.simd_ne_u8x32(a, b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(<[i8; 32]>::from(a.simd_ne(a)), [0; 32]);
    let min = u8x32::from_slice(simd, &[U8_MIN; 32]);
    let max = u8x32::from_slice(simd, &[U8_MAX; 32]);
    assert_eq!(<[i8; 32]>::from(min.simd_ne(U8_MIN)), [0; 32]);
    assert_eq!(<[i8; 32]>::from(max.simd_ne(U8_MIN)), [-1; 32]);
}

#[simd_test]
fn simd_ne_mask8x32<S: Simd>(simd: S) {
    let a = mask8x32::from_slice(
        simd,
        &[
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1,
        ],
    );
    let b = mask8x32::from_slice(
        simd,
        &[
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0,
        ],
    );
    assert_eq!(<[i8; 32]>::from(a.simd_ne(b)), [-1; 32]);
    assert_eq!(<[i8; 32]>::from(simd.simd_ne_mask8x32(a, b)), [-1; 32]);
    assert_eq!(<[i8; 32]>::from(a.simd_ne(a)), [0; 32]);
    let all_false = mask8x32::from_slice(simd, &[0; 32]);
    assert_eq!(
        <[i8; 32]>::from(a.simd_ne(all_false)),
        [
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1
        ]
    );
    assert_eq!(a.simd_ne(b).to_bitmask(), 4294967295_u64);
}

#[simd_test]
fn simd_ne_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            I16_MIN, I16_MAX, 0, 1, I16_MIN, I16_MAX, 0, 1, I16_MIN, I16_MAX, 0, 1, I16_MIN,
            I16_MAX, 0, 1,
        ],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            I16_MAX, I16_MAX, 1, 0, I16_MAX, I16_MAX, 1, 0, I16_MAX, I16_MAX, 1, 0, I16_MAX,
            I16_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i16; 16]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i16; 16]>::from(simd.simd_ne_i16x16(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i16; 16]>::from(a.simd_ne(a)), [0; 16]);
    let min = i16x16::from_slice(simd, &[I16_MIN; 16]);
    let max = i16x16::from_slice(simd, &[I16_MAX; 16]);
    assert_eq!(<[i16; 16]>::from(min.simd_ne(I16_MIN)), [0; 16]);
    assert_eq!(<[i16; 16]>::from(max.simd_ne(I16_MIN)), [-1; 16]);
}

#[simd_test]
fn simd_ne_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[
            U16_MIN, U16_MAX, 0, 1, U16_MIN, U16_MAX, 0, 1, U16_MIN, U16_MAX, 0, 1, U16_MIN,
            U16_MAX, 0, 1,
        ],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            U16_MAX, U16_MAX, 1, 0, U16_MAX, U16_MAX, 1, 0, U16_MAX, U16_MAX, 1, 0, U16_MAX,
            U16_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i16; 16]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i16; 16]>::from(simd.simd_ne_u16x16(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i16; 16]>::from(a.simd_ne(a)), [0; 16]);
    let min = u16x16::from_slice(simd, &[U16_MIN; 16]);
    let max = u16x16::from_slice(simd, &[U16_MAX; 16]);
    assert_eq!(<[i16; 16]>::from(min.simd_ne(U16_MIN)), [0; 16]);
    assert_eq!(<[i16; 16]>::from(max.simd_ne(U16_MIN)), [-1; 16]);
}

#[simd_test]
fn simd_ne_mask16x16<S: Simd>(simd: S) {
    let a = mask16x16::from_slice(
        simd,
        &[0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    );
    let b = mask16x16::from_slice(
        simd,
        &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    assert_eq!(<[i16; 16]>::from(a.simd_ne(b)), [-1; 16]);
    assert_eq!(<[i16; 16]>::from(simd.simd_ne_mask16x16(a, b)), [-1; 16]);
    assert_eq!(<[i16; 16]>::from(a.simd_ne(a)), [0; 16]);
    let all_false = mask16x16::from_slice(simd, &[0; 16]);
    assert_eq!(
        <[i16; 16]>::from(a.simd_ne(all_false)),
        [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]
    );
    assert_eq!(a.simd_ne(b).to_bitmask(), 65535_u64);
}

#[simd_test]
fn simd_ne_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[I32_MIN, I32_MAX, 0, 1, I32_MIN, I32_MAX, 0, 1]);
    let b = i32x8::from_slice(simd, &[I32_MAX, I32_MAX, 1, 0, I32_MAX, I32_MAX, 1, 0]);
    assert_eq!(
        <[i32; 8]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i32; 8]>::from(simd.simd_ne_i32x8(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i32; 8]>::from(a.simd_ne(a)), [0; 8]);
    let min = i32x8::from_slice(simd, &[I32_MIN; 8]);
    let max = i32x8::from_slice(simd, &[I32_MAX; 8]);
    assert_eq!(<[i32; 8]>::from(min.simd_ne(I32_MIN)), [0; 8]);
    assert_eq!(<[i32; 8]>::from(max.simd_ne(I32_MIN)), [-1; 8]);
}

#[simd_test]
fn simd_ne_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[U32_MIN, U32_MAX, 0, 1, U32_MIN, U32_MAX, 0, 1]);
    let b = u32x8::from_slice(simd, &[U32_MAX, U32_MAX, 1, 0, U32_MAX, U32_MAX, 1, 0]);
    assert_eq!(
        <[i32; 8]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i32; 8]>::from(simd.simd_ne_u32x8(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i32; 8]>::from(a.simd_ne(a)), [0; 8]);
    let min = u32x8::from_slice(simd, &[U32_MIN; 8]);
    let max = u32x8::from_slice(simd, &[U32_MAX; 8]);
    assert_eq!(<[i32; 8]>::from(min.simd_ne(U32_MIN)), [0; 8]);
    assert_eq!(<[i32; 8]>::from(max.simd_ne(U32_MIN)), [-1; 8]);
}

#[simd_test]
fn simd_ne_mask32x8<S: Simd>(simd: S) {
    let a = mask32x8::from_slice(simd, &[0, -1, 0, -1, 0, -1, 0, -1]);
    let b = mask32x8::from_slice(simd, &[-1, 0, -1, 0, -1, 0, -1, 0]);
    assert_eq!(<[i32; 8]>::from(a.simd_ne(b)), [-1; 8]);
    assert_eq!(<[i32; 8]>::from(simd.simd_ne_mask32x8(a, b)), [-1; 8]);
    assert_eq!(<[i32; 8]>::from(a.simd_ne(a)), [0; 8]);
    let all_false = mask32x8::from_slice(simd, &[0; 8]);
    assert_eq!(
        <[i32; 8]>::from(a.simd_ne(all_false)),
        [0, -1, 0, -1, 0, -1, 0, -1]
    );
    assert_eq!(a.simd_ne(b).to_bitmask(), 255_u64);
}

#[simd_test]
fn simd_ne_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[I64_MIN, I64_MAX, 0, 1]);
    let b = i64x4::from_slice(simd, &[I64_MAX, I64_MAX, 1, 0]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(b)), [-1, 0, -1, -1]);
    assert_eq!(<[i64; 4]>::from(simd.simd_ne_i64x4(a, b)), [-1, 0, -1, -1]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(a)), [0; 4]);
    let min = i64x4::from_slice(simd, &[I64_MIN; 4]);
    let max = i64x4::from_slice(simd, &[I64_MAX; 4]);
    assert_eq!(<[i64; 4]>::from(min.simd_ne(I64_MIN)), [0; 4]);
    assert_eq!(<[i64; 4]>::from(max.simd_ne(I64_MIN)), [-1; 4]);
}

#[simd_test]
fn simd_ne_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[U64_MIN, U64_MAX, 0, 1]);
    let b = u64x4::from_slice(simd, &[U64_MAX, U64_MAX, 1, 0]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(b)), [-1, 0, -1, -1]);
    assert_eq!(<[i64; 4]>::from(simd.simd_ne_u64x4(a, b)), [-1, 0, -1, -1]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(a)), [0; 4]);
    let min = u64x4::from_slice(simd, &[U64_MIN; 4]);
    let max = u64x4::from_slice(simd, &[U64_MAX; 4]);
    assert_eq!(<[i64; 4]>::from(min.simd_ne(U64_MIN)), [0; 4]);
    assert_eq!(<[i64; 4]>::from(max.simd_ne(U64_MIN)), [-1; 4]);
}

#[simd_test]
fn simd_ne_mask64x4<S: Simd>(simd: S) {
    let a = mask64x4::from_slice(simd, &[0, -1, 0, -1]);
    let b = mask64x4::from_slice(simd, &[-1, 0, -1, 0]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(b)), [-1; 4]);
    assert_eq!(<[i64; 4]>::from(simd.simd_ne_mask64x4(a, b)), [-1; 4]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(a)), [0; 4]);
    let all_false = mask64x4::from_slice(simd, &[0; 4]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(all_false)), [0, -1, 0, -1]);
    assert_eq!(a.simd_ne(b).to_bitmask(), 15_u64);
}

#[simd_test]
fn simd_ne_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, -3.0, 4.0, 1.0, 2.0, -3.0, 4.0]);
    let b = f32x8::from_slice(simd, &[1.0, -2.0, 3.0, 4.0, 1.0, -2.0, 3.0, 4.0]);
    assert_eq!(<[i32; 8]>::from(a.simd_ne(b)), [0, -1, -1, 0, 0, -1, -1, 0]);
    assert_eq!(
        <[i32; 8]>::from(simd.simd_ne_f32x8(a, b)),
        [0, -1, -1, 0, 0, -1, -1, 0]
    );

    let a = f32x8::from_slice(
        simd,
        &[
            0.0,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
        ],
    );
    let b = f32x8::from_slice(
        simd,
        &[
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
        ],
    );
    assert_eq!(<[i32; 8]>::from(a.simd_ne(b)), [0, -1, 0, -1, 0, -1, 0, -1]);
    assert_eq!(<[i32; 8]>::from(a.simd_ne(a)), [0; 8]);
    let zeros = f32x8::from_slice(simd, &[0.0; 8]);
    assert_eq!(<[i32; 8]>::from(zeros.simd_ne(-0.0)), [0; 8]);

    let a = f32x8::from_slice(
        simd,
        &[f32::NAN, 1.0, f32::NAN, 1.0, f32::NAN, 1.0, f32::NAN, 1.0],
    );
    let b = f32x8::from_slice(
        simd,
        &[1.0, f32::NAN, f32::NAN, 1.0, 1.0, f32::NAN, f32::NAN, 1.0],
    );
    assert_eq!(
        <[i32; 8]>::from(a.simd_ne(b)),
        [-1, -1, -1, 0, -1, -1, -1, 0]
    );
    let nan = f32x8::from_slice(simd, &[f32::NAN; 8]);
    assert_eq!(<[i32; 8]>::from(nan.simd_ne(nan)), [-1; 8]);
    assert_eq!(<[i32; 8]>::from(nan.simd_ne(1.0)), [-1; 8]);
    assert_eq!(<[i32; 8]>::from(zeros.simd_ne(f32::NAN)), [-1; 8]);

    let snan = f32x8::from_slice(simd, &[f32::from_bits(0x7f80_0001); 8]);
    assert_eq!(<[i32; 8]>::from(snan.simd_ne(snan)), [-1; 8]);
    assert_eq!(<[i32; 8]>::from(snan.simd_ne(0.0)), [-1; 8]);
    assert_eq!(<[i32; 8]>::from(zeros.simd_ne(snan)), [-1; 8]);
}

#[simd_test]
fn simd_ne_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, -3.0, 4.0]);
    let b = f64x4::from_slice(simd, &[1.0, -2.0, 3.0, 4.0]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(b)), [0, -1, -1, 0]);
    assert_eq!(<[i64; 4]>::from(simd.simd_ne_f64x4(a, b)), [0, -1, -1, 0]);

    let a = f64x4::from_slice(simd, &[0.0, f64::INFINITY, -0.0, f64::NEG_INFINITY]);
    let b = f64x4::from_slice(simd, &[-0.0, f64::NEG_INFINITY, 0.0, f64::INFINITY]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(b)), [0, -1, 0, -1]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(a)), [0; 4]);
    let zeros = f64x4::from_slice(simd, &[0.0; 4]);
    assert_eq!(<[i64; 4]>::from(zeros.simd_ne(-0.0)), [0; 4]);

    let a = f64x4::from_slice(simd, &[f64::NAN, 1.0, f64::NAN, 1.0]);
    let b = f64x4::from_slice(simd, &[1.0, f64::NAN, f64::NAN, 1.0]);
    assert_eq!(<[i64; 4]>::from(a.simd_ne(b)), [-1, -1, -1, 0]);
    let nan = f64x4::from_slice(simd, &[f64::NAN; 4]);
    assert_eq!(<[i64; 4]>::from(nan.simd_ne(nan)), [-1; 4]);
    assert_eq!(<[i64; 4]>::from(nan.simd_ne(1.0)), [-1; 4]);
    assert_eq!(<[i64; 4]>::from(zeros.simd_ne(f64::NAN)), [-1; 4]);

    let snan = f64x4::from_slice(simd, &[f64::from_bits(0x7ff0_0000_0000_0001); 4]);
    assert_eq!(<[i64; 4]>::from(snan.simd_ne(snan)), [-1; 4]);
    assert_eq!(<[i64; 4]>::from(snan.simd_ne(0.0)), [-1; 4]);
    assert_eq!(<[i64; 4]>::from(zeros.simd_ne(snan)), [-1; 4]);
}

#[simd_test]
fn simd_ne_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1,
            I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1,
            I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1,
            I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1, I8_MIN, I8_MAX, 0, 1,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0,
            I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0,
            I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0,
            I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0, I8_MAX, I8_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i8; 64]>::from(a.simd_ne(b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0,
            -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(
        <[i8; 64]>::from(simd.simd_ne_i8x64(a, b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0,
            -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(<[i8; 64]>::from(a.simd_ne(a)), [0; 64]);
    let min = i8x64::from_slice(simd, &[I8_MIN; 64]);
    let max = i8x64::from_slice(simd, &[I8_MAX; 64]);
    assert_eq!(<[i8; 64]>::from(min.simd_ne(I8_MIN)), [0; 64]);
    assert_eq!(<[i8; 64]>::from(max.simd_ne(I8_MIN)), [-1; 64]);
}

#[simd_test]
fn simd_ne_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1,
            U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1,
            U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1,
            U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1, U8_MIN, U8_MAX, 0, 1,
        ],
    );
    let b = u8x64::from_slice(
        simd,
        &[
            U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0,
            U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0,
            U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0,
            U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0, U8_MAX, U8_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i8; 64]>::from(a.simd_ne(b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0,
            -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(
        <[i8; 64]>::from(simd.simd_ne_u8x64(a, b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0,
            -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(<[i8; 64]>::from(a.simd_ne(a)), [0; 64]);
    let min = u8x64::from_slice(simd, &[U8_MIN; 64]);
    let max = u8x64::from_slice(simd, &[U8_MAX; 64]);
    assert_eq!(<[i8; 64]>::from(min.simd_ne(U8_MIN)), [0; 64]);
    assert_eq!(<[i8; 64]>::from(max.simd_ne(U8_MIN)), [-1; 64]);
}

#[simd_test]
fn simd_ne_mask8x64<S: Simd>(simd: S) {
    let a = mask8x64::from_slice(
        simd,
        &[
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
        ],
    );
    let b = mask8x64::from_slice(
        simd,
        &[
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
        ],
    );
    assert_eq!(<[i8; 64]>::from(a.simd_ne(b)), [-1; 64]);
    assert_eq!(<[i8; 64]>::from(simd.simd_ne_mask8x64(a, b)), [-1; 64]);
    assert_eq!(<[i8; 64]>::from(a.simd_ne(a)), [0; 64]);
    let all_false = mask8x64::from_slice(simd, &[0; 64]);
    assert_eq!(
        <[i8; 64]>::from(a.simd_ne(all_false)),
        [
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1
        ]
    );
    assert_eq!(a.simd_ne(b).to_bitmask(), 18446744073709551615_u64);
}

#[simd_test]
fn simd_ne_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            I16_MIN, I16_MAX, 0, 1, I16_MIN, I16_MAX, 0, 1, I16_MIN, I16_MAX, 0, 1, I16_MIN,
            I16_MAX, 0, 1, I16_MIN, I16_MAX, 0, 1, I16_MIN, I16_MAX, 0, 1, I16_MIN, I16_MAX, 0, 1,
            I16_MIN, I16_MAX, 0, 1,
        ],
    );
    let b = i16x32::from_slice(
        simd,
        &[
            I16_MAX, I16_MAX, 1, 0, I16_MAX, I16_MAX, 1, 0, I16_MAX, I16_MAX, 1, 0, I16_MAX,
            I16_MAX, 1, 0, I16_MAX, I16_MAX, 1, 0, I16_MAX, I16_MAX, 1, 0, I16_MAX, I16_MAX, 1, 0,
            I16_MAX, I16_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i16; 32]>::from(a.simd_ne(b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(
        <[i16; 32]>::from(simd.simd_ne_i16x32(a, b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(<[i16; 32]>::from(a.simd_ne(a)), [0; 32]);
    let min = i16x32::from_slice(simd, &[I16_MIN; 32]);
    let max = i16x32::from_slice(simd, &[I16_MAX; 32]);
    assert_eq!(<[i16; 32]>::from(min.simd_ne(I16_MIN)), [0; 32]);
    assert_eq!(<[i16; 32]>::from(max.simd_ne(I16_MIN)), [-1; 32]);
}

#[simd_test]
fn simd_ne_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            U16_MIN, U16_MAX, 0, 1, U16_MIN, U16_MAX, 0, 1, U16_MIN, U16_MAX, 0, 1, U16_MIN,
            U16_MAX, 0, 1, U16_MIN, U16_MAX, 0, 1, U16_MIN, U16_MAX, 0, 1, U16_MIN, U16_MAX, 0, 1,
            U16_MIN, U16_MAX, 0, 1,
        ],
    );
    let b = u16x32::from_slice(
        simd,
        &[
            U16_MAX, U16_MAX, 1, 0, U16_MAX, U16_MAX, 1, 0, U16_MAX, U16_MAX, 1, 0, U16_MAX,
            U16_MAX, 1, 0, U16_MAX, U16_MAX, 1, 0, U16_MAX, U16_MAX, 1, 0, U16_MAX, U16_MAX, 1, 0,
            U16_MAX, U16_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i16; 32]>::from(a.simd_ne(b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(
        <[i16; 32]>::from(simd.simd_ne_u16x32(a, b)),
        [
            -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1,
            -1, -1, 0, -1, -1, -1, 0, -1, -1
        ]
    );
    assert_eq!(<[i16; 32]>::from(a.simd_ne(a)), [0; 32]);
    let min = u16x32::from_slice(simd, &[U16_MIN; 32]);
    let max = u16x32::from_slice(simd, &[U16_MAX; 32]);
    assert_eq!(<[i16; 32]>::from(min.simd_ne(U16_MIN)), [0; 32]);
    assert_eq!(<[i16; 32]>::from(max.simd_ne(U16_MIN)), [-1; 32]);
}

#[simd_test]
fn simd_ne_mask16x32<S: Simd>(simd: S) {
    let a = mask16x32::from_slice(
        simd,
        &[
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1,
        ],
    );
    let b = mask16x32::from_slice(
        simd,
        &[
            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
            0, -1, 0, -1, 0, -1, 0,
        ],
    );
    assert_eq!(<[i16; 32]>::from(a.simd_ne(b)), [-1; 32]);
    assert_eq!(<[i16; 32]>::from(simd.simd_ne_mask16x32(a, b)), [-1; 32]);
    assert_eq!(<[i16; 32]>::from(a.simd_ne(a)), [0; 32]);
    let all_false = mask16x32::from_slice(simd, &[0; 32]);
    assert_eq!(
        <[i16; 32]>::from(a.simd_ne(all_false)),
        [
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
            -1, 0, -1, 0, -1, 0, -1
        ]
    );
    assert_eq!(a.simd_ne(b).to_bitmask(), 4294967295_u64);
}

#[simd_test]
fn simd_ne_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[
            I32_MIN, I32_MAX, 0, 1, I32_MIN, I32_MAX, 0, 1, I32_MIN, I32_MAX, 0, 1, I32_MIN,
            I32_MAX, 0, 1,
        ],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            I32_MAX, I32_MAX, 1, 0, I32_MAX, I32_MAX, 1, 0, I32_MAX, I32_MAX, 1, 0, I32_MAX,
            I32_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i32; 16]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i32; 16]>::from(simd.simd_ne_i32x16(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i32; 16]>::from(a.simd_ne(a)), [0; 16]);
    let min = i32x16::from_slice(simd, &[I32_MIN; 16]);
    let max = i32x16::from_slice(simd, &[I32_MAX; 16]);
    assert_eq!(<[i32; 16]>::from(min.simd_ne(I32_MIN)), [0; 16]);
    assert_eq!(<[i32; 16]>::from(max.simd_ne(I32_MIN)), [-1; 16]);
}

#[simd_test]
fn simd_ne_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            U32_MIN, U32_MAX, 0, 1, U32_MIN, U32_MAX, 0, 1, U32_MIN, U32_MAX, 0, 1, U32_MIN,
            U32_MAX, 0, 1,
        ],
    );
    let b = u32x16::from_slice(
        simd,
        &[
            U32_MAX, U32_MAX, 1, 0, U32_MAX, U32_MAX, 1, 0, U32_MAX, U32_MAX, 1, 0, U32_MAX,
            U32_MAX, 1, 0,
        ],
    );
    assert_eq!(
        <[i32; 16]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i32; 16]>::from(simd.simd_ne_u32x16(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i32; 16]>::from(a.simd_ne(a)), [0; 16]);
    let min = u32x16::from_slice(simd, &[U32_MIN; 16]);
    let max = u32x16::from_slice(simd, &[U32_MAX; 16]);
    assert_eq!(<[i32; 16]>::from(min.simd_ne(U32_MIN)), [0; 16]);
    assert_eq!(<[i32; 16]>::from(max.simd_ne(U32_MIN)), [-1; 16]);
}

#[simd_test]
fn simd_ne_mask32x16<S: Simd>(simd: S) {
    let a = mask32x16::from_slice(
        simd,
        &[0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    );
    let b = mask32x16::from_slice(
        simd,
        &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    assert_eq!(<[i32; 16]>::from(a.simd_ne(b)), [-1; 16]);
    assert_eq!(<[i32; 16]>::from(simd.simd_ne_mask32x16(a, b)), [-1; 16]);
    assert_eq!(<[i32; 16]>::from(a.simd_ne(a)), [0; 16]);
    let all_false = mask32x16::from_slice(simd, &[0; 16]);
    assert_eq!(
        <[i32; 16]>::from(a.simd_ne(all_false)),
        [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]
    );
    assert_eq!(a.simd_ne(b).to_bitmask(), 65535_u64);
}

#[simd_test]
fn simd_ne_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(simd, &[I64_MIN, I64_MAX, 0, 1, I64_MIN, I64_MAX, 0, 1]);
    let b = i64x8::from_slice(simd, &[I64_MAX, I64_MAX, 1, 0, I64_MAX, I64_MAX, 1, 0]);
    assert_eq!(
        <[i64; 8]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i64; 8]>::from(simd.simd_ne_i64x8(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i64; 8]>::from(a.simd_ne(a)), [0; 8]);
    let min = i64x8::from_slice(simd, &[I64_MIN; 8]);
    let max = i64x8::from_slice(simd, &[I64_MAX; 8]);
    assert_eq!(<[i64; 8]>::from(min.simd_ne(I64_MIN)), [0; 8]);
    assert_eq!(<[i64; 8]>::from(max.simd_ne(I64_MIN)), [-1; 8]);
}

#[simd_test]
fn simd_ne_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(simd, &[U64_MIN, U64_MAX, 0, 1, U64_MIN, U64_MAX, 0, 1]);
    let b = u64x8::from_slice(simd, &[U64_MAX, U64_MAX, 1, 0, U64_MAX, U64_MAX, 1, 0]);
    assert_eq!(
        <[i64; 8]>::from(a.simd_ne(b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(
        <[i64; 8]>::from(simd.simd_ne_u64x8(a, b)),
        [-1, 0, -1, -1, -1, 0, -1, -1]
    );
    assert_eq!(<[i64; 8]>::from(a.simd_ne(a)), [0; 8]);
    let min = u64x8::from_slice(simd, &[U64_MIN; 8]);
    let max = u64x8::from_slice(simd, &[U64_MAX; 8]);
    assert_eq!(<[i64; 8]>::from(min.simd_ne(U64_MIN)), [0; 8]);
    assert_eq!(<[i64; 8]>::from(max.simd_ne(U64_MIN)), [-1; 8]);
}

#[simd_test]
fn simd_ne_mask64x8<S: Simd>(simd: S) {
    let a = mask64x8::from_slice(simd, &[0, -1, 0, -1, 0, -1, 0, -1]);
    let b = mask64x8::from_slice(simd, &[-1, 0, -1, 0, -1, 0, -1, 0]);
    assert_eq!(<[i64; 8]>::from(a.simd_ne(b)), [-1; 8]);
    assert_eq!(<[i64; 8]>::from(simd.simd_ne_mask64x8(a, b)), [-1; 8]);
    assert_eq!(<[i64; 8]>::from(a.simd_ne(a)), [0; 8]);
    let all_false = mask64x8::from_slice(simd, &[0; 8]);
    assert_eq!(
        <[i64; 8]>::from(a.simd_ne(all_false)),
        [0, -1, 0, -1, 0, -1, 0, -1]
    );
    assert_eq!(a.simd_ne(b).to_bitmask(), 255_u64);
}

#[simd_test]
fn simd_ne_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, -3.0, 4.0, 1.0, 2.0, -3.0, 4.0, 1.0, 2.0, -3.0, 4.0, 1.0, 2.0, -3.0, 4.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0, -2.0, 3.0, 4.0, 1.0, -2.0, 3.0, 4.0, 1.0, -2.0, 3.0, 4.0, 1.0, -2.0, 3.0, 4.0,
        ],
    );
    assert_eq!(
        <[i32; 16]>::from(a.simd_ne(b)),
        [0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0]
    );
    assert_eq!(
        <[i32; 16]>::from(simd.simd_ne_f32x16(a, b)),
        [0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0]
    );

    let a = f32x16::from_slice(
        simd,
        &[
            0.0,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
            -0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::INFINITY,
        ],
    );
    assert_eq!(
        <[i32; 16]>::from(a.simd_ne(b)),
        [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1]
    );
    assert_eq!(<[i32; 16]>::from(a.simd_ne(a)), [0; 16]);
    let zeros = f32x16::from_slice(simd, &[0.0; 16]);
    assert_eq!(<[i32; 16]>::from(zeros.simd_ne(-0.0)), [0; 16]);

    let a = f32x16::from_slice(
        simd,
        &[
            f32::NAN,
            1.0,
            f32::NAN,
            1.0,
            f32::NAN,
            1.0,
            f32::NAN,
            1.0,
            f32::NAN,
            1.0,
            f32::NAN,
            1.0,
            f32::NAN,
            1.0,
            f32::NAN,
            1.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0,
            f32::NAN,
            f32::NAN,
            1.0,
            1.0,
            f32::NAN,
            f32::NAN,
            1.0,
            1.0,
            f32::NAN,
            f32::NAN,
            1.0,
            1.0,
            f32::NAN,
            f32::NAN,
            1.0,
        ],
    );
    assert_eq!(
        <[i32; 16]>::from(a.simd_ne(b)),
        [-1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0]
    );
    let nan = f32x16::from_slice(simd, &[f32::NAN; 16]);
    assert_eq!(<[i32; 16]>::from(nan.simd_ne(nan)), [-1; 16]);
    assert_eq!(<[i32; 16]>::from(nan.simd_ne(1.0)), [-1; 16]);
    assert_eq!(<[i32; 16]>::from(zeros.simd_ne(f32::NAN)), [-1; 16]);

    let snan = f32x16::from_slice(simd, &[f32::from_bits(0x7f80_0001); 16]);
    assert_eq!(<[i32; 16]>::from(snan.simd_ne(snan)), [-1; 16]);
    assert_eq!(<[i32; 16]>::from(snan.simd_ne(0.0)), [-1; 16]);
    assert_eq!(<[i32; 16]>::from(zeros.simd_ne(snan)), [-1; 16]);
}

#[simd_test]
fn simd_ne_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, 2.0, -3.0, 4.0, 1.0, 2.0, -3.0, 4.0]);
    let b = f64x8::from_slice(simd, &[1.0, -2.0, 3.0, 4.0, 1.0, -2.0, 3.0, 4.0]);
    assert_eq!(<[i64; 8]>::from(a.simd_ne(b)), [0, -1, -1, 0, 0, -1, -1, 0]);
    assert_eq!(
        <[i64; 8]>::from(simd.simd_ne_f64x8(a, b)),
        [0, -1, -1, 0, 0, -1, -1, 0]
    );

    let a = f64x8::from_slice(
        simd,
        &[
            0.0,
            f64::INFINITY,
            -0.0,
            f64::NEG_INFINITY,
            0.0,
            f64::INFINITY,
            -0.0,
            f64::NEG_INFINITY,
        ],
    );
    let b = f64x8::from_slice(
        simd,
        &[
            -0.0,
            f64::NEG_INFINITY,
            0.0,
            f64::INFINITY,
            -0.0,
            f64::NEG_INFINITY,
            0.0,
            f64::INFINITY,
        ],
    );
    assert_eq!(<[i64; 8]>::from(a.simd_ne(b)), [0, -1, 0, -1, 0, -1, 0, -1]);
    assert_eq!(<[i64; 8]>::from(a.simd_ne(a)), [0; 8]);
    let zeros = f64x8::from_slice(simd, &[0.0; 8]);
    assert_eq!(<[i64; 8]>::from(zeros.simd_ne(-0.0)), [0; 8]);

    let a = f64x8::from_slice(
        simd,
        &[f64::NAN, 1.0, f64::NAN, 1.0, f64::NAN, 1.0, f64::NAN, 1.0],
    );
    let b = f64x8::from_slice(
        simd,
        &[1.0, f64::NAN, f64::NAN, 1.0, 1.0, f64::NAN, f64::NAN, 1.0],
    );
    assert_eq!(
        <[i64; 8]>::from(a.simd_ne(b)),
        [-1, -1, -1, 0, -1, -1, -1, 0]
    );
    let nan = f64x8::from_slice(simd, &[f64::NAN; 8]);
    assert_eq!(<[i64; 8]>::from(nan.simd_ne(nan)), [-1; 8]);
    assert_eq!(<[i64; 8]>::from(nan.simd_ne(1.0)), [-1; 8]);
    assert_eq!(<[i64; 8]>::from(zeros.simd_ne(f64::NAN)), [-1; 8]);

    let snan = f64x8::from_slice(simd, &[f64::from_bits(0x7ff0_0000_0000_0001); 8]);
    assert_eq!(<[i64; 8]>::from(snan.simd_ne(snan)), [-1; 8]);
    assert_eq!(<[i64; 8]>::from(snan.simd_ne(0.0)), [-1; 8]);
    assert_eq!(<[i64; 8]>::from(zeros.simd_ne(snan)), [-1; 8]);
}
