// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

const I8_MIN: i8 = i8::MIN;
const I8_MAX: i8 = i8::MAX;
const I16_MIN: i16 = i16::MIN;
const I16_MAX: i16 = i16::MAX;
const I32_MIN: i32 = i32::MIN;
const I32_MAX: i32 = i32::MAX;
const U8_MAX: u8 = u8::MAX;
const U8_MAX_M1: u8 = U8_MAX - 1;
const U16_MAX: u16 = u16::MAX;
const U16_MAX_M1: u16 = U16_MAX - 1;
const U32_MAX: u32 = u32::MAX;
const U32_MAX_M1: u32 = U32_MAX - 1;

#[simd_test]
fn widening_add_i8_exhaustive<S: Simd>(simd: S) {
    for left in i8::MIN..=i8::MAX {
        for base in (i16::from(i8::MIN)..=i16::from(i8::MAX)).step_by(64) {
            let a = i8x64::splat(simd, left);
            let values: [i8; 64] =
                core::array::from_fn(|i| i8::try_from(base + i16::try_from(i).unwrap()).unwrap());
            for chunk in values.as_chunks::<16>().0 {
                let a = i8x16::splat(simd, left);
                let b = i8x16::from_slice(simd, chunk);
                let (low, high) = a.widening_add(b);
                for i in 0..8 {
                    assert_eq!(low[i], i16::from(left) + i16::from(chunk[i]));
                    assert_eq!(high[i], i16::from(left) + i16::from(chunk[i + 8]));
                }
            }
            for chunk in values.as_chunks::<32>().0 {
                let a = i8x32::splat(simd, left);
                let b = i8x32::from_slice(simd, chunk);
                let (low, high) = a.widening_add(b);
                for i in 0..16 {
                    assert_eq!(low[i], i16::from(left) + i16::from(chunk[i]));
                    assert_eq!(high[i], i16::from(left) + i16::from(chunk[i + 16]));
                }
            }
            let b = i8x64::from_slice(simd, &values);
            let (low, high) = a.widening_add(b);
            for i in 0..32 {
                assert_eq!(low[i], i16::from(left) + i16::from(values[i]));
                assert_eq!(high[i], i16::from(left) + i16::from(values[i + 32]));
            }
        }
    }
}

#[simd_test]
fn widening_add_u8_exhaustive<S: Simd>(simd: S) {
    for left in u8::MIN..=u8::MAX {
        for base in (u16::from(u8::MIN)..=u16::from(u8::MAX)).step_by(64) {
            let a = u8x64::splat(simd, left);
            let values: [u8; 64] =
                core::array::from_fn(|i| u8::try_from(base + u16::try_from(i).unwrap()).unwrap());
            for chunk in values.as_chunks::<16>().0 {
                let a = u8x16::splat(simd, left);
                let b = u8x16::from_slice(simd, chunk);
                let (low, high) = a.widening_add(b);
                for i in 0..8 {
                    assert_eq!(low[i], u16::from(left) + u16::from(chunk[i]));
                    assert_eq!(high[i], u16::from(left) + u16::from(chunk[i + 8]));
                }
            }
            for chunk in values.as_chunks::<32>().0 {
                let a = u8x32::splat(simd, left);
                let b = u8x32::from_slice(simd, chunk);
                let (low, high) = a.widening_add(b);
                for i in 0..16 {
                    assert_eq!(low[i], u16::from(left) + u16::from(chunk[i]));
                    assert_eq!(high[i], u16::from(left) + u16::from(chunk[i + 16]));
                }
            }
            let b = u8x64::from_slice(simd, &values);
            let (low, high) = a.widening_add(b);
            for i in 0..32 {
                assert_eq!(low[i], u16::from(left) + u16::from(values[i]));
                assert_eq!(high[i], u16::from(left) + u16::from(values[i + 32]));
            }
        }
    }
}

#[simd_test]
fn widening_add_i8x16<S: Simd>(simd: S) {
    let a: [i8; 16] = [
        I8_MIN, I8_MAX, 0, 1, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15,
    ];
    let b: [i8; 16] = [
        I8_MIN, I8_MAX, -1, -1, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
    ];
    let lhs = i8x16::from_slice(simd, &a);
    let rhs = i8x16::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..8 {
        assert_eq!(low[i], (a[i] as i16) + (b[i] as i16));
        assert_eq!(high[i], (a[i + 8] as i16) + (b[i + 8] as i16));
    }
}

#[simd_test]
fn widening_add_i8x32<S: Simd>(simd: S) {
    let a: [i8; 32] = [
        I8_MIN, I8_MAX, 0, 1, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20,
        -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31,
    ];
    let b: [i8; 32] = [
        I8_MIN, I8_MAX, -1, -1, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
        43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
    ];
    let lhs = i8x32::from_slice(simd, &a);
    let rhs = i8x32::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..16 {
        assert_eq!(low[i], (a[i] as i16) + (b[i] as i16));
        assert_eq!(high[i], (a[i + 16] as i16) + (b[i + 16] as i16));
    }
}

#[simd_test]
fn widening_add_i8x64<S: Simd>(simd: S) {
    let a: [i8; 64] = [
        I8_MIN, I8_MAX, 0, 1, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20,
        -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31, 32, -33, 34, -35, 36, -37, 38, -39, 40,
        -41, 42, -43, 44, -45, 46, -47, 48, -49, 50, -51, 52, -53, 54, -55, 56, -57, 58, -59, 60,
        -61, 62, -63,
    ];
    let b: [i8; 64] = [
        I8_MIN, I8_MAX, -1, -1, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
        43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87,
        89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125,
        127,
    ];
    let lhs = i8x64::from_slice(simd, &a);
    let rhs = i8x64::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..32 {
        assert_eq!(low[i], (a[i] as i16) + (b[i] as i16));
        assert_eq!(high[i], (a[i + 32] as i16) + (b[i + 32] as i16));
    }
}

#[simd_test]
fn widening_add_i16x8<S: Simd>(simd: S) {
    let a: [i16; 8] = [I16_MIN, I16_MAX, 0, 1, 4, -5, 6, -7];
    let b: [i16; 8] = [I16_MIN, I16_MAX, -1, -1, 9, 11, 13, 15];
    let lhs = i16x8::from_slice(simd, &a);
    let rhs = i16x8::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..4 {
        assert_eq!(low[i], (a[i] as i32) + (b[i] as i32));
        assert_eq!(high[i], (a[i + 4] as i32) + (b[i + 4] as i32));
    }
}

#[simd_test]
fn widening_add_i16x16<S: Simd>(simd: S) {
    let a: [i16; 16] = [
        I16_MIN, I16_MAX, 0, 1, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15,
    ];
    let b: [i16; 16] = [
        I16_MIN, I16_MAX, -1, -1, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
    ];
    let lhs = i16x16::from_slice(simd, &a);
    let rhs = i16x16::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..8 {
        assert_eq!(low[i], (a[i] as i32) + (b[i] as i32));
        assert_eq!(high[i], (a[i + 8] as i32) + (b[i + 8] as i32));
    }
}

#[simd_test]
fn widening_add_i16x32<S: Simd>(simd: S) {
    let a: [i16; 32] = [
        I16_MIN, I16_MAX, 0, 1, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19,
        20, -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31,
    ];
    let b: [i16; 32] = [
        I16_MIN, I16_MAX, -1, -1, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
        41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
    ];
    let lhs = i16x32::from_slice(simd, &a);
    let rhs = i16x32::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..16 {
        assert_eq!(low[i], (a[i] as i32) + (b[i] as i32));
        assert_eq!(high[i], (a[i + 16] as i32) + (b[i + 16] as i32));
    }
}

#[simd_test]
fn widening_add_i32x4<S: Simd>(simd: S) {
    let a: [i32; 4] = [I32_MIN, I32_MAX, 0, 1];
    let b: [i32; 4] = [I32_MIN, I32_MAX, -1, -1];
    let lhs = i32x4::from_slice(simd, &a);
    let rhs = i32x4::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..2 {
        assert_eq!(low[i], (a[i] as i64) + (b[i] as i64));
        assert_eq!(high[i], (a[i + 2] as i64) + (b[i + 2] as i64));
    }
}

#[simd_test]
fn widening_add_i32x8<S: Simd>(simd: S) {
    let a: [i32; 8] = [I32_MIN, I32_MAX, 0, 1, 4, -5, 6, -7];
    let b: [i32; 8] = [I32_MIN, I32_MAX, -1, -1, 9, 11, 13, 15];
    let lhs = i32x8::from_slice(simd, &a);
    let rhs = i32x8::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..4 {
        assert_eq!(low[i], (a[i] as i64) + (b[i] as i64));
        assert_eq!(high[i], (a[i + 4] as i64) + (b[i + 4] as i64));
    }
}

#[simd_test]
fn widening_add_i32x16<S: Simd>(simd: S) {
    let a: [i32; 16] = [
        I32_MIN, I32_MAX, 0, 1, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15,
    ];
    let b: [i32; 16] = [
        I32_MIN, I32_MAX, -1, -1, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
    ];
    let lhs = i32x16::from_slice(simd, &a);
    let rhs = i32x16::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..8 {
        assert_eq!(low[i], (a[i] as i64) + (b[i] as i64));
        assert_eq!(high[i], (a[i + 8] as i64) + (b[i + 8] as i64));
    }
}

#[simd_test]
fn widening_add_u8x16<S: Simd>(simd: S) {
    let a: [u8; 16] = [
        U8_MAX, U8_MAX_M1, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ];
    let b: [u8; 16] = [
        U8_MAX, U8_MAX, 1, 0, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
    ];
    let lhs = u8x16::from_slice(simd, &a);
    let rhs = u8x16::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..8 {
        assert_eq!(low[i], (a[i] as u16) + (b[i] as u16));
        assert_eq!(high[i], (a[i + 8] as u16) + (b[i + 8] as u16));
    }
}

#[simd_test]
fn widening_add_u8x32<S: Simd>(simd: S) {
    let a: [u8; 32] = [
        U8_MAX, U8_MAX_M1, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];
    let b: [u8; 32] = [
        U8_MAX, U8_MAX, 1, 0, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
        43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
    ];
    let lhs = u8x32::from_slice(simd, &a);
    let rhs = u8x32::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..16 {
        assert_eq!(low[i], (a[i] as u16) + (b[i] as u16));
        assert_eq!(high[i], (a[i + 16] as u16) + (b[i + 16] as u16));
    }
}

#[simd_test]
fn widening_add_u8x64<S: Simd>(simd: S) {
    let a: [u8; 64] = [
        U8_MAX, U8_MAX_M1, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ];
    let b: [u8; 64] = [
        U8_MAX, U8_MAX, 1, 0, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
        43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87,
        89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125,
        127,
    ];
    let lhs = u8x64::from_slice(simd, &a);
    let rhs = u8x64::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..32 {
        assert_eq!(low[i], (a[i] as u16) + (b[i] as u16));
        assert_eq!(high[i], (a[i + 32] as u16) + (b[i + 32] as u16));
    }
}

#[simd_test]
fn widening_add_u16x8<S: Simd>(simd: S) {
    let a: [u16; 8] = [U16_MAX, U16_MAX_M1, 0, 1, 4, 5, 6, 7];
    let b: [u16; 8] = [U16_MAX, U16_MAX, 1, 0, 9, 11, 13, 15];
    let lhs = u16x8::from_slice(simd, &a);
    let rhs = u16x8::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..4 {
        assert_eq!(low[i], (a[i] as u32) + (b[i] as u32));
        assert_eq!(high[i], (a[i + 4] as u32) + (b[i + 4] as u32));
    }
}

#[simd_test]
fn widening_add_u16x16<S: Simd>(simd: S) {
    let a: [u16; 16] = [
        U16_MAX, U16_MAX_M1, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ];
    let b: [u16; 16] = [
        U16_MAX, U16_MAX, 1, 0, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
    ];
    let lhs = u16x16::from_slice(simd, &a);
    let rhs = u16x16::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..8 {
        assert_eq!(low[i], (a[i] as u32) + (b[i] as u32));
        assert_eq!(high[i], (a[i + 8] as u32) + (b[i + 8] as u32));
    }
}

#[simd_test]
fn widening_add_u16x32<S: Simd>(simd: S) {
    let a: [u16; 32] = [
        U16_MAX, U16_MAX_M1, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];
    let b: [u16; 32] = [
        U16_MAX, U16_MAX, 1, 0, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
        43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
    ];
    let lhs = u16x32::from_slice(simd, &a);
    let rhs = u16x32::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..16 {
        assert_eq!(low[i], (a[i] as u32) + (b[i] as u32));
        assert_eq!(high[i], (a[i + 16] as u32) + (b[i + 16] as u32));
    }
}

#[simd_test]
fn widening_add_u32x4<S: Simd>(simd: S) {
    let a: [u32; 4] = [U32_MAX, U32_MAX_M1, 0, 1];
    let b: [u32; 4] = [U32_MAX, U32_MAX, 1, 0];
    let lhs = u32x4::from_slice(simd, &a);
    let rhs = u32x4::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..2 {
        assert_eq!(low[i], (a[i] as u64) + (b[i] as u64));
        assert_eq!(high[i], (a[i + 2] as u64) + (b[i + 2] as u64));
    }
}

#[simd_test]
fn widening_add_u32x8<S: Simd>(simd: S) {
    let a: [u32; 8] = [U32_MAX, U32_MAX_M1, 0, 1, 4, 5, 6, 7];
    let b: [u32; 8] = [U32_MAX, U32_MAX, 1, 0, 9, 11, 13, 15];
    let lhs = u32x8::from_slice(simd, &a);
    let rhs = u32x8::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..4 {
        assert_eq!(low[i], (a[i] as u64) + (b[i] as u64));
        assert_eq!(high[i], (a[i + 4] as u64) + (b[i + 4] as u64));
    }
}

#[simd_test]
fn widening_add_u32x16<S: Simd>(simd: S) {
    let a: [u32; 16] = [
        U32_MAX, U32_MAX_M1, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ];
    let b: [u32; 16] = [
        U32_MAX, U32_MAX, 1, 0, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
    ];
    let lhs = u32x16::from_slice(simd, &a);
    let rhs = u32x16::from_slice(simd, &b);
    let (low, high) = lhs.widening_add(rhs);

    for i in 0..8 {
        assert_eq!(low[i], (a[i] as u64) + (b[i] as u64));
        assert_eq!(high[i], (a[i + 8] as u64) + (b[i + 8] as u64));
    }
}

#[simd_test]
fn widening_add_generic<S: Simd>(simd: S) {
    fn add<S: Simd, V: SimdIntWiden<S>>(a: V, b: V) -> (V::Widened, V::Widened) {
        a.widening_add(b)
    }

    let a = i32x4::from_slice(simd, &[I32_MIN, I32_MAX, -1, 0]);
    let b = i32x4::from_slice(simd, &[I32_MIN, I32_MAX, 1, -1]);
    let (low, high) = add(a, b);
    assert_eq!(*low, [-4_294_967_296, 4_294_967_294]);
    assert_eq!(*high, [0, -1]);
}

#[simd_test]
fn widening_add_native_i8<S: Simd>(simd: S) {
    let a = S::i8s::splat(simd, I8_MAX);
    let b = S::i8s::splat(simd, I8_MAX);
    let (low, high): (S::i16s, S::i16s) = a.widening_add(b);
    let expected = (I8_MAX as i16) + (I8_MAX as i16);
    assert!(low.as_slice().iter().all(|&lane| lane == expected));
    assert!(high.as_slice().iter().all(|&lane| lane == expected));
}

#[simd_test]
fn widening_add_native_i16<S: Simd>(simd: S) {
    let a = S::i16s::splat(simd, I16_MAX);
    let b = S::i16s::splat(simd, I16_MAX);
    let (low, high): (S::i32s, S::i32s) = a.widening_add(b);
    let expected = (I16_MAX as i32) + (I16_MAX as i32);
    assert!(low.as_slice().iter().all(|&lane| lane == expected));
    assert!(high.as_slice().iter().all(|&lane| lane == expected));
}

#[simd_test]
fn widening_add_native_i32<S: Simd>(simd: S) {
    let a = S::i32s::splat(simd, I32_MAX);
    let b = S::i32s::splat(simd, I32_MAX);
    let (low, high): (S::i64s, S::i64s) = a.widening_add(b);
    let expected = (I32_MAX as i64) + (I32_MAX as i64);
    assert!(low.as_slice().iter().all(|&lane| lane == expected));
    assert!(high.as_slice().iter().all(|&lane| lane == expected));
}

#[simd_test]
fn widening_add_native_u8<S: Simd>(simd: S) {
    let a = S::u8s::splat(simd, U8_MAX);
    let b = S::u8s::splat(simd, U8_MAX);
    let (low, high): (S::u16s, S::u16s) = a.widening_add(b);
    let expected = (U8_MAX as u16) + (U8_MAX as u16);
    assert!(low.as_slice().iter().all(|&lane| lane == expected));
    assert!(high.as_slice().iter().all(|&lane| lane == expected));
}

#[simd_test]
fn widening_add_native_u16<S: Simd>(simd: S) {
    let a = S::u16s::splat(simd, U16_MAX);
    let b = S::u16s::splat(simd, U16_MAX);
    let (low, high): (S::u32s, S::u32s) = a.widening_add(b);
    let expected = (U16_MAX as u32) + (U16_MAX as u32);
    assert!(low.as_slice().iter().all(|&lane| lane == expected));
    assert!(high.as_slice().iter().all(|&lane| lane == expected));
}

#[simd_test]
fn widening_add_native_u32<S: Simd>(simd: S) {
    let a = S::u32s::splat(simd, U32_MAX);
    let b = S::u32s::splat(simd, U32_MAX);
    let (low, high): (S::u64s, S::u64s) = a.widening_add(b);
    let expected = (U32_MAX as u64) + (U32_MAX as u64);
    assert!(low.as_slice().iter().all(|&lane| lane == expected));
    assert!(high.as_slice().iter().all(|&lane| lane == expected));
}
