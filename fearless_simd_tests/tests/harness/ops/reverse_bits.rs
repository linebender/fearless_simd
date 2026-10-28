// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn reverse_bits_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB,
            0xCD, 0xEF,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u8::reverse_bits));
}

#[simd_test]
fn reverse_bits_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB,
            0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67,
            0x89, 0xAB, 0xCD, 0xEF,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u8::reverse_bits));
}

#[simd_test]
fn reverse_bits_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB,
            0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67,
            0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23,
            0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u8::reverse_bits));
}

#[simd_test]
fn reverse_bits_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(
        simd,
        &[
            0x0123, 0x4567, 0x89AB, 0xCDEF, 0x0123, 0x4567, 0x89AB, 0xCDEF,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u16::reverse_bits));
}

#[simd_test]
fn reverse_bits_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[
            0x0123, 0x4567, 0x89AB, 0xCDEF, 0x0123, 0x4567, 0x89AB, 0xCDEF, 0x0123, 0x4567, 0x89AB,
            0xCDEF, 0x0123, 0x4567, 0x89AB, 0xCDEF,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u16::reverse_bits));
}

#[simd_test]
fn reverse_bits_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            0x0123, 0x4567, 0x89AB, 0xCDEF, 0x0123, 0x4567, 0x89AB, 0xCDEF, 0x0123, 0x4567, 0x89AB,
            0xCDEF, 0x0123, 0x4567, 0x89AB, 0xCDEF, 0x0123, 0x4567, 0x89AB, 0xCDEF, 0x0123, 0x4567,
            0x89AB, 0xCDEF, 0x0123, 0x4567, 0x89AB, 0xCDEF, 0x0123, 0x4567, 0x89AB, 0xCDEF,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u16::reverse_bits));
}

#[simd_test]
fn reverse_bits_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0x01234567, 0x89ABCDEF, 0x01234567, 0x89ABCDEF]);
    assert_eq!(*a.reverse_bits(), a.to_array().map(u32::reverse_bits));
}

#[simd_test]
fn reverse_bits_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(
        simd,
        &[
            0x01234567, 0x89ABCDEF, 0x01234567, 0x89ABCDEF, 0x01234567, 0x89ABCDEF, 0x01234567,
            0x89ABCDEF,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u32::reverse_bits));
}

#[simd_test]
fn reverse_bits_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            0x01234567, 0x89ABCDEF, 0x01234567, 0x89ABCDEF, 0x01234567, 0x89ABCDEF, 0x01234567,
            0x89ABCDEF, 0x01234567, 0x89ABCDEF, 0x01234567, 0x89ABCDEF, 0x01234567, 0x89ABCDEF,
            0x01234567, 0x89ABCDEF,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u32::reverse_bits));
}

#[simd_test]
fn reverse_bits_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[0x0123456789ABCDEF, 0x0123456789ABCDEF]);
    assert_eq!(*a.reverse_bits(), a.to_array().map(u64::reverse_bits));
}

#[simd_test]
fn reverse_bits_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(
        simd,
        &[
            0x0123456789ABCDEF,
            0x0123456789ABCDEF,
            0x0123456789ABCDEF,
            0x0123456789ABCDEF,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u64::reverse_bits));
}

#[simd_test]
fn reverse_bits_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[
            0x0123456789ABCDEF,
            0xFEDCBA9876543210,
            0x8000000000000001,
            0x0102040810204080,
            0,
            u64::MAX,
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u64::reverse_bits));
}

#[simd_test]
fn reverse_bits_u8x16_exhaustive<S: Simd>(simd: S) {
    for value in 0..=u8::MAX {
        let input: [u8; 16] =
            core::array::from_fn(|i| value.wrapping_add(u8::try_from(i).unwrap()));
        let a = u8x16::from_slice(simd, &input);
        assert_eq!(*a.reverse_bits(), input.map(u8::reverse_bits));
    }
}

#[simd_test]
fn reverse_bits_u8x32_exhaustive<S: Simd>(simd: S) {
    for value in 0..=u8::MAX {
        let input: [u8; 32] =
            core::array::from_fn(|i| value.wrapping_add(u8::try_from(i).unwrap()));
        let a = u8x32::from_slice(simd, &input);
        assert_eq!(*a.reverse_bits(), input.map(u8::reverse_bits));
    }
}

#[simd_test]
fn reverse_bits_i8x16_exhaustive<S: Simd>(simd: S) {
    for value in 0..=u8::MAX {
        let input: [i8; 16] =
            core::array::from_fn(|i| value.wrapping_add(u8::try_from(i).unwrap()).cast_signed());
        let a = i8x16::from_slice(simd, &input);
        assert_eq!(*a.reverse_bits(), input.map(i8::reverse_bits));
    }
}

#[simd_test]
fn reverse_bits_i8x32_exhaustive<S: Simd>(simd: S) {
    for value in 0..=u8::MAX {
        let input: [i8; 32] =
            core::array::from_fn(|i| value.wrapping_add(u8::try_from(i).unwrap()).cast_signed());
        let a = i8x32::from_slice(simd, &input);
        assert_eq!(*a.reverse_bits(), input.map(i8::reverse_bits));
    }
}

#[simd_test]
fn reverse_bits_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            0,
            -1,
            i16::MIN,
            i16::MAX,
            0x0123,
            -0x0123,
            0x4567,
            -0x4567,
            1,
            -2,
            0x0102,
            0x4080,
            0x5555,
            -0x5556,
            0x0008,
            0x0800,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(i16::reverse_bits));
}

#[simd_test]
fn reverse_bits_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(
        simd,
        &[
            0,
            -1,
            i32::MIN,
            i32::MAX,
            0x01234567,
            -0x01234567,
            1,
            0x01020408,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(i32::reverse_bits));
}

#[simd_test]
fn reverse_bits_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[
            0,
            -1,
            i64::MIN,
            i64::MAX,
            0x0123456789ABCDEF,
            -0x0123456789ABCDEF,
            1,
            0x0102040810204080,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(i64::reverse_bits));
}
