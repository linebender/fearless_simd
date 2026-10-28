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
            0x0123456789ABCDEF,
            0x0123456789ABCDEF,
            0x0123456789ABCDEF,
            0x0123456789ABCDEF,
            0x0123456789ABCDEF,
            0x0123456789ABCDEF,
            0x0123456789ABCDEF,
        ],
    );
    assert_eq!(*a.reverse_bits(), a.to_array().map(u64::reverse_bits));
}
