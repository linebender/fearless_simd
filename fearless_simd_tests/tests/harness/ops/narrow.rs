// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type and narrowing mode.

#[simd_test]
fn narrow_u16x8<S: Simd>(simd: S) {
    let low = u16x8::from_slice(simd, &[0, 255, 256, 65_535, 1, 127, 128, 300]);
    let high = u16x8::from_slice(simd, &[0, 255, 256, 65_535, 1, 127, 128, 300]);

    assert_eq!(
        *low.narrow(high),
        [
            0, 255, 0, 255, 1, 127, 128, 44, 0, 255, 0, 255, 1, 127, 128, 44
        ]
    );
}

#[simd_test]
fn saturating_narrow_u16x8<S: Simd>(simd: S) {
    let low = u16x8::from_slice(simd, &[0, 255, 256, 65_535, 1, 127, 128, 300]);
    let high = u16x8::from_slice(simd, &[0, 255, 256, 65_535, 1, 127, 128, 300]);

    assert_eq!(
        *low.saturating_narrow(high),
        [
            0, 255, 255, 255, 1, 127, 128, 255, 0, 255, 255, 255, 1, 127, 128, 255
        ]
    );
}

#[simd_test]
fn narrow_u16x16<S: Simd>(simd: S) {
    let low = u16x16::from_slice(
        simd,
        &[
            0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300,
        ],
    );
    let high = u16x16::from_slice(
        simd,
        &[
            0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300,
        ],
    );

    assert_eq!(
        *low.narrow(high),
        [
            0, 255, 0, 255, 1, 127, 128, 44, 0, 255, 0, 255, 1, 127, 128, 44, 0, 255, 0, 255, 1,
            127, 128, 44, 0, 255, 0, 255, 1, 127, 128, 44
        ]
    );
}

#[simd_test]
fn saturating_narrow_u16x16<S: Simd>(simd: S) {
    let low = u16x16::from_slice(
        simd,
        &[
            0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300,
        ],
    );
    let high = u16x16::from_slice(
        simd,
        &[
            0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300,
        ],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            0, 255, 255, 255, 1, 127, 128, 255, 0, 255, 255, 255, 1, 127, 128, 255, 0, 255, 255,
            255, 1, 127, 128, 255, 0, 255, 255, 255, 1, 127, 128, 255
        ]
    );
}

#[simd_test]
fn narrow_u16x32<S: Simd>(simd: S) {
    let low = u16x32::from_slice(
        simd,
        &[
            0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255,
            256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300,
        ],
    );
    let high = u16x32::from_slice(
        simd,
        &[
            0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255,
            256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300,
        ],
    );

    assert_eq!(
        *low.narrow(high),
        [
            0, 255, 0, 255, 1, 127, 128, 44, 0, 255, 0, 255, 1, 127, 128, 44, 0, 255, 0, 255, 1,
            127, 128, 44, 0, 255, 0, 255, 1, 127, 128, 44, 0, 255, 0, 255, 1, 127, 128, 44, 0, 255,
            0, 255, 1, 127, 128, 44, 0, 255, 0, 255, 1, 127, 128, 44, 0, 255, 0, 255, 1, 127, 128,
            44
        ]
    );
}

#[simd_test]
fn saturating_narrow_u16x32<S: Simd>(simd: S) {
    let low = u16x32::from_slice(
        simd,
        &[
            0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255,
            256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300,
        ],
    );
    let high = u16x32::from_slice(
        simd,
        &[
            0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300, 0, 255,
            256, 65_535, 1, 127, 128, 300, 0, 255, 256, 65_535, 1, 127, 128, 300,
        ],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            0, 255, 255, 255, 1, 127, 128, 255, 0, 255, 255, 255, 1, 127, 128, 255, 0, 255, 255,
            255, 1, 127, 128, 255, 0, 255, 255, 255, 1, 127, 128, 255, 0, 255, 255, 255, 1, 127,
            128, 255, 0, 255, 255, 255, 1, 127, 128, 255, 0, 255, 255, 255, 1, 127, 128, 255, 0,
            255, 255, 255, 1, 127, 128, 255
        ]
    );
}

#[simd_test]
fn narrow_i16x8<S: Simd>(simd: S) {
    let low = i16x8::from_slice(simd, &[-129, -128, 127, 128, -32_768, -1, 0, 32_767]);
    let high = i16x8::from_slice(simd, &[-129, -128, 127, 128, -32_768, -1, 0, 32_767]);

    assert_eq!(
        *low.narrow(high),
        [
            127, -128, 127, -128, 0, -1, 0, -1, 127, -128, 127, -128, 0, -1, 0, -1
        ]
    );
}

#[simd_test]
fn saturating_narrow_i16x8<S: Simd>(simd: S) {
    let low = i16x8::from_slice(simd, &[-129, -128, 127, 128, -32_768, -1, 0, 32_767]);
    let high = i16x8::from_slice(simd, &[-129, -128, 127, 128, -32_768, -1, 0, 32_767]);

    assert_eq!(
        *low.saturating_narrow(high),
        [
            -128, -128, 127, 127, -128, -1, 0, 127, -128, -128, 127, 127, -128, -1, 0, 127
        ]
    );
}

#[simd_test]
fn narrow_i16x16<S: Simd>(simd: S) {
    let low = i16x16::from_slice(
        simd,
        &[
            -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768, -1, 0,
            32_767,
        ],
    );
    let high = i16x16::from_slice(
        simd,
        &[
            -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768, -1, 0,
            32_767,
        ],
    );

    assert_eq!(
        *low.narrow(high),
        [
            127, -128, 127, -128, 0, -1, 0, -1, 127, -128, 127, -128, 0, -1, 0, -1, 127, -128, 127,
            -128, 0, -1, 0, -1, 127, -128, 127, -128, 0, -1, 0, -1
        ]
    );
}

#[simd_test]
fn saturating_narrow_i16x16<S: Simd>(simd: S) {
    let low = i16x16::from_slice(
        simd,
        &[
            -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768, -1, 0,
            32_767,
        ],
    );
    let high = i16x16::from_slice(
        simd,
        &[
            -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768, -1, 0,
            32_767,
        ],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            -128, -128, 127, 127, -128, -1, 0, 127, -128, -128, 127, 127, -128, -1, 0, 127, -128,
            -128, 127, 127, -128, -1, 0, 127, -128, -128, 127, 127, -128, -1, 0, 127
        ]
    );
}

#[simd_test]
fn narrow_i16x32<S: Simd>(simd: S) {
    let low = i16x32::from_slice(
        simd,
        &[
            -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768, -1, 0,
            32_767, -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768,
            -1, 0, 32_767,
        ],
    );
    let high = i16x32::from_slice(
        simd,
        &[
            -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768, -1, 0,
            32_767, -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768,
            -1, 0, 32_767,
        ],
    );

    assert_eq!(
        *low.narrow(high),
        [
            127, -128, 127, -128, 0, -1, 0, -1, 127, -128, 127, -128, 0, -1, 0, -1, 127, -128, 127,
            -128, 0, -1, 0, -1, 127, -128, 127, -128, 0, -1, 0, -1, 127, -128, 127, -128, 0, -1, 0,
            -1, 127, -128, 127, -128, 0, -1, 0, -1, 127, -128, 127, -128, 0, -1, 0, -1, 127, -128,
            127, -128, 0, -1, 0, -1
        ]
    );
}

#[simd_test]
fn saturating_narrow_i16x32<S: Simd>(simd: S) {
    let low = i16x32::from_slice(
        simd,
        &[
            -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768, -1, 0,
            32_767, -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768,
            -1, 0, 32_767,
        ],
    );
    let high = i16x32::from_slice(
        simd,
        &[
            -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768, -1, 0,
            32_767, -129, -128, 127, 128, -32_768, -1, 0, 32_767, -129, -128, 127, 128, -32_768,
            -1, 0, 32_767,
        ],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            -128, -128, 127, 127, -128, -1, 0, 127, -128, -128, 127, 127, -128, -1, 0, 127, -128,
            -128, 127, 127, -128, -1, 0, 127, -128, -128, 127, 127, -128, -1, 0, 127, -128, -128,
            127, 127, -128, -1, 0, 127, -128, -128, 127, 127, -128, -1, 0, 127, -128, -128, 127,
            127, -128, -1, 0, 127, -128, -128, 127, 127, -128, -1, 0, 127
        ]
    );
}

#[simd_test]
fn narrow_u32x4<S: Simd>(simd: S) {
    let low = u32x4::from_slice(simd, &[0, 65_535, 65_536, 4_294_967_295]);
    let high = u32x4::from_slice(simd, &[1, 32_767, 32_768, 70_000]);

    assert_eq!(
        *low.narrow(high),
        [0, 65_535, 0, 65_535, 1, 32_767, 32_768, 4_464]
    );
}

#[simd_test]
fn saturating_narrow_u32x4<S: Simd>(simd: S) {
    let low = u32x4::from_slice(simd, &[0, 65_535, 65_536, 4_294_967_295]);
    let high = u32x4::from_slice(simd, &[1, 32_767, 32_768, 70_000]);

    assert_eq!(
        *low.saturating_narrow(high),
        [0, 65_535, 65_535, 65_535, 1, 32_767, 32_768, 65_535]
    );
}

#[simd_test]
fn narrow_u32x8<S: Simd>(simd: S) {
    let low = u32x8::from_slice(
        simd,
        &[0, 65_535, 65_536, 4_294_967_295, 1, 32_767, 32_768, 70_000],
    );
    let high = u32x8::from_slice(
        simd,
        &[0, 65_535, 65_536, 4_294_967_295, 1, 32_767, 32_768, 70_000],
    );

    assert_eq!(
        *low.narrow(high),
        [
            0, 65_535, 0, 65_535, 1, 32_767, 32_768, 4_464, 0, 65_535, 0, 65_535, 1, 32_767,
            32_768, 4_464
        ]
    );
}

#[simd_test]
fn saturating_narrow_u32x8<S: Simd>(simd: S) {
    let low = u32x8::from_slice(
        simd,
        &[0, 65_535, 65_536, 4_294_967_295, 1, 32_767, 32_768, 70_000],
    );
    let high = u32x8::from_slice(
        simd,
        &[0, 65_535, 65_536, 4_294_967_295, 1, 32_767, 32_768, 70_000],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            0, 65_535, 65_535, 65_535, 1, 32_767, 32_768, 65_535, 0, 65_535, 65_535, 65_535, 1,
            32_767, 32_768, 65_535
        ]
    );
}

#[simd_test]
fn narrow_u32x16<S: Simd>(simd: S) {
    let low = u32x16::from_slice(
        simd,
        &[
            0,
            65_535,
            65_536,
            4_294_967_295,
            1,
            32_767,
            32_768,
            70_000,
            0,
            65_535,
            65_536,
            4_294_967_295,
            1,
            32_767,
            32_768,
            70_000,
        ],
    );
    let high = u32x16::from_slice(
        simd,
        &[
            0,
            65_535,
            65_536,
            4_294_967_295,
            1,
            32_767,
            32_768,
            70_000,
            0,
            65_535,
            65_536,
            4_294_967_295,
            1,
            32_767,
            32_768,
            70_000,
        ],
    );

    assert_eq!(
        *low.narrow(high),
        [
            0, 65_535, 0, 65_535, 1, 32_767, 32_768, 4_464, 0, 65_535, 0, 65_535, 1, 32_767,
            32_768, 4_464, 0, 65_535, 0, 65_535, 1, 32_767, 32_768, 4_464, 0, 65_535, 0, 65_535, 1,
            32_767, 32_768, 4_464
        ]
    );
}

#[simd_test]
fn saturating_narrow_u32x16<S: Simd>(simd: S) {
    let low = u32x16::from_slice(
        simd,
        &[
            0,
            65_535,
            65_536,
            4_294_967_295,
            1,
            32_767,
            32_768,
            70_000,
            0,
            65_535,
            65_536,
            4_294_967_295,
            1,
            32_767,
            32_768,
            70_000,
        ],
    );
    let high = u32x16::from_slice(
        simd,
        &[
            0,
            65_535,
            65_536,
            4_294_967_295,
            1,
            32_767,
            32_768,
            70_000,
            0,
            65_535,
            65_536,
            4_294_967_295,
            1,
            32_767,
            32_768,
            70_000,
        ],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            0, 65_535, 65_535, 65_535, 1, 32_767, 32_768, 65_535, 0, 65_535, 65_535, 65_535, 1,
            32_767, 32_768, 65_535, 0, 65_535, 65_535, 65_535, 1, 32_767, 32_768, 65_535, 0,
            65_535, 65_535, 65_535, 1, 32_767, 32_768, 65_535
        ]
    );
}

#[simd_test]
fn narrow_i32x4<S: Simd>(simd: S) {
    let low = i32x4::from_slice(simd, &[-32_769, -32_768, 32_767, 32_768]);
    let high = i32x4::from_slice(simd, &[-2_147_483_648, -1, 0, 2_147_483_647]);

    assert_eq!(
        *low.narrow(high),
        [32_767, -32_768, 32_767, -32_768, 0, -1, 0, -1]
    );
}

#[simd_test]
fn saturating_narrow_i32x4<S: Simd>(simd: S) {
    let low = i32x4::from_slice(simd, &[-32_769, -32_768, 32_767, 32_768]);
    let high = i32x4::from_slice(simd, &[-2_147_483_648, -1, 0, 2_147_483_647]);

    assert_eq!(
        *low.saturating_narrow(high),
        [-32_768, -32_768, 32_767, 32_767, -32_768, -1, 0, 32_767]
    );
}

#[simd_test]
fn narrow_i32x8<S: Simd>(simd: S) {
    let low = i32x8::from_slice(
        simd,
        &[
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
        ],
    );
    let high = i32x8::from_slice(
        simd,
        &[
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
        ],
    );

    assert_eq!(
        *low.narrow(high),
        [
            32_767, -32_768, 32_767, -32_768, 0, -1, 0, -1, 32_767, -32_768, 32_767, -32_768, 0,
            -1, 0, -1
        ]
    );
}

#[simd_test]
fn saturating_narrow_i32x8<S: Simd>(simd: S) {
    let low = i32x8::from_slice(
        simd,
        &[
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
        ],
    );
    let high = i32x8::from_slice(
        simd,
        &[
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
        ],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            -32_768, -32_768, 32_767, 32_767, -32_768, -1, 0, 32_767, -32_768, -32_768, 32_767,
            32_767, -32_768, -1, 0, 32_767
        ]
    );
}

#[simd_test]
fn narrow_i32x16<S: Simd>(simd: S) {
    let low = i32x16::from_slice(
        simd,
        &[
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
        ],
    );
    let high = i32x16::from_slice(
        simd,
        &[
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
        ],
    );

    assert_eq!(
        *low.narrow(high),
        [
            32_767, -32_768, 32_767, -32_768, 0, -1, 0, -1, 32_767, -32_768, 32_767, -32_768, 0,
            -1, 0, -1, 32_767, -32_768, 32_767, -32_768, 0, -1, 0, -1, 32_767, -32_768, 32_767,
            -32_768, 0, -1, 0, -1
        ]
    );
}

#[simd_test]
fn saturating_narrow_i32x16<S: Simd>(simd: S) {
    let low = i32x16::from_slice(
        simd,
        &[
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
        ],
    );
    let high = i32x16::from_slice(
        simd,
        &[
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
            -32_769,
            -32_768,
            32_767,
            32_768,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
        ],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            -32_768, -32_768, 32_767, 32_767, -32_768, -1, 0, 32_767, -32_768, -32_768, 32_767,
            32_767, -32_768, -1, 0, 32_767, -32_768, -32_768, 32_767, 32_767, -32_768, -1, 0,
            32_767, -32_768, -32_768, 32_767, 32_767, -32_768, -1, 0, 32_767
        ]
    );
}

#[simd_test]
fn narrow_u64x2<S: Simd>(simd: S) {
    let low = u64x2::from_slice(simd, &[0, 4_294_967_295]);
    let high = u64x2::from_slice(simd, &[4_294_967_296, 18_446_744_073_709_551_615]);

    assert_eq!(*low.narrow(high), [0, 4_294_967_295, 0, 4_294_967_295]);
}

#[simd_test]
fn saturating_narrow_u64x2<S: Simd>(simd: S) {
    let low = u64x2::from_slice(simd, &[0, 4_294_967_295]);
    let high = u64x2::from_slice(simd, &[4_294_967_296, 18_446_744_073_709_551_615]);

    assert_eq!(
        *low.saturating_narrow(high),
        [0, 4_294_967_295, 4_294_967_295, 4_294_967_295]
    );
}

#[simd_test]
fn narrow_u64x4<S: Simd>(simd: S) {
    let low = u64x4::from_slice(
        simd,
        &[0, 4_294_967_295, 4_294_967_296, 18_446_744_073_709_551_615],
    );
    let high = u64x4::from_slice(simd, &[1, 2_147_483_647, 2_147_483_648, 5_000_000_000]);

    assert_eq!(
        *low.narrow(high),
        [
            0,
            4_294_967_295,
            0,
            4_294_967_295,
            1,
            2_147_483_647,
            2_147_483_648,
            705_032_704
        ]
    );
}

#[simd_test]
fn saturating_narrow_u64x4<S: Simd>(simd: S) {
    let low = u64x4::from_slice(
        simd,
        &[0, 4_294_967_295, 4_294_967_296, 18_446_744_073_709_551_615],
    );
    let high = u64x4::from_slice(simd, &[1, 2_147_483_647, 2_147_483_648, 5_000_000_000]);

    assert_eq!(
        *low.saturating_narrow(high),
        [
            0,
            4_294_967_295,
            4_294_967_295,
            4_294_967_295,
            1,
            2_147_483_647,
            2_147_483_648,
            4_294_967_295
        ]
    );
}

#[simd_test]
fn narrow_u64x8<S: Simd>(simd: S) {
    let low = u64x8::from_slice(
        simd,
        &[
            0,
            4_294_967_295,
            4_294_967_296,
            18_446_744_073_709_551_615,
            1,
            2_147_483_647,
            2_147_483_648,
            5_000_000_000,
        ],
    );
    let high = u64x8::from_slice(
        simd,
        &[
            0,
            4_294_967_295,
            4_294_967_296,
            18_446_744_073_709_551_615,
            1,
            2_147_483_647,
            2_147_483_648,
            5_000_000_000,
        ],
    );

    assert_eq!(
        *low.narrow(high),
        [
            0,
            4_294_967_295,
            0,
            4_294_967_295,
            1,
            2_147_483_647,
            2_147_483_648,
            705_032_704,
            0,
            4_294_967_295,
            0,
            4_294_967_295,
            1,
            2_147_483_647,
            2_147_483_648,
            705_032_704
        ]
    );
}

#[simd_test]
fn saturating_narrow_u64x8<S: Simd>(simd: S) {
    let low = u64x8::from_slice(
        simd,
        &[
            0,
            4_294_967_295,
            4_294_967_296,
            18_446_744_073_709_551_615,
            1,
            2_147_483_647,
            2_147_483_648,
            5_000_000_000,
        ],
    );
    let high = u64x8::from_slice(
        simd,
        &[
            0,
            4_294_967_295,
            4_294_967_296,
            18_446_744_073_709_551_615,
            1,
            2_147_483_647,
            2_147_483_648,
            5_000_000_000,
        ],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            0,
            4_294_967_295,
            4_294_967_295,
            4_294_967_295,
            1,
            2_147_483_647,
            2_147_483_648,
            4_294_967_295,
            0,
            4_294_967_295,
            4_294_967_295,
            4_294_967_295,
            1,
            2_147_483_647,
            2_147_483_648,
            4_294_967_295
        ]
    );
}

#[simd_test]
fn narrow_i64x2<S: Simd>(simd: S) {
    let low = i64x2::from_slice(simd, &[-2_147_483_649, -2_147_483_648]);
    let high = i64x2::from_slice(simd, &[2_147_483_647, 2_147_483_648]);

    assert_eq!(
        *low.narrow(high),
        [2_147_483_647, -2_147_483_648, 2_147_483_647, -2_147_483_648]
    );
}

#[simd_test]
fn saturating_narrow_i64x2<S: Simd>(simd: S) {
    let low = i64x2::from_slice(simd, &[-2_147_483_649, -2_147_483_648]);
    let high = i64x2::from_slice(simd, &[2_147_483_647, 2_147_483_648]);

    assert_eq!(
        *low.saturating_narrow(high),
        [-2_147_483_648, -2_147_483_648, 2_147_483_647, 2_147_483_647]
    );
}

#[simd_test]
fn narrow_i64x4<S: Simd>(simd: S) {
    let low = i64x4::from_slice(
        simd,
        &[-2_147_483_649, -2_147_483_648, 2_147_483_647, 2_147_483_648],
    );
    let high = i64x4::from_slice(
        simd,
        &[-9_223_372_036_854_775_808, -1, 0, 9_223_372_036_854_775_807],
    );

    assert_eq!(
        *low.narrow(high),
        [
            2_147_483_647,
            -2_147_483_648,
            2_147_483_647,
            -2_147_483_648,
            0,
            -1,
            0,
            -1
        ]
    );
}

#[simd_test]
fn saturating_narrow_i64x4<S: Simd>(simd: S) {
    let low = i64x4::from_slice(
        simd,
        &[-2_147_483_649, -2_147_483_648, 2_147_483_647, 2_147_483_648],
    );
    let high = i64x4::from_slice(
        simd,
        &[-9_223_372_036_854_775_808, -1, 0, 9_223_372_036_854_775_807],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            -2_147_483_648,
            -2_147_483_648,
            2_147_483_647,
            2_147_483_647,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647
        ]
    );
}

#[simd_test]
fn narrow_i64x8<S: Simd>(simd: S) {
    let low = i64x8::from_slice(
        simd,
        &[
            -2_147_483_649,
            -2_147_483_648,
            2_147_483_647,
            2_147_483_648,
            -9_223_372_036_854_775_808,
            -1,
            0,
            9_223_372_036_854_775_807,
        ],
    );
    let high = i64x8::from_slice(
        simd,
        &[
            -2_147_483_649,
            -2_147_483_648,
            2_147_483_647,
            2_147_483_648,
            -9_223_372_036_854_775_808,
            -1,
            0,
            9_223_372_036_854_775_807,
        ],
    );

    assert_eq!(
        *low.narrow(high),
        [
            2_147_483_647,
            -2_147_483_648,
            2_147_483_647,
            -2_147_483_648,
            0,
            -1,
            0,
            -1,
            2_147_483_647,
            -2_147_483_648,
            2_147_483_647,
            -2_147_483_648,
            0,
            -1,
            0,
            -1
        ]
    );
}

#[simd_test]
fn saturating_narrow_i64x8<S: Simd>(simd: S) {
    let low = i64x8::from_slice(
        simd,
        &[
            -2_147_483_649,
            -2_147_483_648,
            2_147_483_647,
            2_147_483_648,
            -9_223_372_036_854_775_808,
            -1,
            0,
            9_223_372_036_854_775_807,
        ],
    );
    let high = i64x8::from_slice(
        simd,
        &[
            -2_147_483_649,
            -2_147_483_648,
            2_147_483_647,
            2_147_483_648,
            -9_223_372_036_854_775_808,
            -1,
            0,
            9_223_372_036_854_775_807,
        ],
    );

    assert_eq!(
        *low.saturating_narrow(high),
        [
            -2_147_483_648,
            -2_147_483_648,
            2_147_483_647,
            2_147_483_647,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647,
            -2_147_483_648,
            -2_147_483_648,
            2_147_483_647,
            2_147_483_647,
            -2_147_483_648,
            -1,
            0,
            2_147_483_647
        ]
    );
}

#[simd_test]
fn narrow_f64x2<S: Simd>(simd: S) {
    let low = f64x2::from_slice(simd, &[-2.25, -0.5]);
    let high = f64x2::from_slice(simd, &[0.5, 1.5]);

    assert_eq!(*low.narrow(high), [-2.25, -0.5, 0.5, 1.5]);
}

#[simd_test]
fn saturating_narrow_f64x2<S: Simd>(simd: S) {
    let low = f64x2::from_slice(simd, &[-2.25, -0.5]);
    let high = f64x2::from_slice(simd, &[0.5, 1.5]);

    assert_eq!(*low.saturating_narrow(high), [-2.25, -0.5, 0.5, 1.5]);
}

#[simd_test]
fn narrow_f64x4<S: Simd>(simd: S) {
    let low = f64x4::from_slice(simd, &[-1_000.0, -42.5, -2.25, -0.5]);
    let high = f64x4::from_slice(simd, &[0.5, 1.5, 42.5, 1_000.0]);

    assert_eq!(
        *low.narrow(high),
        [-1_000.0, -42.5, -2.25, -0.5, 0.5, 1.5, 42.5, 1_000.0]
    );
}

#[simd_test]
fn saturating_narrow_f64x4<S: Simd>(simd: S) {
    let low = f64x4::from_slice(simd, &[-1_000.0, -42.5, -2.25, -0.5]);
    let high = f64x4::from_slice(simd, &[0.5, 1.5, 42.5, 1_000.0]);

    assert_eq!(
        *low.saturating_narrow(high),
        [-1_000.0, -42.5, -2.25, -0.5, 0.5, 1.5, 42.5, 1_000.0]
    );
}

#[simd_test]
fn narrow_f64x8<S: Simd>(simd: S) {
    let low = f64x8::from_slice(
        simd,
        &[-65_536.0, -1_000.0, -42.5, -2.25, -1.5, -1.0, -0.5, -0.25],
    );
    let high = f64x8::from_slice(simd, &[0.25, 0.5, 1.0, 1.5, 2.25, 42.5, 1_000.0, 65_536.0]);

    assert_eq!(
        *low.narrow(high),
        [
            -65_536.0, -1_000.0, -42.5, -2.25, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.25,
            42.5, 1_000.0, 65_536.0
        ]
    );
}

#[simd_test]
fn saturating_narrow_f64x8<S: Simd>(simd: S) {
    let low = f64x8::from_slice(
        simd,
        &[-65_536.0, -1_000.0, -42.5, -2.25, -1.5, -1.0, -0.5, -0.25],
    );
    let high = f64x8::from_slice(simd, &[0.25, 0.5, 1.0, 1.5, 2.25, 42.5, 1_000.0, 65_536.0]);

    assert_eq!(
        *low.saturating_narrow(high),
        [
            -65_536.0, -1_000.0, -42.5, -2.25, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.25,
            42.5, 1_000.0, 65_536.0
        ]
    );
}

#[simd_test]
#[ignore] // slow: run with `cargo test --release widen_narrow_random -- --ignored`.
fn widen_narrow_random<S: Simd>(simd: S) {
    let mut rng = fastrand::Rng::with_seed(0x243f_6a88_85a3_08d3);

    for iteration in 0..100_000 {
        let input_u8: [u8; 32] = core::array::from_fn(|_| rng.u8(..));
        let input_u8 = u8x32::from_slice(simd, &input_u8);
        let (low, high) = input_u8.widen();
        let expected_low: [u16; 16] = core::array::from_fn(|i| input_u8[i] as u16);
        let expected_high: [u16; 16] = core::array::from_fn(|i| input_u8[i + 16] as u16);
        assert_eq!(*low, expected_low, "u8 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "u8 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *input_u8,
            "u8 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *input_u8,
            "u8 saturating roundtrip iteration {iteration}",
        );

        let input_i8: [i8; 32] = core::array::from_fn(|_| rng.i8(..));
        let input_i8 = i8x32::from_slice(simd, &input_i8);
        let (low, high) = input_i8.widen();
        let expected_low: [i16; 16] = core::array::from_fn(|i| input_i8[i] as i16);
        let expected_high: [i16; 16] = core::array::from_fn(|i| input_i8[i + 16] as i16);
        assert_eq!(*low, expected_low, "i8 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "i8 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *input_i8,
            "i8 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *input_i8,
            "i8 saturating roundtrip iteration {iteration}",
        );

        let a_u16: [u16; 16] = core::array::from_fn(|_| rng.u16(..));
        let b_u16: [u16; 16] = core::array::from_fn(|_| rng.u16(..));
        let a_u16 = u16x16::from_slice(simd, &a_u16);
        let b_u16 = u16x16::from_slice(simd, &b_u16);
        let (low, high) = a_u16.widen();
        let expected_low: [u32; 8] = core::array::from_fn(|i| a_u16[i] as u32);
        let expected_high: [u32; 8] = core::array::from_fn(|i| a_u16[i + 8] as u32);
        assert_eq!(*low, expected_low, "u16 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "u16 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *a_u16,
            "u16 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *a_u16,
            "u16 saturating roundtrip iteration {iteration}",
        );
        let truncated = a_u16.narrow(b_u16);
        let saturated = a_u16.saturating_narrow(b_u16);
        let expected_truncated: [u8; 32] = core::array::from_fn(|i| {
            let value = if i < 16 { a_u16[i] } else { b_u16[i - 16] };
            value as u8
        });
        let expected_saturated: [u8; 32] = core::array::from_fn(|i| {
            let value = if i < 16 { a_u16[i] } else { b_u16[i - 16] };
            value.min(u8::MAX as u16) as u8
        });
        assert_eq!(
            *truncated, expected_truncated,
            "u16 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "u16 saturation iteration {iteration}",
        );

        let a_i16: [i16; 16] = core::array::from_fn(|_| rng.i16(..));
        let b_i16: [i16; 16] = core::array::from_fn(|_| rng.i16(..));
        let a_i16 = i16x16::from_slice(simd, &a_i16);
        let b_i16 = i16x16::from_slice(simd, &b_i16);
        let (low, high) = a_i16.widen();
        let expected_low: [i32; 8] = core::array::from_fn(|i| a_i16[i] as i32);
        let expected_high: [i32; 8] = core::array::from_fn(|i| a_i16[i + 8] as i32);
        assert_eq!(*low, expected_low, "i16 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "i16 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *a_i16,
            "i16 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *a_i16,
            "i16 saturating roundtrip iteration {iteration}",
        );
        let truncated = a_i16.narrow(b_i16);
        let saturated = a_i16.saturating_narrow(b_i16);
        let expected_truncated: [i8; 32] = core::array::from_fn(|i| {
            let value = if i < 16 { a_i16[i] } else { b_i16[i - 16] };
            value as i8
        });
        let expected_saturated: [i8; 32] = core::array::from_fn(|i| {
            let value = if i < 16 { a_i16[i] } else { b_i16[i - 16] };
            value.clamp(i8::MIN as i16, i8::MAX as i16) as i8
        });
        assert_eq!(
            *truncated, expected_truncated,
            "i16 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "i16 saturation iteration {iteration}",
        );

        let a_u32: [u32; 8] = core::array::from_fn(|_| rng.u32(..));
        let b_u32: [u32; 8] = core::array::from_fn(|_| rng.u32(..));
        let a_u32 = u32x8::from_slice(simd, &a_u32);
        let b_u32 = u32x8::from_slice(simd, &b_u32);
        let (low, high) = a_u32.widen();
        let expected_low: [u64; 4] = core::array::from_fn(|i| a_u32[i] as u64);
        let expected_high: [u64; 4] = core::array::from_fn(|i| a_u32[i + 4] as u64);
        assert_eq!(*low, expected_low, "u32 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "u32 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *a_u32,
            "u32 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *a_u32,
            "u32 saturating roundtrip iteration {iteration}",
        );
        let truncated = a_u32.narrow(b_u32);
        let saturated = a_u32.saturating_narrow(b_u32);
        let expected_truncated: [u16; 16] = core::array::from_fn(|i| {
            let value = if i < 8 { a_u32[i] } else { b_u32[i - 8] };
            value as u16
        });
        let expected_saturated: [u16; 16] = core::array::from_fn(|i| {
            let value = if i < 8 { a_u32[i] } else { b_u32[i - 8] };
            value.min(u16::MAX as u32) as u16
        });
        assert_eq!(
            *truncated, expected_truncated,
            "u32 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "u32 saturation iteration {iteration}",
        );

        let a_i32: [i32; 8] = core::array::from_fn(|_| rng.i32(..));
        let b_i32: [i32; 8] = core::array::from_fn(|_| rng.i32(..));
        let a_i32 = i32x8::from_slice(simd, &a_i32);
        let b_i32 = i32x8::from_slice(simd, &b_i32);
        let (low, high) = a_i32.widen();
        let expected_low: [i64; 4] = core::array::from_fn(|i| a_i32[i] as i64);
        let expected_high: [i64; 4] = core::array::from_fn(|i| a_i32[i + 4] as i64);
        assert_eq!(*low, expected_low, "i32 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "i32 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *a_i32,
            "i32 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *a_i32,
            "i32 saturating roundtrip iteration {iteration}",
        );
        let truncated = a_i32.narrow(b_i32);
        let saturated = a_i32.saturating_narrow(b_i32);
        let expected_truncated: [i16; 16] = core::array::from_fn(|i| {
            let value = if i < 8 { a_i32[i] } else { b_i32[i - 8] };
            value as i16
        });
        let expected_saturated: [i16; 16] = core::array::from_fn(|i| {
            let value = if i < 8 { a_i32[i] } else { b_i32[i - 8] };
            value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
        });
        assert_eq!(
            *truncated, expected_truncated,
            "i32 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "i32 saturation iteration {iteration}",
        );

        let a_u64: [u64; 4] = core::array::from_fn(|_| rng.u64(..));
        let b_u64: [u64; 4] = core::array::from_fn(|_| rng.u64(..));
        let a_u64 = u64x4::from_slice(simd, &a_u64);
        let b_u64 = u64x4::from_slice(simd, &b_u64);
        let truncated = a_u64.narrow(b_u64);
        let saturated = a_u64.saturating_narrow(b_u64);
        let expected_truncated: [u32; 8] = core::array::from_fn(|i| {
            let value = if i < 4 { a_u64[i] } else { b_u64[i - 4] };
            value as u32
        });
        let expected_saturated: [u32; 8] = core::array::from_fn(|i| {
            let value = if i < 4 { a_u64[i] } else { b_u64[i - 4] };
            value.min(u32::MAX as u64) as u32
        });
        assert_eq!(
            *truncated, expected_truncated,
            "u64 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "u64 saturation iteration {iteration}",
        );

        let a_i64: [i64; 4] = core::array::from_fn(|_| rng.i64(..));
        let b_i64: [i64; 4] = core::array::from_fn(|_| rng.i64(..));
        let a_i64 = i64x4::from_slice(simd, &a_i64);
        let b_i64 = i64x4::from_slice(simd, &b_i64);
        let truncated = a_i64.narrow(b_i64);
        let saturated = a_i64.saturating_narrow(b_i64);
        let expected_truncated: [i32; 8] = core::array::from_fn(|i| {
            let value = if i < 4 { a_i64[i] } else { b_i64[i - 4] };
            value as i32
        });
        let expected_saturated: [i32; 8] = core::array::from_fn(|i| {
            let value = if i < 4 { a_i64[i] } else { b_i64[i - 4] };
            value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
        });
        assert_eq!(
            *truncated, expected_truncated,
            "i64 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "i64 saturation iteration {iteration}",
        );

        let input_f32: [f32; 8] = core::array::from_fn(|_| f32::from_bits(rng.u32(..)));
        let input_f32 = f32x8::from_slice(simd, &input_f32);
        let (low, high) = input_f32.widen();
        for i in 0..f64x4::<S>::N {
            let expected_low = input_f32[i] as f64;
            let expected_high = input_f32[i + 4] as f64;
            if expected_low.is_nan() {
                assert!(low[i].is_nan(), "f32 low widening iteration {iteration}");
            } else {
                assert_eq!(
                    low[i].to_bits(),
                    expected_low.to_bits(),
                    "f32 low widening iteration {iteration}",
                );
            }
            if expected_high.is_nan() {
                assert!(high[i].is_nan(), "f32 high widening iteration {iteration}");
            } else {
                assert_eq!(
                    high[i].to_bits(),
                    expected_high.to_bits(),
                    "f32 high widening iteration {iteration}",
                );
            }
        }
        let roundtrip = low.narrow(high);
        let saturated_roundtrip = low.saturating_narrow(high);
        for i in 0..f32x8::<S>::N {
            if input_f32[i].is_nan() {
                assert!(roundtrip[i].is_nan(), "f32 roundtrip iteration {iteration}");
                assert!(
                    saturated_roundtrip[i].is_nan(),
                    "f32 saturating roundtrip iteration {iteration}",
                );
            } else {
                assert_eq!(
                    roundtrip[i].to_bits(),
                    input_f32[i].to_bits(),
                    "f32 roundtrip iteration {iteration}",
                );
                assert_eq!(
                    saturated_roundtrip[i].to_bits(),
                    input_f32[i].to_bits(),
                    "f32 saturating roundtrip iteration {iteration}",
                );
            }
        }

        let a_f64: [f64; 4] = core::array::from_fn(|_| f64::from_bits(rng.u64(..)));
        let b_f64: [f64; 4] = core::array::from_fn(|_| f64::from_bits(rng.u64(..)));
        let a_f64 = f64x4::from_slice(simd, &a_f64);
        let b_f64 = f64x4::from_slice(simd, &b_f64);
        let narrowed = a_f64.narrow(b_f64);
        let saturated = a_f64.saturating_narrow(b_f64);
        for i in 0..f32x8::<S>::N {
            let source = if i < 4 { a_f64[i] } else { b_f64[i - 4] };
            let expected = source as f32;
            if expected.is_nan() {
                assert!(narrowed[i].is_nan(), "f64 narrowing iteration {iteration}");
                assert!(
                    saturated[i].is_nan(),
                    "f64 saturating narrowing iteration {iteration}",
                );
            } else {
                assert_eq!(
                    narrowed[i].to_bits(),
                    expected.to_bits(),
                    "f64 narrowing iteration {iteration}",
                );
                assert_eq!(
                    saturated[i].to_bits(),
                    expected.to_bits(),
                    "f64 saturating narrowing iteration {iteration}",
                );
            }
        }
    }
}
