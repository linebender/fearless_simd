// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported integer vector type.

#[simd_test]
fn reduce_or_i8x16<S: Simd>(simd: S) {
    let value = i8x16::simd_from(
        simd,
        [
            2, 1, -128, 64, 32, 16, 8, 4, -126, 1, -128, 64, 32, 16, 8, 20,
        ],
    );
    assert_eq!(value.reduce_or(), -1);
}

#[simd_test]
fn reduce_or_u8x16<S: Simd>(simd: S) {
    let value = u8x16::simd_from(
        simd,
        [2, 1, 128, 64, 32, 16, 8, 4, 130, 1, 128, 64, 32, 16, 8, 20],
    );
    assert_eq!(value.reduce_or(), 255);
}

#[simd_test]
fn reduce_or_i8x32<S: Simd>(simd: S) {
    let value = i8x32::simd_from(
        simd,
        [
            2, 1, -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, -126, 1, -128, 64, 32, 16,
            8, 4, 2, 1, -128, 64, 32, 16, 8, 20,
        ],
    );
    assert_eq!(value.reduce_or(), -1);
}

#[simd_test]
fn reduce_or_u8x32<S: Simd>(simd: S) {
    let value = u8x32::simd_from(
        simd,
        [
            2, 1, 128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 130, 1, 128, 64, 32, 16, 8,
            4, 2, 1, 128, 64, 32, 16, 8, 20,
        ],
    );
    assert_eq!(value.reduce_or(), 255);
}

#[simd_test]
fn reduce_or_i8x64<S: Simd>(simd: S) {
    let value = i8x64::simd_from(
        simd,
        [
            2, 1, -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8,
            4, 2, 1, -128, 64, 32, 16, 8, 4, -126, 1, -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32,
            16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 20,
        ],
    );
    assert_eq!(value.reduce_or(), -1);
}

#[simd_test]
fn reduce_or_u8x64<S: Simd>(simd: S) {
    let value = u8x64::simd_from(
        simd,
        [
            2, 1, 128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4,
            2, 1, 128, 64, 32, 16, 8, 4, 130, 1, 128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8,
            4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 20,
        ],
    );
    assert_eq!(value.reduce_or(), 255);
}

#[simd_test]
fn reduce_or_i16x8<S: Simd>(simd: S) {
    let value = i16x8::simd_from(simd, [2, 256, -32768, 64, -24576, 16, 2048, 260]);
    assert_eq!(value.reduce_or(), -22186);
}

#[simd_test]
fn reduce_or_u16x8<S: Simd>(simd: S) {
    let value = u16x8::simd_from(simd, [2, 256, 32768, 64, 40960, 16, 2048, 260]);
    assert_eq!(value.reduce_or(), 43350);
}

#[simd_test]
fn reduce_or_i16x16<S: Simd>(simd: S) {
    let value = i16x16::simd_from(
        simd,
        [
            2, 256, -32768, 64, 8192, 16, 2048, 4, -32256, 1, 128, 16384, 32, 4096, 8, 1280,
        ],
    );
    assert_eq!(value.reduce_or(), -1);
}

#[simd_test]
fn reduce_or_u16x16<S: Simd>(simd: S) {
    let value = u16x16::simd_from(
        simd,
        [
            2, 256, 32768, 64, 8192, 16, 2048, 4, 33280, 1, 128, 16384, 32, 4096, 8, 1280,
        ],
    );
    assert_eq!(value.reduce_or(), 65535);
}

#[simd_test]
fn reduce_or_i16x32<S: Simd>(simd: S) {
    let value = i16x32::simd_from(
        simd,
        [
            2, 256, -32768, 64, 8192, 16, 2048, 4, 512, 1, 128, 16384, 32, 4096, 8, 1024, -32766,
            256, -32768, 64, 8192, 16, 2048, 4, 512, 1, 128, 16384, 32, 4096, 8, 1280,
        ],
    );
    assert_eq!(value.reduce_or(), -1);
}

#[simd_test]
fn reduce_or_u16x32<S: Simd>(simd: S) {
    let value = u16x32::simd_from(
        simd,
        [
            2, 256, 32768, 64, 8192, 16, 2048, 4, 512, 1, 128, 16384, 32, 4096, 8, 1024, 32770,
            256, 32768, 64, 8192, 16, 2048, 4, 512, 1, 128, 16384, 32, 4096, 8, 1280,
        ],
    );
    assert_eq!(value.reduce_or(), 65535);
}

#[simd_test]
fn reduce_or_i32x4<S: Simd>(simd: S) {
    let value = i32x4::simd_from(simd, [2, 256, -2147450880, 4259840]);
    assert_eq!(value.reduce_or(), -2143190782);
}

#[simd_test]
fn reduce_or_u32x4<S: Simd>(simd: S) {
    let value = u32x4::simd_from(simd, [2, 256, 2147516416, 4259840]);
    assert_eq!(value.reduce_or(), 2151776514);
}

#[simd_test]
fn reduce_or_i32x8<S: Simd>(simd: S) {
    let value = i32x8::simd_from(
        simd,
        [2, 256, 32768, 4194304, -1610612736, 16, 2048, 327680],
    );
    assert_eq!(value.reduce_or(), -1606055662);
}

#[simd_test]
fn reduce_or_u32x8<S: Simd>(simd: S) {
    let value = u32x8::simd_from(simd, [2, 256, 32768, 4194304, 2684354560, 16, 2048, 327680]);
    assert_eq!(value.reduce_or(), 2688911634);
}

#[simd_test]
fn reduce_or_i32x16<S: Simd>(simd: S) {
    let value = i32x16::simd_from(
        simd,
        [
            2,
            256,
            32768,
            4194304,
            536870912,
            16,
            2048,
            262144,
            -2113929216,
            1,
            128,
            16384,
            2097152,
            268435456,
            8,
            66560,
        ],
    );
    assert_eq!(value.reduce_or(), -1301951077);
}

#[simd_test]
fn reduce_or_u32x16<S: Simd>(simd: S) {
    let value = u32x16::simd_from(
        simd,
        [
            2, 256, 32768, 4194304, 536870912, 16, 2048, 262144, 2181038080, 1, 128, 16384,
            2097152, 268435456, 8, 66560,
        ],
    );
    assert_eq!(value.reduce_or(), 2993016219);
}

#[simd_test]
fn reduce_or_i64x2<S: Simd>(simd: S) {
    let value = i64x2::simd_from(simd, [2, -9223372032559808256]);
    assert_eq!(value.reduce_or(), -9223372032559808254);
}

#[simd_test]
fn reduce_or_u64x2<S: Simd>(simd: S) {
    let value = u64x2::simd_from(simd, [2, 9223372041149743360]);
    assert_eq!(value.reduce_or(), 9223372041149743362);
}

#[simd_test]
fn reduce_or_i64x4<S: Simd>(simd: S) {
    let value = i64x4::simd_from(simd, [2, 256, -9223372036854743040, 4299161600]);
    assert_eq!(value.reduce_or(), -9223372032555581182);
}

#[simd_test]
fn reduce_or_u64x4<S: Simd>(simd: S) {
    let value = u64x4::simd_from(simd, [2, 256, 9223372036854808576, 4299161600]);
    assert_eq!(value.reduce_or(), 9223372041153970434);
}

#[simd_test]
fn reduce_or_i64x8<S: Simd>(simd: S) {
    let value = i64x8::simd_from(
        simd,
        [
            2,
            256,
            32768,
            4194304,
            -9223372036317904896,
            68719476736,
            8796093022208,
            1125904201809920,
        ],
    );
    assert_eq!(value.reduce_or(), -9222237267299368702);
}

#[simd_test]
fn reduce_or_u64x8<S: Simd>(simd: S) {
    let value = u64x8::simd_from(
        simd,
        [
            2,
            256,
            32768,
            4194304,
            9223372037391646720,
            68719476736,
            8796093022208,
            1125904201809920,
        ],
    );
    assert_eq!(value.reduce_or(), 9224506806410182914);
}
