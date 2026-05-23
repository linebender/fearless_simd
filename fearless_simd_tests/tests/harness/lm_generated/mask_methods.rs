// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn mask8x16_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask8x16::from_bitmask(simd, 0x0000);
    assert_eq!(mask.to_bitmask(), 0x0000);

    let mask = mask8x16::from_bitmask(simd, 0x0001);
    assert_eq!(mask.to_bitmask(), 0x0001);

    let mask = mask8x16::from_bitmask(simd, 0x8000);
    assert_eq!(mask.to_bitmask(), 0x8000);

    let mask = mask8x16::from_bitmask(simd, 0x00ff);
    assert_eq!(mask.to_bitmask(), 0x00ff);

    let mask = mask8x16::from_bitmask(simd, 0xff00);
    assert_eq!(mask.to_bitmask(), 0xff00);

    let mask = mask8x16::from_bitmask(simd, 0x5555);
    assert_eq!(mask.to_bitmask(), 0x5555);

    let mask = mask8x16::from_bitmask(simd, 0xaaaa);
    assert_eq!(mask.to_bitmask(), 0xaaaa);

    let mask = mask8x16::from_bitmask(simd, 0xaa55);
    assert_eq!(mask.to_bitmask(), 0xaa55);

    let mask = mask8x16::from_bitmask(simd, 0xffff);
    assert_eq!(mask.to_bitmask(), 0xffff);

    let mask = mask8x16::from_bitmask(simd, 0xffff_0000);
    assert_eq!(mask.to_bitmask(), 0x0000);

    let mask = mask8x16::from_bitmask(simd, 0xffff_aa55);
    assert_eq!(mask.to_bitmask(), 0xaa55);

    let mask = mask8x16::from_bitmask(simd, 0xffff_ffff);
    assert_eq!(mask.to_bitmask(), 0xffff);
}

#[simd_test]
fn mask16x8_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask16x8::from_bitmask(simd, 0x00);
    assert_eq!(mask.to_bitmask(), 0x00);

    let mask = mask16x8::from_bitmask(simd, 0x01);
    assert_eq!(mask.to_bitmask(), 0x01);

    let mask = mask16x8::from_bitmask(simd, 0x80);
    assert_eq!(mask.to_bitmask(), 0x80);

    let mask = mask16x8::from_bitmask(simd, 0x0f);
    assert_eq!(mask.to_bitmask(), 0x0f);

    let mask = mask16x8::from_bitmask(simd, 0xf0);
    assert_eq!(mask.to_bitmask(), 0xf0);

    let mask = mask16x8::from_bitmask(simd, 0x55);
    assert_eq!(mask.to_bitmask(), 0x55);

    let mask = mask16x8::from_bitmask(simd, 0xaa);
    assert_eq!(mask.to_bitmask(), 0xaa);

    let mask = mask16x8::from_bitmask(simd, 0xa5);
    assert_eq!(mask.to_bitmask(), 0xa5);

    let mask = mask16x8::from_bitmask(simd, 0xff);
    assert_eq!(mask.to_bitmask(), 0xff);

    let mask = mask16x8::from_bitmask(simd, 0xff00);
    assert_eq!(mask.to_bitmask(), 0x00);

    let mask = mask16x8::from_bitmask(simd, 0xffa5);
    assert_eq!(mask.to_bitmask(), 0xa5);

    let mask = mask16x8::from_bitmask(simd, 0xffff);
    assert_eq!(mask.to_bitmask(), 0xff);
}

#[simd_test]
fn mask32x4_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask32x4::from_bitmask(simd, 0x0);
    assert_eq!(mask.to_bitmask(), 0x0);

    let mask = mask32x4::from_bitmask(simd, 0x1);
    assert_eq!(mask.to_bitmask(), 0x1);

    let mask = mask32x4::from_bitmask(simd, 0x8);
    assert_eq!(mask.to_bitmask(), 0x8);

    let mask = mask32x4::from_bitmask(simd, 0x5);
    assert_eq!(mask.to_bitmask(), 0x5);

    let mask = mask32x4::from_bitmask(simd, 0xa);
    assert_eq!(mask.to_bitmask(), 0xa);

    let mask = mask32x4::from_bitmask(simd, 0xd);
    assert_eq!(mask.to_bitmask(), 0xd);

    let mask = mask32x4::from_bitmask(simd, 0xf);
    assert_eq!(mask.to_bitmask(), 0xf);

    let mask = mask32x4::from_bitmask(simd, 0xf0);
    assert_eq!(mask.to_bitmask(), 0x0);

    let mask = mask32x4::from_bitmask(simd, 0xfd);
    assert_eq!(mask.to_bitmask(), 0xd);

    let mask = mask32x4::from_bitmask(simd, 0xff);
    assert_eq!(mask.to_bitmask(), 0xf);
}

#[simd_test]
fn mask64x2_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask64x2::from_bitmask(simd, 0x0);
    assert_eq!(mask.to_bitmask(), 0x0);

    let mask = mask64x2::from_bitmask(simd, 0x1);
    assert_eq!(mask.to_bitmask(), 0x1);

    let mask = mask64x2::from_bitmask(simd, 0x2);
    assert_eq!(mask.to_bitmask(), 0x2);

    let mask = mask64x2::from_bitmask(simd, 0x3);
    assert_eq!(mask.to_bitmask(), 0x3);

    let mask = mask64x2::from_bitmask(simd, 0xfc);
    assert_eq!(mask.to_bitmask(), 0x0);

    let mask = mask64x2::from_bitmask(simd, 0xfd);
    assert_eq!(mask.to_bitmask(), 0x1);

    let mask = mask64x2::from_bitmask(simd, 0xff);
    assert_eq!(mask.to_bitmask(), 0x3);
}

#[simd_test]
fn mask8x32_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask8x32::from_bitmask(simd, 0x0000_0000);
    assert_eq!(mask.to_bitmask(), 0x0000_0000);

    let mask = mask8x32::from_bitmask(simd, 0x0000_0001);
    assert_eq!(mask.to_bitmask(), 0x0000_0001);

    let mask = mask8x32::from_bitmask(simd, 0x8000_0000);
    assert_eq!(mask.to_bitmask(), 0x8000_0000);

    let mask = mask8x32::from_bitmask(simd, 0x0000_ffff);
    assert_eq!(mask.to_bitmask(), 0x0000_ffff);

    let mask = mask8x32::from_bitmask(simd, 0xffff_0000);
    assert_eq!(mask.to_bitmask(), 0xffff_0000);

    let mask = mask8x32::from_bitmask(simd, 0x5555_5555);
    assert_eq!(mask.to_bitmask(), 0x5555_5555);

    let mask = mask8x32::from_bitmask(simd, 0xaaaa_aaaa);
    assert_eq!(mask.to_bitmask(), 0xaaaa_aaaa);

    let mask = mask8x32::from_bitmask(simd, 0x8000_aa55);
    assert_eq!(mask.to_bitmask(), 0x8000_aa55);

    let mask = mask8x32::from_bitmask(simd, 0xffff_ffff);
    assert_eq!(mask.to_bitmask(), 0xffff_ffff);

    let mask = mask8x32::from_bitmask(simd, 0xffff_ffff_0000_0000);
    assert_eq!(mask.to_bitmask(), 0x0000_0000);

    let mask = mask8x32::from_bitmask(simd, 0xffff_ffff_8000_aa55);
    assert_eq!(mask.to_bitmask(), 0x8000_aa55);

    let mask = mask8x32::from_bitmask(simd, 0xffff_ffff_ffff_ffff);
    assert_eq!(mask.to_bitmask(), 0xffff_ffff);
}

#[simd_test]
fn mask16x16_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask16x16::from_bitmask(simd, 0x0000);
    assert_eq!(mask.to_bitmask(), 0x0000);

    let mask = mask16x16::from_bitmask(simd, 0x0001);
    assert_eq!(mask.to_bitmask(), 0x0001);

    let mask = mask16x16::from_bitmask(simd, 0x8000);
    assert_eq!(mask.to_bitmask(), 0x8000);

    let mask = mask16x16::from_bitmask(simd, 0x00ff);
    assert_eq!(mask.to_bitmask(), 0x00ff);

    let mask = mask16x16::from_bitmask(simd, 0xff00);
    assert_eq!(mask.to_bitmask(), 0xff00);

    let mask = mask16x16::from_bitmask(simd, 0x5555);
    assert_eq!(mask.to_bitmask(), 0x5555);

    let mask = mask16x16::from_bitmask(simd, 0xaaaa);
    assert_eq!(mask.to_bitmask(), 0xaaaa);

    let mask = mask16x16::from_bitmask(simd, 0xaa55);
    assert_eq!(mask.to_bitmask(), 0xaa55);

    let mask = mask16x16::from_bitmask(simd, 0xffff);
    assert_eq!(mask.to_bitmask(), 0xffff);

    let mask = mask16x16::from_bitmask(simd, 0xffff_0000);
    assert_eq!(mask.to_bitmask(), 0x0000);

    let mask = mask16x16::from_bitmask(simd, 0xffff_aa55);
    assert_eq!(mask.to_bitmask(), 0xaa55);

    let mask = mask16x16::from_bitmask(simd, 0xffff_ffff);
    assert_eq!(mask.to_bitmask(), 0xffff);
}

#[simd_test]
fn mask32x8_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask32x8::from_bitmask(simd, 0x00);
    assert_eq!(mask.to_bitmask(), 0x00);

    let mask = mask32x8::from_bitmask(simd, 0x01);
    assert_eq!(mask.to_bitmask(), 0x01);

    let mask = mask32x8::from_bitmask(simd, 0x80);
    assert_eq!(mask.to_bitmask(), 0x80);

    let mask = mask32x8::from_bitmask(simd, 0x0f);
    assert_eq!(mask.to_bitmask(), 0x0f);

    let mask = mask32x8::from_bitmask(simd, 0xf0);
    assert_eq!(mask.to_bitmask(), 0xf0);

    let mask = mask32x8::from_bitmask(simd, 0x55);
    assert_eq!(mask.to_bitmask(), 0x55);

    let mask = mask32x8::from_bitmask(simd, 0xaa);
    assert_eq!(mask.to_bitmask(), 0xaa);

    let mask = mask32x8::from_bitmask(simd, 0xa5);
    assert_eq!(mask.to_bitmask(), 0xa5);

    let mask = mask32x8::from_bitmask(simd, 0xff);
    assert_eq!(mask.to_bitmask(), 0xff);

    let mask = mask32x8::from_bitmask(simd, 0xff00);
    assert_eq!(mask.to_bitmask(), 0x00);

    let mask = mask32x8::from_bitmask(simd, 0xffa5);
    assert_eq!(mask.to_bitmask(), 0xa5);

    let mask = mask32x8::from_bitmask(simd, 0xffff);
    assert_eq!(mask.to_bitmask(), 0xff);
}

#[simd_test]
fn mask64x4_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask64x4::from_bitmask(simd, 0x0);
    assert_eq!(mask.to_bitmask(), 0x0);

    let mask = mask64x4::from_bitmask(simd, 0x1);
    assert_eq!(mask.to_bitmask(), 0x1);

    let mask = mask64x4::from_bitmask(simd, 0x8);
    assert_eq!(mask.to_bitmask(), 0x8);

    let mask = mask64x4::from_bitmask(simd, 0x5);
    assert_eq!(mask.to_bitmask(), 0x5);

    let mask = mask64x4::from_bitmask(simd, 0xa);
    assert_eq!(mask.to_bitmask(), 0xa);

    let mask = mask64x4::from_bitmask(simd, 0xd);
    assert_eq!(mask.to_bitmask(), 0xd);

    let mask = mask64x4::from_bitmask(simd, 0xf);
    assert_eq!(mask.to_bitmask(), 0xf);

    let mask = mask64x4::from_bitmask(simd, 0xf0);
    assert_eq!(mask.to_bitmask(), 0x0);

    let mask = mask64x4::from_bitmask(simd, 0xfd);
    assert_eq!(mask.to_bitmask(), 0xd);

    let mask = mask64x4::from_bitmask(simd, 0xff);
    assert_eq!(mask.to_bitmask(), 0xf);
}

#[simd_test]
fn mask8x64_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask8x64::from_bitmask(simd, 0x0000_0000_0000_0000);
    assert_eq!(mask.to_bitmask(), 0x0000_0000_0000_0000);

    let mask = mask8x64::from_bitmask(simd, 0x0000_0000_0000_0001);
    assert_eq!(mask.to_bitmask(), 0x0000_0000_0000_0001);

    let mask = mask8x64::from_bitmask(simd, 0x8000_0000_0000_0000);
    assert_eq!(mask.to_bitmask(), 0x8000_0000_0000_0000);

    let mask = mask8x64::from_bitmask(simd, 0x0000_0000_ffff_ffff);
    assert_eq!(mask.to_bitmask(), 0x0000_0000_ffff_ffff);

    let mask = mask8x64::from_bitmask(simd, 0xffff_ffff_0000_0000);
    assert_eq!(mask.to_bitmask(), 0xffff_ffff_0000_0000);

    let mask = mask8x64::from_bitmask(simd, 0x5555_5555_5555_5555);
    assert_eq!(mask.to_bitmask(), 0x5555_5555_5555_5555);

    let mask = mask8x64::from_bitmask(simd, 0xaaaa_aaaa_aaaa_aaaa);
    assert_eq!(mask.to_bitmask(), 0xaaaa_aaaa_aaaa_aaaa);

    let mask = mask8x64::from_bitmask(simd, 0x8000_0001_5555_aaab);
    assert_eq!(mask.to_bitmask(), 0x8000_0001_5555_aaab);

    let mask = mask8x64::from_bitmask(simd, 0xffff_ffff_ffff_ffff);
    assert_eq!(mask.to_bitmask(), 0xffff_ffff_ffff_ffff);
}

#[simd_test]
fn mask16x32_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask16x32::from_bitmask(simd, 0x0000_0000);
    assert_eq!(mask.to_bitmask(), 0x0000_0000);

    let mask = mask16x32::from_bitmask(simd, 0x0000_0001);
    assert_eq!(mask.to_bitmask(), 0x0000_0001);

    let mask = mask16x32::from_bitmask(simd, 0x8000_0000);
    assert_eq!(mask.to_bitmask(), 0x8000_0000);

    let mask = mask16x32::from_bitmask(simd, 0x0000_ffff);
    assert_eq!(mask.to_bitmask(), 0x0000_ffff);

    let mask = mask16x32::from_bitmask(simd, 0xffff_0000);
    assert_eq!(mask.to_bitmask(), 0xffff_0000);

    let mask = mask16x32::from_bitmask(simd, 0x5555_5555);
    assert_eq!(mask.to_bitmask(), 0x5555_5555);

    let mask = mask16x32::from_bitmask(simd, 0xaaaa_aaaa);
    assert_eq!(mask.to_bitmask(), 0xaaaa_aaaa);

    let mask = mask16x32::from_bitmask(simd, 0x8000_aa55);
    assert_eq!(mask.to_bitmask(), 0x8000_aa55);

    let mask = mask16x32::from_bitmask(simd, 0xffff_ffff);
    assert_eq!(mask.to_bitmask(), 0xffff_ffff);

    let mask = mask16x32::from_bitmask(simd, 0xffff_ffff_0000_0000);
    assert_eq!(mask.to_bitmask(), 0x0000_0000);

    let mask = mask16x32::from_bitmask(simd, 0xffff_ffff_8000_aa55);
    assert_eq!(mask.to_bitmask(), 0x8000_aa55);

    let mask = mask16x32::from_bitmask(simd, 0xffff_ffff_ffff_ffff);
    assert_eq!(mask.to_bitmask(), 0xffff_ffff);
}

#[simd_test]
fn mask32x16_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask32x16::from_bitmask(simd, 0x0000);
    assert_eq!(mask.to_bitmask(), 0x0000);

    let mask = mask32x16::from_bitmask(simd, 0x0001);
    assert_eq!(mask.to_bitmask(), 0x0001);

    let mask = mask32x16::from_bitmask(simd, 0x8000);
    assert_eq!(mask.to_bitmask(), 0x8000);

    let mask = mask32x16::from_bitmask(simd, 0x00ff);
    assert_eq!(mask.to_bitmask(), 0x00ff);

    let mask = mask32x16::from_bitmask(simd, 0xff00);
    assert_eq!(mask.to_bitmask(), 0xff00);

    let mask = mask32x16::from_bitmask(simd, 0x5555);
    assert_eq!(mask.to_bitmask(), 0x5555);

    let mask = mask32x16::from_bitmask(simd, 0xaaaa);
    assert_eq!(mask.to_bitmask(), 0xaaaa);

    let mask = mask32x16::from_bitmask(simd, 0xaa55);
    assert_eq!(mask.to_bitmask(), 0xaa55);

    let mask = mask32x16::from_bitmask(simd, 0xffff);
    assert_eq!(mask.to_bitmask(), 0xffff);

    let mask = mask32x16::from_bitmask(simd, 0xffff_0000);
    assert_eq!(mask.to_bitmask(), 0x0000);

    let mask = mask32x16::from_bitmask(simd, 0xffff_aa55);
    assert_eq!(mask.to_bitmask(), 0xaa55);

    let mask = mask32x16::from_bitmask(simd, 0xffff_ffff);
    assert_eq!(mask.to_bitmask(), 0xffff);
}

#[simd_test]
fn mask64x8_bitmask_roundtrip<S: Simd>(simd: S) {
    let mask = mask64x8::from_bitmask(simd, 0x00);
    assert_eq!(mask.to_bitmask(), 0x00);

    let mask = mask64x8::from_bitmask(simd, 0x01);
    assert_eq!(mask.to_bitmask(), 0x01);

    let mask = mask64x8::from_bitmask(simd, 0x80);
    assert_eq!(mask.to_bitmask(), 0x80);

    let mask = mask64x8::from_bitmask(simd, 0x0f);
    assert_eq!(mask.to_bitmask(), 0x0f);

    let mask = mask64x8::from_bitmask(simd, 0xf0);
    assert_eq!(mask.to_bitmask(), 0xf0);

    let mask = mask64x8::from_bitmask(simd, 0x55);
    assert_eq!(mask.to_bitmask(), 0x55);

    let mask = mask64x8::from_bitmask(simd, 0xaa);
    assert_eq!(mask.to_bitmask(), 0xaa);

    let mask = mask64x8::from_bitmask(simd, 0xa5);
    assert_eq!(mask.to_bitmask(), 0xa5);

    let mask = mask64x8::from_bitmask(simd, 0xff);
    assert_eq!(mask.to_bitmask(), 0xff);

    let mask = mask64x8::from_bitmask(simd, 0xff00);
    assert_eq!(mask.to_bitmask(), 0x00);

    let mask = mask64x8::from_bitmask(simd, 0xffa5);
    assert_eq!(mask.to_bitmask(), 0xa5);

    let mask = mask64x8::from_bitmask(simd, 0xffff);
    assert_eq!(mask.to_bitmask(), 0xff);
}
