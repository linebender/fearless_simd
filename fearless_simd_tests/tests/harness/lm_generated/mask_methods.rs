// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn mask8x16_bitmask_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let mask = mask8x16::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask16x8_bitmask_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let mask = mask16x8::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits & 0xff);
    }
}

#[simd_test]
fn mask32x4_bitmask_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let mask = mask32x4::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits & 0xf);
    }
}

#[simd_test]
fn mask64x2_bitmask_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let mask = mask64x2::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits & 0x3);
    }
}

#[simd_test]
fn mask16x16_bitmask_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let mask = mask16x16::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask32x8_bitmask_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let mask = mask32x8::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits & 0xff);
    }
}

#[simd_test]
fn mask64x4_bitmask_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let mask = mask64x4::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits & 0xf);
    }
}

#[simd_test]
fn mask32x16_bitmask_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let mask = mask32x16::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask64x8_bitmask_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let mask = mask64x8::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits & 0xff);
    }
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
