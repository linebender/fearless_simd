// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

/// Verifies that `SimdMask::set` can set and clear every lane while keeping
/// `to_bitmask` and `test` in sync with the expected compact bitmask.
fn assert_mask_set_roundtrip<S: Simd, M: SimdMask<S>>(simd: S) {
    let mut mask = M::from_bitmask(simd, 0);
    let mut expected = 0_u64;
    for i in 0..M::N {
        mask.set(i, true);
        expected |= 1_u64 << i;
        assert_eq!(mask.to_bitmask(), expected);
        assert!(mask.test(i));
    }

    for i in 0..M::N {
        mask.set(i, false);
        expected &= !(1_u64 << i);
        assert_eq!(mask.to_bitmask(), expected);
        assert!(!mask.test(i));
    }
}

#[simd_test]
fn mask8x16_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask8x16<S>>(simd);
}

#[simd_test]
fn mask16x8_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask16x8<S>>(simd);
}

#[simd_test]
fn mask32x4_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask32x4<S>>(simd);
}

#[simd_test]
fn mask64x2_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask64x2<S>>(simd);
}

#[simd_test]
fn mask8x32_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask8x32<S>>(simd);
}

#[simd_test]
fn mask16x16_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask16x16<S>>(simd);
}

#[simd_test]
fn mask32x8_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask32x8<S>>(simd);
}

#[simd_test]
fn mask64x4_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask64x4<S>>(simd);
}

#[simd_test]
fn mask8x64_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask8x64<S>>(simd);
}

#[simd_test]
fn mask16x32_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask16x32<S>>(simd);
}

#[simd_test]
fn mask32x16_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask32x16<S>>(simd);
}

#[simd_test]
fn mask64x8_set_roundtrip<S: Simd>(simd: S) {
    assert_mask_set_roundtrip::<S, mask64x8<S>>(simd);
}

#[simd_test]
fn mask8x16_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let values: [i8; 16] =
            core::array::from_fn(|i| if bits & (1_u64 << i) == 0 { 0 } else { -1 });
        let mask = mask8x16::from_slice(simd, &values);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask16x8_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0xff_u64 {
        let values: [i16; 8] =
            core::array::from_fn(|i| if bits & (1_u64 << i) == 0 { 0 } else { -1 });
        let mask = mask16x8::from_slice(simd, &values);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask32x4_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0xf_u64 {
        let values: [i32; 4] =
            core::array::from_fn(|i| if bits & (1_u64 << i) == 0 { 0 } else { -1 });
        let mask = mask32x4::from_slice(simd, &values);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask64x2_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0x3_u64 {
        let values: [i64; 2] =
            core::array::from_fn(|i| if bits & (1_u64 << i) == 0 { 0 } else { -1 });
        let mask = mask64x2::from_slice(simd, &values);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask16x16_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let values: [i16; 16] =
            core::array::from_fn(|i| if bits & (1_u64 << i) == 0 { 0 } else { -1 });
        let mask = mask16x16::from_slice(simd, &values);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask32x8_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0xff_u64 {
        let values: [i32; 8] =
            core::array::from_fn(|i| if bits & (1_u64 << i) == 0 { 0 } else { -1 });
        let mask = mask32x8::from_slice(simd, &values);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask64x4_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0xf_u64 {
        let values: [i64; 4] =
            core::array::from_fn(|i| if bits & (1_u64 << i) == 0 { 0 } else { -1 });
        let mask = mask64x4::from_slice(simd, &values);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask32x16_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        let values: [i32; 16] =
            core::array::from_fn(|i| if bits & (1_u64 << i) == 0 { 0 } else { -1 });
        let mask = mask32x16::from_slice(simd, &values);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
fn mask64x8_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0xff_u64 {
        let values: [i64; 8] =
            core::array::from_fn(|i| if bits & (1_u64 << i) == 0 { 0 } else { -1 });
        let mask = mask64x8::from_slice(simd, &values);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
#[ignore = "exhaustive test; run in release mode"]
fn mask8x32_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0xffff_ffff_u64 {
        let mask = mask8x32::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
#[ignore = "exhaustive test; run in release mode"]
fn mask16x32_to_bitmask_exhaustive<S: Simd>(simd: S) {
    for bits in 0..=0xffff_ffff_u64 {
        let mask = mask16x32::from_bitmask(simd, bits);
        assert_eq!(mask.to_bitmask(), bits);
    }
}

#[simd_test]
#[ignore = "random stress test; run in release mode"]
fn mask8x64_to_bitmask_random<S: Simd>(simd: S) {
    let mut rng = fastrand::Rng::with_seed(0x243f_6a88_85a3_08d3);

    for iteration in 0..100_000_000 {
        let bits = rng.u64(..);
        let values: [i8; 64] =
            core::array::from_fn(|i| if bits & (1_u64 << i) == 0 { 0 } else { -1 });
        let mask = mask8x64::from_slice(simd, &values);
        assert_eq!(mask.to_bitmask(), bits, "iteration {iteration}");
    }
}

// selected interesting bit patterns to test always
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

// selected interesting bit patterns to test always
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
