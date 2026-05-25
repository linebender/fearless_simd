// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::convert::TryFrom;
use core::mem::size_of;

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

fn lane_mask(lanes: usize) -> u64 {
    if lanes == u64::BITS as usize {
        u64::MAX
    } else {
        (1_u64 << lanes) - 1
    }
}

fn lanes_from_bits<L, const LANES: usize>(bits: u64) -> [L; LANES]
where
    L: Copy + From<i8>,
{
    let bits = bits & lane_mask(LANES);
    core::array::from_fn(|i| {
        if ((bits >> i) & 1) != 0 {
            L::from(-1)
        } else {
            L::from(0)
        }
    })
}

#[allow(
    clippy::disallowed_methods,
    reason = "test-only checked wrapper around transmute_copy"
)]
unsafe fn checked_transmute_copy<Src: Copy, Dst: Copy>(src: &Src) -> Dst {
    const {
        assert!(
            size_of::<Src>() == size_of::<Dst>(),
            "checked_transmute_copy requires source and destination to have the same size"
        );
    }
    // Safety: the caller upholds `transmute_copy`'s validity requirements, and
    // the const assertion above verifies that the source and destination sizes match.
    unsafe { core::mem::transmute_copy(src) }
}

fn assert_native_vector_roundtrip<S, M, A, L, const LANES: usize>(simd: S, bits: u64)
where
    S: Simd,
    M: SimdMask<S> + SimdFrom<A, S> + Into<A>,
    A: Copy,
    L: Copy + Eq + core::fmt::Debug + From<i8>,
{
    let expected_bits = bits & lane_mask(LANES);
    let expected_lanes = lanes_from_bits::<L, LANES>(bits);

    assert_eq!(size_of::<A>(), size_of::<[L; LANES]>());

    let mask = M::from_bitmask(simd, bits);
    let arch: A = mask.into();
    // Safety: the size assertion above verifies that the x86 vector type has
    // the same size as the signed integer lane representation used for masks.
    let lanes = unsafe { checked_transmute_copy::<A, [L; LANES]>(&arch) };
    assert_eq!(lanes, expected_lanes);

    // Safety: this builds the native x86 vector value from the lane
    // representation expected by the public mask conversion.
    let arch = unsafe { checked_transmute_copy::<[L; LANES], A>(&expected_lanes) };
    let mask = M::simd_from(simd, arch);
    assert_eq!(mask.to_bitmask(), expected_bits);
}

fn assert_native_mask_roundtrip<S, M, A, const LANES: usize>(simd: S, bits: u64)
where
    S: Simd,
    M: SimdMask<S> + SimdFrom<A, S> + Into<A>,
    A: Copy + Eq + core::fmt::Debug + TryFrom<u64>,
    A::Error: core::fmt::Debug,
{
    let expected_bits = bits & lane_mask(LANES);
    let expected_arch = A::try_from(expected_bits).expect("masked bits fit in native mask type");

    let mask = M::from_bitmask(simd, bits);
    let arch: A = mask.into();
    assert_eq!(arch, expected_arch);

    let mask = M::simd_from(simd, expected_arch);
    assert_eq!(mask.to_bitmask(), expected_bits);

    let arch: A = mask.into();
    assert_eq!(arch, expected_arch);
}

#[simd_test]
fn mask8x16_m128i_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_vector_roundtrip::<S, mask8x16<S>, __m128i, i8, 16>(simd, bits);
    }
}

#[simd_test]
fn mask16x8_m128i_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_vector_roundtrip::<S, mask16x8<S>, __m128i, i16, 8>(simd, bits);
    }
}

#[simd_test]
fn mask32x4_m128i_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_vector_roundtrip::<S, mask32x4<S>, __m128i, i32, 4>(simd, bits);
    }
}

#[simd_test]
fn mask64x2_m128i_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_vector_roundtrip::<S, mask64x2<S>, __m128i, i64, 2>(simd, bits);
    }
}

#[simd_test]
fn mask8x32_m256i_roundtrip<S: Simd>(simd: S) {
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0x0000_0000);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0x0000_0001);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0x8000_0000);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0x0000_ffff);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0xffff_0000);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0x5555_5555);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0xaaaa_aaaa);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0x8000_aa55);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0xffff_ffff);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0xffff_ffff_0000_0000);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0xffff_ffff_8000_aa55);
    assert_native_vector_roundtrip::<S, mask8x32<S>, __m256i, i8, 32>(simd, 0xffff_ffff_ffff_ffff);
}

#[simd_test]
fn mask16x16_m256i_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_vector_roundtrip::<S, mask16x16<S>, __m256i, i16, 16>(simd, bits);
    }
}

#[simd_test]
fn mask32x8_m256i_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_vector_roundtrip::<S, mask32x8<S>, __m256i, i32, 8>(simd, bits);
    }
}

#[simd_test]
fn mask64x4_m256i_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_vector_roundtrip::<S, mask64x4<S>, __m256i, i64, 4>(simd, bits);
    }
}

#[simd_test]
fn mask8x16_mmask16_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_mask_roundtrip::<S, mask8x16<S>, __mmask16, 16>(simd, bits);
    }
}

#[simd_test]
fn mask16x8_mmask8_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_mask_roundtrip::<S, mask16x8<S>, __mmask8, 8>(simd, bits);
    }
}

#[simd_test]
fn mask32x4_mmask8_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_mask_roundtrip::<S, mask32x4<S>, __mmask8, 4>(simd, bits);
    }
}

#[simd_test]
fn mask64x2_mmask8_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_mask_roundtrip::<S, mask64x2<S>, __mmask8, 2>(simd, bits);
    }
}

#[simd_test]
fn mask8x32_mmask32_roundtrip<S: Simd>(simd: S) {
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0x0000_0000);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0x0000_0001);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0x8000_0000);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0x0000_ffff);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0xffff_0000);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0x5555_5555);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0xaaaa_aaaa);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0x8000_aa55);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0xffff_ffff);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0xffff_ffff_0000_0000);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0xffff_ffff_8000_aa55);
    assert_native_mask_roundtrip::<S, mask8x32<S>, __mmask32, 32>(simd, 0xffff_ffff_ffff_ffff);
}

#[simd_test]
fn mask16x16_mmask16_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_mask_roundtrip::<S, mask16x16<S>, __mmask16, 16>(simd, bits);
    }
}

#[simd_test]
fn mask32x8_mmask8_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_mask_roundtrip::<S, mask32x8<S>, __mmask8, 8>(simd, bits);
    }
}

#[simd_test]
fn mask64x4_mmask8_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_mask_roundtrip::<S, mask64x4<S>, __mmask8, 4>(simd, bits);
    }
}

#[simd_test]
fn mask8x64_mmask64_roundtrip<S: Simd>(simd: S) {
    assert_native_mask_roundtrip::<S, mask8x64<S>, __mmask64, 64>(simd, 0x0000_0000_0000_0000);
    assert_native_mask_roundtrip::<S, mask8x64<S>, __mmask64, 64>(simd, 0x0000_0000_0000_0001);
    assert_native_mask_roundtrip::<S, mask8x64<S>, __mmask64, 64>(simd, 0x8000_0000_0000_0000);
    assert_native_mask_roundtrip::<S, mask8x64<S>, __mmask64, 64>(simd, 0x0000_0000_ffff_ffff);
    assert_native_mask_roundtrip::<S, mask8x64<S>, __mmask64, 64>(simd, 0xffff_ffff_0000_0000);
    assert_native_mask_roundtrip::<S, mask8x64<S>, __mmask64, 64>(simd, 0x5555_5555_5555_5555);
    assert_native_mask_roundtrip::<S, mask8x64<S>, __mmask64, 64>(simd, 0xaaaa_aaaa_aaaa_aaaa);
    assert_native_mask_roundtrip::<S, mask8x64<S>, __mmask64, 64>(simd, 0x8000_0001_5555_aaab);
    assert_native_mask_roundtrip::<S, mask8x64<S>, __mmask64, 64>(simd, 0xffff_ffff_ffff_ffff);
}

#[simd_test]
fn mask16x32_mmask32_roundtrip<S: Simd>(simd: S) {
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0x0000_0000);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0x0000_0001);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0x8000_0000);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0x0000_ffff);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0xffff_0000);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0x5555_5555);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0xaaaa_aaaa);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0x8000_aa55);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0xffff_ffff);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0xffff_ffff_0000_0000);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0xffff_ffff_8000_aa55);
    assert_native_mask_roundtrip::<S, mask16x32<S>, __mmask32, 32>(simd, 0xffff_ffff_ffff_ffff);
}

#[simd_test]
fn mask32x16_mmask16_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_mask_roundtrip::<S, mask32x16<S>, __mmask16, 16>(simd, bits);
    }
}

#[simd_test]
fn mask64x8_mmask8_roundtrip<S: Simd>(simd: S) {
    for bits in 0..=0xffff_u64 {
        assert_native_mask_roundtrip::<S, mask64x8<S>, __mmask8, 8>(simd, bits);
    }
}
