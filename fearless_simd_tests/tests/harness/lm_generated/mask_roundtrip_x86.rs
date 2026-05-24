// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

const INTERESTING_32: &[u64] = &[
    0x0000_0000,
    0x0000_0001,
    0x8000_0000,
    0x0000_ffff,
    0xffff_0000,
    0x5555_5555,
    0xaaaa_aaaa,
    0x8000_aa55,
    0xffff_ffff,
    0xffff_ffff_0000_0000,
    0xffff_ffff_8000_aa55,
    0xffff_ffff_ffff_ffff,
];

const INTERESTING_64: &[u64] = &[
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x8000_0000_0000_0000,
    0x0000_0000_ffff_ffff,
    0xffff_ffff_0000_0000,
    0x5555_5555_5555_5555,
    0xaaaa_aaaa_aaaa_aaaa,
    0x8000_0001_5555_aaab,
    0xffff_ffff_ffff_ffff,
];

fn lane_mask(lanes: usize) -> u64 {
    if lanes == u64::BITS as usize {
        u64::MAX
    } else {
        (1_u64 << lanes) - 1
    }
}

trait MaskArch: Copy + Eq + core::fmt::Debug {
    fn from_bits(bits: u64) -> Self;
}

impl MaskArch for u8 {
    fn from_bits(bits: u64) -> Self {
        Self::try_from(bits).expect("masked bits fit in __mmask8")
    }
}

impl MaskArch for u16 {
    fn from_bits(bits: u64) -> Self {
        Self::try_from(bits).expect("masked bits fit in __mmask16")
    }
}

impl MaskArch for u32 {
    fn from_bits(bits: u64) -> Self {
        Self::try_from(bits).expect("masked bits fit in __mmask32")
    }
}

impl MaskArch for u64 {
    fn from_bits(bits: u64) -> Self {
        bits
    }
}

macro_rules! assert_native_vector_roundtrip {
    ($simd:expr, $mask:ident, $arch:ty, $lane:ty, $lanes:literal, $bits:expr) => {{
        let bits = $bits;
        let expected_bits = bits & lane_mask($lanes);
        let expected_lanes: [$lane; $lanes] = core::array::from_fn(|i| {
            if ((expected_bits >> i) & 1) != 0 {
                -1
            } else {
                0
            }
        });

        let mask = $mask::from_bitmask($simd, bits);
        let arch: $arch = mask.into();
        // Safety: these x86 vector types have the same size and lane layout as the signed
        // integer arrays used for mask values.
        let lanes = unsafe { core::mem::transmute::<$arch, [$lane; $lanes]>(arch) };
        assert_eq!(
            lanes,
            expected_lanes,
            "{} -> {} lane values for {bits:#018x}",
            stringify!($mask),
            stringify!($arch)
        );

        // Safety: this builds the native x86 vector value from the lane representation expected
        // by the public mask conversion.
        let arch = unsafe { core::mem::transmute::<[$lane; $lanes], $arch>(expected_lanes) };
        let mask = $mask::simd_from($simd, arch);
        assert_eq!(
            mask.to_bitmask(),
            expected_bits,
            "{} <- {} bitmask for {bits:#018x}",
            stringify!($mask),
            stringify!($arch)
        );
    }};
}

macro_rules! assert_native_mask_roundtrip {
    ($simd:expr, $mask:ident, $arch:ty, $lanes:literal, $bits:expr) => {{
        let bits = $bits;
        let expected_bits = bits & lane_mask($lanes);
        let expected_arch = <$arch as MaskArch>::from_bits(expected_bits);

        let mask = $mask::from_bitmask($simd, bits);
        let arch: $arch = mask.into();
        assert_eq!(
            arch,
            expected_arch,
            "{} -> {} for {bits:#018x}",
            stringify!($mask),
            stringify!($arch)
        );

        let mask = $mask::simd_from($simd, expected_arch);
        assert_eq!(
            mask.to_bitmask(),
            expected_bits,
            "{} <- {} bitmask for {bits:#018x}",
            stringify!($mask),
            stringify!($arch)
        );

        let arch: $arch = mask.into();
        assert_eq!(
            arch,
            expected_arch,
            "{} -> {} after roundtrip for {bits:#018x}",
            stringify!($mask),
            stringify!($arch)
        );
    }};
}

macro_rules! native_vector_roundtrip_exhaustive {
    ($test:ident, $mask:ident, $arch:ty, $lane:ty, $lanes:literal) => {
        #[simd_test]
        fn $test<S: Simd>(simd: S) {
            for bits in 0..=0xffff_u64 {
                assert_native_vector_roundtrip!(simd, $mask, $arch, $lane, $lanes, bits);
            }
        }
    };
}

macro_rules! native_vector_roundtrip_interesting {
    ($test:ident, $mask:ident, $arch:ty, $lane:ty, $lanes:literal, $values:ident) => {
        #[simd_test]
        fn $test<S: Simd>(simd: S) {
            for &bits in $values {
                assert_native_vector_roundtrip!(simd, $mask, $arch, $lane, $lanes, bits);
            }
        }
    };
}

macro_rules! native_mask_roundtrip_exhaustive {
    ($test:ident, $mask:ident, $arch:ty, $lanes:literal) => {
        #[simd_test]
        fn $test<S: Simd>(simd: S) {
            for bits in 0..=0xffff_u64 {
                assert_native_mask_roundtrip!(simd, $mask, $arch, $lanes, bits);
            }
        }
    };
}

macro_rules! native_mask_roundtrip_interesting {
    ($test:ident, $mask:ident, $arch:ty, $lanes:literal, $values:ident) => {
        #[simd_test]
        fn $test<S: Simd>(simd: S) {
            for &bits in $values {
                assert_native_mask_roundtrip!(simd, $mask, $arch, $lanes, bits);
            }
        }
    };
}

native_vector_roundtrip_exhaustive!(mask8x16_m128i_roundtrip, mask8x16, __m128i, i8, 16);
native_vector_roundtrip_exhaustive!(mask16x8_m128i_roundtrip, mask16x8, __m128i, i16, 8);
native_vector_roundtrip_exhaustive!(mask32x4_m128i_roundtrip, mask32x4, __m128i, i32, 4);
native_vector_roundtrip_exhaustive!(mask64x2_m128i_roundtrip, mask64x2, __m128i, i64, 2);

native_vector_roundtrip_interesting!(
    mask8x32_m256i_roundtrip,
    mask8x32,
    __m256i,
    i8,
    32,
    INTERESTING_32
);
native_vector_roundtrip_exhaustive!(mask16x16_m256i_roundtrip, mask16x16, __m256i, i16, 16);
native_vector_roundtrip_exhaustive!(mask32x8_m256i_roundtrip, mask32x8, __m256i, i32, 8);
native_vector_roundtrip_exhaustive!(mask64x4_m256i_roundtrip, mask64x4, __m256i, i64, 4);

native_mask_roundtrip_exhaustive!(mask8x16_mmask16_roundtrip, mask8x16, __mmask16, 16);
native_mask_roundtrip_exhaustive!(mask16x8_mmask8_roundtrip, mask16x8, __mmask8, 8);
native_mask_roundtrip_exhaustive!(mask32x4_mmask8_roundtrip, mask32x4, __mmask8, 4);
native_mask_roundtrip_exhaustive!(mask64x2_mmask8_roundtrip, mask64x2, __mmask8, 2);
native_mask_roundtrip_interesting!(
    mask8x32_mmask32_roundtrip,
    mask8x32,
    __mmask32,
    32,
    INTERESTING_32
);
native_mask_roundtrip_exhaustive!(mask16x16_mmask16_roundtrip, mask16x16, __mmask16, 16);
native_mask_roundtrip_exhaustive!(mask32x8_mmask8_roundtrip, mask32x8, __mmask8, 8);
native_mask_roundtrip_exhaustive!(mask64x4_mmask8_roundtrip, mask64x4, __mmask8, 4);
native_mask_roundtrip_interesting!(
    mask8x64_mmask64_roundtrip,
    mask8x64,
    __mmask64,
    64,
    INTERESTING_64
);
native_mask_roundtrip_interesting!(
    mask16x32_mmask32_roundtrip,
    mask16x32,
    __mmask32,
    32,
    INTERESTING_32
);
native_mask_roundtrip_exhaustive!(mask32x16_mmask16_roundtrip, mask32x16, __mmask16, 16);
native_mask_roundtrip_exhaustive!(mask64x8_mmask8_roundtrip, mask64x8, __mmask8, 8);
