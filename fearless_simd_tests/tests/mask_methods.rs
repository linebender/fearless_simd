// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

fn mask_bits(len: usize) -> u64 {
    if len == 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    }
}

const CHUNK_PATTERNS_16: [u64; 8] = [
    0x0000, 0x0001, 0x00ff, 0x5555, 0x8000, 0xaaaa, 0xff00, 0xffff,
];

fn for_each_exhaustive_bitmask<F: FnMut(u64)>(len: usize, mut f: F) {
    assert!(
        len <= 16,
        "exhaustive bitmask roundtrip tests are only practical up to 16 lanes"
    );

    let all_bits = mask_bits(len);
    for bits in 0..(1u64 << len) {
        f(bits);
        f(bits | !all_bits);
    }
}

fn for_each_chunked_bitmask<F: FnMut(u64)>(len: usize, mut f: F) {
    assert!(
        len % 16 == 0,
        "chunked bitmask roundtrip tests expect 16-lane chunks"
    );
    assert!(
        len <= 64,
        "chunked bitmask roundtrip tests only support u64 bitmasks"
    );

    let chunks = len / 16;
    let mut pattern_count = 1usize;
    for _ in 0..chunks {
        pattern_count *= CHUNK_PATTERNS_16.len();
    }

    for mut pattern_index in 0..pattern_count {
        let mut bits = 0u64;
        for chunk in 0..chunks {
            let chunk_pattern = CHUNK_PATTERNS_16[pattern_index % CHUNK_PATTERNS_16.len()];
            pattern_index /= CHUNK_PATTERNS_16.len();
            bits |= chunk_pattern << (chunk * 16);
        }
        f(bits);
    }
}

fn for_each_wide_bitmask<F: FnMut(u64)>(len: usize, mut f: F) {
    let all_bits = mask_bits(len);
    let mut check = |bits| {
        f(bits);
        f(bits | !all_bits);
    };

    check(0);
    check(all_bits);
    check(all_bits & 0x5555_5555_5555_5555);
    check(all_bits & 0xaaaa_aaaa_aaaa_aaaa);

    for bit in 0..len {
        let bits = 1u64 << bit;
        check(bits);
        check(all_bits ^ bits);
    }

    for_each_chunked_bitmask(len, check);
}

macro_rules! check_bitmask_roundtrip {
    ($simd:expr, $mask:ident, $len:expr, $bits:expr) => {{
        let raw_bits = $bits;
        let expected = raw_bits & mask_bits($len);
        let mask = <$mask<_> as SimdMask<_>>::from_bitmask($simd, raw_bits);

        assert_eq!(
            mask.to_bitmask(),
            expected,
            "{}::from_bitmask({raw_bits:#018x}).to_bitmask()",
            stringify!($mask)
        );
        assert_eq!(
            <$mask<_> as SimdMask<_>>::from_bitmask($simd, mask.to_bitmask()).to_bitmask(),
            expected,
            "{}::from_bitmask({raw_bits:#018x}).to_bitmask() roundtripped again",
            stringify!($mask)
        );
    }};
}

macro_rules! check_mask_methods {
    ($simd:expr, $mask:ident, $len:expr, $bits:expr) => {{
        let all_bits = mask_bits($len);
        let mut expected = $bits & all_bits;
        let mut mask = <$mask<_> as SimdMask<_>>::from_bitmask($simd, $bits);

        assert_eq!(mask.to_bitmask(), expected);
        for i in 0..$len {
            assert_eq!(mask.test(i), ((expected >> i) & 1) != 0);
        }

        mask.set(0, false);
        expected &= !1;
        assert_eq!(mask.to_bitmask(), expected);

        mask.set($len - 1, true);
        expected |= 1u64 << ($len - 1);
        assert_eq!(mask.to_bitmask(), expected);

        mask.set(1, true);
        expected |= 1u64 << 1;
        assert!(mask.test(1));
        assert_eq!(mask.to_bitmask(), expected);

        mask.set(1, false);
        expected &= !(1u64 << 1);
        assert!(!mask.test(1));
        assert_eq!(mask.to_bitmask(), expected);
    }};
}

#[simd_test]
fn mask_bitmask_roundtrip_exhaustive<S: Simd>(simd: S) {
    for_each_exhaustive_bitmask(16, |bits| {
        check_bitmask_roundtrip!(simd, mask8x16, 16, bits);
        check_bitmask_roundtrip!(simd, mask16x16, 16, bits);
        check_bitmask_roundtrip!(simd, mask32x16, 16, bits);
    });

    for_each_exhaustive_bitmask(8, |bits| {
        check_bitmask_roundtrip!(simd, mask16x8, 8, bits);
        check_bitmask_roundtrip!(simd, mask32x8, 8, bits);
        check_bitmask_roundtrip!(simd, mask64x8, 8, bits);
    });

    for_each_exhaustive_bitmask(4, |bits| {
        check_bitmask_roundtrip!(simd, mask32x4, 4, bits);
        check_bitmask_roundtrip!(simd, mask64x4, 4, bits);
    });

    for_each_exhaustive_bitmask(2, |bits| {
        check_bitmask_roundtrip!(simd, mask64x2, 2, bits);
    });
}

#[simd_test]
fn mask_bitmask_roundtrip_wide_patterns<S: Simd>(simd: S) {
    for_each_wide_bitmask(32, |bits| {
        check_bitmask_roundtrip!(simd, mask8x32, 32, bits);
        check_bitmask_roundtrip!(simd, mask16x32, 32, bits);
    });

    for_each_wide_bitmask(64, |bits| {
        check_bitmask_roundtrip!(simd, mask8x64, 64, bits);
    });
}

#[simd_test]
fn mask_bitmask_roundtrip_test_set<S: Simd>(simd: S) {
    check_mask_methods!(simd, mask8x16, 16, 0x1_aa55_8001);
    check_mask_methods!(simd, mask16x8, 8, 0x1_a5);
    check_mask_methods!(simd, mask32x4, 4, 0x1d);
    check_mask_methods!(simd, mask64x2, 2, 0x6);

    check_mask_methods!(simd, mask8x32, 32, 0x1_8000_aa55);
    check_mask_methods!(simd, mask16x16, 16, 0x1_aa55_8001);
    check_mask_methods!(simd, mask32x8, 8, 0x1_a5);
    check_mask_methods!(simd, mask64x4, 4, 0x1d);

    check_mask_methods!(simd, mask8x64, 64, 0x8000_0001_5555_aaab);
    check_mask_methods!(simd, mask16x32, 32, 0x1_8000_aa55);
    check_mask_methods!(simd, mask32x16, 16, 0x1_aa55_8001);
    check_mask_methods!(simd, mask64x8, 8, 0x1_a5);
}
