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
