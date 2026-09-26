// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

macro_rules! check_mask_narrow {
    ($name:ident, $wide:ident, $small:ident, $lanes:expr,
     $narrow:ident, $saturating:ident, $relaxed:ident, $widen:ident) => {
        #[simd_test]
        fn $name<S: Simd>(simd: S) {
            let lane_mask = (1_u64 << $lanes) - 1;
            let low_bits = 0x965a_c369_f0e1_d278_u64 & lane_mask;
            let high_bits = 0x39a7_5c81_42ef_b6d0_u64 & lane_mask;
            let low = $wide::from_bitmask(simd, low_bits);
            let high = $wide::from_bitmask(simd, high_bits);
            let expected = low_bits | (high_bits << $lanes);
            let results: [$small<S>; 3] = [
                simd.$narrow(low, high),
                simd.$saturating(low, high),
                simd.$relaxed(low, high),
            ];
            for result in results {
                assert_eq!(result.to_bitmask(), expected);
                let mut lanes = [0; $lanes * 2];
                result.store_slice(&mut lanes);
                for (index, &lane) in lanes.iter().enumerate() {
                    assert_eq!(lane, if (expected >> index) & 1 != 0 { -1 } else { 0 });
                }
                let (lo, hi) = simd.$widen(result);
                assert_eq!(lo.to_bitmask(), low_bits);
                assert_eq!(hi.to_bitmask(), high_bits);
            }
        }
    };
}

check_mask_narrow!(
    mask16x8_narrow,
    mask16x8,
    mask8x16,
    8,
    narrow_mask16x8,
    saturating_narrow_mask16x8,
    relaxed_narrow_mask16x8,
    widen_mask8x16
);
check_mask_narrow!(
    mask32x4_narrow,
    mask32x4,
    mask16x8,
    4,
    narrow_mask32x4,
    saturating_narrow_mask32x4,
    relaxed_narrow_mask32x4,
    widen_mask16x8
);
check_mask_narrow!(
    mask64x2_narrow,
    mask64x2,
    mask32x4,
    2,
    narrow_mask64x2,
    saturating_narrow_mask64x2,
    relaxed_narrow_mask64x2,
    widen_mask32x4
);
check_mask_narrow!(
    mask16x16_narrow,
    mask16x16,
    mask8x32,
    16,
    narrow_mask16x16,
    saturating_narrow_mask16x16,
    relaxed_narrow_mask16x16,
    widen_mask8x32
);
check_mask_narrow!(
    mask32x8_narrow,
    mask32x8,
    mask16x16,
    8,
    narrow_mask32x8,
    saturating_narrow_mask32x8,
    relaxed_narrow_mask32x8,
    widen_mask16x16
);
check_mask_narrow!(
    mask64x4_narrow,
    mask64x4,
    mask32x8,
    4,
    narrow_mask64x4,
    saturating_narrow_mask64x4,
    relaxed_narrow_mask64x4,
    widen_mask32x8
);
check_mask_narrow!(
    mask16x32_narrow,
    mask16x32,
    mask8x64,
    32,
    narrow_mask16x32,
    saturating_narrow_mask16x32,
    relaxed_narrow_mask16x32,
    widen_mask8x64
);
check_mask_narrow!(
    mask32x16_narrow,
    mask32x16,
    mask16x32,
    16,
    narrow_mask32x16,
    saturating_narrow_mask32x16,
    relaxed_narrow_mask32x16,
    widen_mask16x32
);
check_mask_narrow!(
    mask64x8_narrow,
    mask64x8,
    mask32x16,
    8,
    narrow_mask64x8,
    saturating_narrow_mask64x8,
    relaxed_narrow_mask64x8,
    widen_mask32x16
);
