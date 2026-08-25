// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn expand_all_widths<S: Simd>(simd: S) {
    macro_rules! check_width {
        ($bytes:ident, $mask:ident, $lanes:literal, $expand:ident, $expand_merge:ident) => {{
            let values: [u8; $lanes] = core::array::from_fn(|lane| (lane * 3 + 1) as u8);
            let merge: [u8; $lanes] = core::array::from_fn(|lane| 0xe0_u8.wrapping_add(lane as u8));
            let patterned_mask: u64 = (0..$lanes)
                .filter(|lane| lane % 3 == 0 || lane % 7 == 2)
                .fold(0, |mask, lane| mask | (1_u64 << lane));
            let values_vec = $bytes::simd_from(simd, values);
            let merge_vec = $bytes::simd_from(simd, merge);
            let all_lanes = u64::MAX >> (64 - $lanes);

            for mask_bits in [0, 1, patterned_mask, all_lanes] {
                let mask = $mask::from_bitmask(simd, mask_bits);
                let mut expected = [0; $lanes];
                let mut expected_merge = merge;
                let mut input_lane = 0;
                for lane in 0..$lanes {
                    if mask_bits & (1_u64 << lane) != 0 {
                        expected[lane] = values[input_lane];
                        expected_merge[lane] = values[input_lane];
                        input_lane += 1;
                    }
                }

                assert_eq!(*simd.$expand(values_vec, mask), expected);
                assert_eq!(
                    *simd.$expand_merge(values_vec, mask, merge_vec),
                    expected_merge
                );
            }
        }};
    }

    check_width!(u8x16, mask8x16, 16, expand_u8x16, expand_merge_u8x16);
    check_width!(u8x32, mask8x32, 32, expand_u8x32, expand_merge_u8x32);
    check_width!(u8x64, mask8x64, 64, expand_u8x64, expand_merge_u8x64);
}
