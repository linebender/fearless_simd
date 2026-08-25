// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn load_expand_all_widths<S: Simd>(simd: S) {
    macro_rules! check_width {
        ($bytes:ident, $mask:ident, $lanes:literal, $load_expand:ident, $load_expand_merge:ident) => {{
            let merge: [u8; $lanes] = core::array::from_fn(|lane| 0xe0_u8.wrapping_add(lane as u8));
            let patterned_mask: u64 = (0..$lanes)
                .filter(|lane| lane % 3 == 0 || lane % 7 == 2)
                .fold(0, |mask, lane| mask | (1_u64 << lane));
            let merge_vec = $bytes::simd_from(simd, merge);
            let all_lanes = u64::MAX >> (64 - $lanes);

            for mask_bits in [0, 1, patterned_mask, all_lanes] {
                let mask = $mask::from_bitmask(simd, mask_bits);
                let mut source = [0; $lanes];
                for (index, byte) in source
                    .iter_mut()
                    .take(mask_bits.count_ones() as usize)
                    .enumerate()
                {
                    *byte = 0x80_u8.wrapping_add(index as u8);
                }

                let mut expected = [0; $lanes];
                let mut expected_merge = merge;
                let mut source_index = 0;
                for lane in 0..$lanes {
                    if mask_bits & (1_u64 << lane) != 0 {
                        expected[lane] = source[source_index];
                        expected_merge[lane] = source[source_index];
                        source_index += 1;
                    }
                }

                assert_eq!(*simd.$load_expand(&source, mask), expected);
                assert_eq!(
                    *simd.$load_expand_merge(&source, mask, merge_vec),
                    expected_merge
                );
            }
        }};
    }

    check_width!(
        u8x16,
        mask8x16,
        16,
        load_expand_u8x16,
        load_expand_merge_u8x16
    );
    check_width!(
        u8x32,
        mask8x32,
        32,
        load_expand_u8x32,
        load_expand_merge_u8x32
    );
    check_width!(
        u8x64,
        mask8x64,
        64,
        load_expand_u8x64,
        load_expand_merge_u8x64
    );
}
