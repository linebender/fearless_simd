// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn compress_all_widths<S: Simd>(simd: S) {
    macro_rules! check_width {
        ($bytes:ident, $mask:ident, $lanes:literal, $compress:ident, $compress_merge:ident) => {{
            let values: [u8; $lanes] = core::array::from_fn(|lane| (lane * 3 + 1) as u8);
            let merge: [u8; $lanes] = core::array::from_fn(|lane| 0xe0_u8.wrapping_add(lane as u8));
            let patterned_mask: u64 = (0..$lanes)
                .filter(|lane| lane % 3 == 0 || lane % 7 == 2)
                .fold(0, |mask, lane| mask | (1_u64 << lane));
            let values_vec = $bytes::simd_from(simd, values);
            let merge_vec = $bytes::simd_from(simd, merge);
            let all_lanes = u64::MAX >> (64 - $lanes);

            let repeated_byte_masks = (0_u64..=255).map(|byte_mask| {
                (0..($lanes / 8)).fold(0, |mask, block| mask | (byte_mask << (block * 8)))
            });
            let stitch_masks = ($lanes == 64).then(|| {
                (0..=32).map(|low_count| {
                    let low = if low_count == 32 {
                        u64::from(u32::MAX)
                    } else {
                        (1_u64 << low_count) - 1
                    };
                    low | (0xa5a5_a5a5_u64 << 32)
                })
            });
            for mask_bits in [0, 1, patterned_mask, all_lanes]
                .into_iter()
                .chain(repeated_byte_masks)
                .chain(stitch_masks.into_iter().flatten())
            {
                let mask = $mask::from_bitmask(simd, mask_bits);
                let mut expected = [0; $lanes];
                let mut expected_merge = merge;
                let mut output_lane = 0;
                for (input_lane, value) in values.into_iter().enumerate() {
                    if mask_bits & (1_u64 << input_lane) != 0 {
                        expected[output_lane] = value;
                        expected_merge[output_lane] = value;
                        output_lane += 1;
                    }
                }

                assert_eq!(*simd.$compress(values_vec, mask), expected);
                assert_eq!(
                    *simd.$compress_merge(values_vec, mask, merge_vec),
                    expected_merge
                );
            }
        }};
    }

    check_width!(u8x16, mask8x16, 16, compress_u8x16, compress_merge_u8x16);
    check_width!(u8x32, mask8x32, 32, compress_u8x32, compress_merge_u8x32);
    check_width!(u8x64, mask8x64, 64, compress_u8x64, compress_merge_u8x64);
}
