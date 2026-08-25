// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn multishift_all_widths<S: Simd>(simd: S) {
    macro_rules! check_width {
        ($bytes:ident, $words:ident, $lanes:literal, $method:ident) => {{
            let data_lanes: [u64; $lanes / 8] = core::array::from_fn(|lane| {
                0x0123_4567_89ab_cdef_u64.rotate_left((lane * 11) as u32)
            });
            let bit_offsets: [u8; $lanes] =
                core::array::from_fn(|lane| (lane * 13 + (lane % 3) * 64) as u8);
            let shifted = simd.$method(
                $words::simd_from(simd, data_lanes),
                $bytes::simd_from(simd, bit_offsets),
            );
            let expected: [u8; $lanes] = core::array::from_fn(|lane| {
                data_lanes[lane / 8].rotate_right(u32::from(bit_offsets[lane] & 63)) as u8
            });
            assert_eq!(*shifted, expected);
        }};
    }

    check_width!(u8x16, u64x2, 16, multishift_u8x16);
    check_width!(u8x32, u64x4, 32, multishift_u8x32);
    check_width!(u8x64, u64x8, 64, multishift_u8x64);
}
