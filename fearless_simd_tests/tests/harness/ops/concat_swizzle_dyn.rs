// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn concat_swizzle_dyn_all_widths<S: Simd>(simd: S) {
    macro_rules! check_width {
        ($bytes:ident, $lanes:literal, $method:ident) => {{
            let low: [u8; $lanes] = core::array::from_fn(|lane| lane as u8);
            let high: [u8; $lanes] = core::array::from_fn(|lane| (lane + $lanes) as u8);
            let indices: [u8; $lanes] =
                core::array::from_fn(|lane| ((lane * 37 + 131) % (2 * $lanes)) as u8);
            let selected = simd.$method(
                $bytes::simd_from(simd, low),
                $bytes::simd_from(simd, high),
                $bytes::simd_from(simd, indices),
            );
            assert_eq!(*selected, indices);
        }};
    }

    check_width!(u8x16, 16, concat_swizzle_dyn_u8x16);
    check_width!(u8x32, 32, concat_swizzle_dyn_u8x32);
    check_width!(u8x64, 64, concat_swizzle_dyn_u8x64);
}
