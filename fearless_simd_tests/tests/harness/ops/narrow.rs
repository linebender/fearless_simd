// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn narrow_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[
            0, 1, 127, 128, 255, 256, 300, 1000, 128, 192, 224, 240, 248, 252, 254, 65535,
        ],
    );
    assert_eq!(
        *simd.narrow_u16x16(a),
        [
            0, 1, 127, 128, 255, 0, 44, 232, 128, 192, 224, 240, 248, 252, 254, 255
        ]
    );
}

#[simd_test]
fn narrow_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            0, 1, 127, 128, 255, 256, 300, 1000, 128, 192, 224, 240, 248, 252, 254, 255, 100, 200,
            255, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65535, 0, 1, 2, 3,
        ],
    );
    assert_eq!(
        *simd.narrow_u16x32(a),
        [
            0, 1, 127, 128, 255, 0, 44, 232, 128, 192, 224, 240, 248, 252, 254, 255, 100, 200, 255,
            0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 1, 2, 3
        ]
    );
}
