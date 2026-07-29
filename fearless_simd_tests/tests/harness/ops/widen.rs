// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn widen_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    assert_eq!(
        *simd.widen_u8x16(a),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    );
}

#[simd_test]
fn widen_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    assert_eq!(
        *simd.widen_u8x32(a),
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
        ]
    );
}

#[simd_test]
fn widen_u8x16_all_values<S: Simd>(simd: S) {
    for base in (0_u16..=240).step_by(16) {
        let values: [u8; 16] = core::array::from_fn(|i| (base + i as u16) as u8);
        let expected = values.map(u16::from);
        let result = simd.widen_u8x16(u8x16::from_slice(simd, &values));

        assert_eq!(result.as_slice(), expected.as_slice(), "input base {base}",);
    }
}

#[simd_test]
fn widen_u8x32_all_values<S: Simd>(simd: S) {
    for base in (0_u16..=224).step_by(32) {
        let values: [u8; 32] = core::array::from_fn(|i| (base + i as u16) as u8);
        let expected = values.map(u16::from);
        let result = simd.widen_u8x32(u8x32::from_slice(simd, &values));

        assert_eq!(result.as_slice(), expected.as_slice(), "input base {base}",);
    }
}
