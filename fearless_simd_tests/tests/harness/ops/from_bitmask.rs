// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn from_bitmask_mask64x2<S: Simd>(simd: S) {
    let a = simd.from_bitmask_mask64x2(3);
    assert_eq!(<[i64; 2]>::from(a), [-1_i64, -1_i64]);
}

#[simd_test]
fn from_bitmask_mask64x4<S: Simd>(simd: S) {
    let a = simd.from_bitmask_mask64x4(9);
    assert_eq!(<[i64; 4]>::from(a), [-1_i64, 0_i64, 0_i64, -1_i64]);
}

#[simd_test]
fn from_bitmask_mask64x8<S: Simd>(simd: S) {
    let a = simd.from_bitmask_mask64x8(129);
    assert_eq!(
        <[i64; 8]>::from(a),
        [-1_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, -1_i64]
    );
}
