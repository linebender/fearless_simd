// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn to_bitmask_mask64x2<S: Simd>(simd: S) {
    let a = mask64x2::from_slice(simd, &[-1_i64, -1_i64]);
    assert_eq!(simd.to_bitmask_mask64x2(a), 3);
}

#[simd_test]
fn to_bitmask_mask64x4<S: Simd>(simd: S) {
    let a = mask64x4::from_slice(simd, &[-1_i64, 0_i64, 0_i64, -1_i64]);
    assert_eq!(simd.to_bitmask_mask64x4(a), 9);
}

#[simd_test]
fn to_bitmask_mask64x8<S: Simd>(simd: S) {
    let a = mask64x8::from_slice(
        simd,
        &[-1_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, 0_i64, -1_i64],
    );
    assert_eq!(simd.to_bitmask_mask64x8(a), 129);
}
