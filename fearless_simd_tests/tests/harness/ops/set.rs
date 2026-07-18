// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn set_mask64x2<S: Simd>(simd: S) {
    let mut a = simd.splat_mask64x2(false);
    simd.set_mask64x2(&mut a, 0, true);
    simd.set_mask64x2(&mut a, 1, true);
    assert_eq!(simd.to_bitmask_mask64x2(a), 3);
}

#[simd_test]
fn set_mask64x4<S: Simd>(simd: S) {
    let mut a = simd.splat_mask64x4(false);
    simd.set_mask64x4(&mut a, 0, true);
    simd.set_mask64x4(&mut a, 3, true);
    assert_eq!(simd.to_bitmask_mask64x4(a), 9);
}

#[simd_test]
fn set_mask64x8<S: Simd>(simd: S) {
    let mut a = simd.splat_mask64x8(false);
    simd.set_mask64x8(&mut a, 0, true);
    simd.set_mask64x8(&mut a, 7, true);
    assert_eq!(simd.to_bitmask_mask64x8(a), 129);
}
