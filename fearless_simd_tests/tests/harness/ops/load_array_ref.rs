// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn load_array_ref_i64x2<S: Simd>(simd: S) {
    let values = [1_i64, -2_i64];
    let a = simd.load_array_ref_i64x2(&values);
    assert_eq!(*a, values);
}

#[simd_test]
fn load_array_ref_i64x4<S: Simd>(simd: S) {
    let values = [1_i64, -2_i64, 3_i64, -4_i64];
    let a = simd.load_array_ref_i64x4(&values);
    assert_eq!(*a, values);
}

#[simd_test]
fn load_array_ref_i64x8<S: Simd>(simd: S) {
    let values = [1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64];
    let a = simd.load_array_ref_i64x8(&values);
    assert_eq!(*a, values);
}

#[simd_test]
fn load_array_ref_u64x2<S: Simd>(simd: S) {
    let values = [1_u64, 2_u64];
    let a = simd.load_array_ref_u64x2(&values);
    assert_eq!(*a, values);
}

#[simd_test]
fn load_array_ref_u64x4<S: Simd>(simd: S) {
    let values = [1_u64, 2_u64, 3_u64, 4_u64];
    let a = simd.load_array_ref_u64x4(&values);
    assert_eq!(*a, values);
}

#[simd_test]
fn load_array_ref_u64x8<S: Simd>(simd: S) {
    let values = [1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64];
    let a = simd.load_array_ref_u64x8(&values);
    assert_eq!(*a, values);
}
