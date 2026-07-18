// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn store_slice_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let mut dest = [0.0_f32; 4];
    a.store_slice(&mut dest);
    assert_eq!(dest, [1.0, 2.0, 3.0, 4.0]);
}

#[simd_test]
fn store_slice_mask32x4<S: Simd>(simd: S) {
    let mask = mask32x4::from_slice(simd, &[-1, 0, -1, 0]);
    let mut dest = [0_i32; 4];
    mask.store_slice(&mut dest);
    assert_eq!(dest, [-1, 0, -1, 0]);
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn store_slice_i64x2<S: Simd>(simd: S) {
    let a = i64x2::from_slice(simd, &[1_i64, -2_i64]);
    let mut out = [0_i64; 2];
    a.store_slice(&mut out);
    assert_eq!(out, [1_i64, -2_i64]);
}

#[simd_test]
fn store_slice_i64x4<S: Simd>(simd: S) {
    let a = i64x4::from_slice(simd, &[1_i64, -2_i64, 3_i64, -4_i64]);
    let mut out = [0_i64; 4];
    a.store_slice(&mut out);
    assert_eq!(out, [1_i64, -2_i64, 3_i64, -4_i64]);
}

#[simd_test]
fn store_slice_i64x8<S: Simd>(simd: S) {
    let a = i64x8::from_slice(
        simd,
        &[1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64],
    );
    let mut out = [0_i64; 8];
    a.store_slice(&mut out);
    assert_eq!(
        out,
        [1_i64, -2_i64, 3_i64, -4_i64, 5_i64, -6_i64, 7_i64, -8_i64]
    );
}

#[simd_test]
fn store_slice_u64x2<S: Simd>(simd: S) {
    let a = u64x2::from_slice(simd, &[1_u64, 2_u64]);
    let mut out = [0_u64; 2];
    a.store_slice(&mut out);
    assert_eq!(out, [1_u64, 2_u64]);
}

#[simd_test]
fn store_slice_u64x4<S: Simd>(simd: S) {
    let a = u64x4::from_slice(simd, &[1_u64, 2_u64, 3_u64, 4_u64]);
    let mut out = [0_u64; 4];
    a.store_slice(&mut out);
    assert_eq!(out, [1_u64, 2_u64, 3_u64, 4_u64]);
}

#[simd_test]
fn store_slice_u64x8<S: Simd>(simd: S) {
    let a = u64x8::from_slice(
        simd,
        &[1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64],
    );
    let mut out = [0_u64; 8];
    a.store_slice(&mut out);
    assert_eq!(
        out,
        [1_u64, 2_u64, 3_u64, 4_u64, 5_u64, 6_u64, 7_u64, 8_u64]
    );
}
