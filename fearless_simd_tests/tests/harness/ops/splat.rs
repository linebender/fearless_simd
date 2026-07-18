// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn splat_f32x4<S: Simd>(simd: S) {
    let a = f32x4::splat(simd, 4.2);
    assert_eq!(*a, [4.2, 4.2, 4.2, 4.2]);
}

#[simd_test]
fn mask_trait_splat_mask32x4<S: Simd>(simd: S) {
    let t = mask32x4::splat(simd, true);
    assert_eq!(<[i32; 4]>::from(t), [-1; 4]);

    let f = mask32x4::splat(simd, false);
    assert_eq!(<[i32; 4]>::from(f), [0; 4]);
}

#[simd_test]
fn splat_native_mask<S: Simd>(simd: S) {
    let all_true = S::mask32s::splat(simd, true);
    assert!(all_true.all_true());

    let all_false = S::mask32s::splat(simd, false);
    assert!(all_false.all_false());
}

#[simd_test]
fn splat_f32x8<S: Simd>(simd: S) {
    let a = f32x8::splat(simd, 4.2);
    assert_eq!(*a, [4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2]);
}

#[simd_test]
fn splat_f32x16<S: Simd>(simd: S) {
    let a = f32x16::splat(simd, 4.2);
    assert_eq!(
        *a,
        [
            4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2
        ]
    );
}

#[simd_test]
fn splat_f64x8<S: Simd>(simd: S) {
    let a = f64x8::splat(simd, 4.2);
    assert_eq!(*a, [4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2]);
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn splat_i64x2<S: Simd>(simd: S) {
    let a = i64x2::splat(simd, 1_i64);
    assert_eq!(*a, [1_i64, 1_i64]);
}

#[simd_test]
fn splat_i64x4<S: Simd>(simd: S) {
    let a = i64x4::splat(simd, 1_i64);
    assert_eq!(*a, [1_i64, 1_i64, 1_i64, 1_i64]);
}

#[simd_test]
fn splat_i64x8<S: Simd>(simd: S) {
    let a = i64x8::splat(simd, 1_i64);
    assert_eq!(*a, [1_i64, 1_i64, 1_i64, 1_i64, 1_i64, 1_i64, 1_i64, 1_i64]);
}

#[simd_test]
fn splat_u64x2<S: Simd>(simd: S) {
    let a = u64x2::splat(simd, 1_u64);
    assert_eq!(*a, [1_u64, 1_u64]);
}

#[simd_test]
fn splat_u64x4<S: Simd>(simd: S) {
    let a = u64x4::splat(simd, 1_u64);
    assert_eq!(*a, [1_u64, 1_u64, 1_u64, 1_u64]);
}

#[simd_test]
fn splat_u64x8<S: Simd>(simd: S) {
    let a = u64x8::splat(simd, 1_u64);
    assert_eq!(*a, [1_u64, 1_u64, 1_u64, 1_u64, 1_u64, 1_u64, 1_u64, 1_u64]);
}

#[simd_test]
fn splat_mask64x2<S: Simd>(simd: S) {
    let a = simd.splat_mask64x2(true);
    assert_eq!(<[i64; 2]>::from(a), [-1; 2]);
}

#[simd_test]
fn splat_mask64x4<S: Simd>(simd: S) {
    let a = simd.splat_mask64x4(true);
    assert_eq!(<[i64; 4]>::from(a), [-1; 4]);
}

#[simd_test]
fn splat_mask64x8<S: Simd>(simd: S) {
    let a = simd.splat_mask64x8(true);
    assert_eq!(<[i64; 8]>::from(a), [-1; 8]);
}
