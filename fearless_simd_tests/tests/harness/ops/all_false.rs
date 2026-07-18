// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn all_false_mask8x16<S: Simd>(simd: S) {
    let all_zero = mask8x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(simd.all_false_mask8x16(all_zero));

    let all_neg = mask8x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.all_false_mask8x16(all_neg));

    let one_neg = mask8x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]);
    assert!(!simd.all_false_mask8x16(one_neg));
}

#[simd_test]
fn all_false_mask16x8<S: Simd>(simd: S) {
    let all_zero = mask16x8::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(simd.all_false_mask16x8(all_zero));

    let all_neg = mask16x8::from_slice(simd, &[-1, -1, -1, -1, -1, -1, -1, -1]);
    assert!(!simd.all_false_mask16x8(all_neg));

    let one_neg = mask16x8::from_slice(simd, &[0, 0, 0, -1, 0, 0, 0, 0]);
    assert!(!simd.all_false_mask16x8(one_neg));
}

#[simd_test]
fn all_false_mask32x4<S: Simd>(simd: S) {
    let all_zero = mask32x4::from_slice(simd, &[0, 0, 0, 0]);
    assert!(simd.all_false_mask32x4(all_zero));

    let all_neg = mask32x4::from_slice(simd, &[-1, -1, -1, -1]);
    assert!(!simd.all_false_mask32x4(all_neg));

    let one_neg = mask32x4::from_slice(simd, &[0, -1, 0, 0]);
    assert!(!simd.all_false_mask32x4(one_neg));
}

#[simd_test]
fn all_false_mask64x2<S: Simd>(simd: S) {
    let all_zero = mask64x2::from_slice(simd, &[0, 0]);
    assert!(simd.all_false_mask64x2(all_zero));

    let all_neg = mask64x2::from_slice(simd, &[-1, -1]);
    assert!(!simd.all_false_mask64x2(all_neg));

    let one_neg = mask64x2::from_slice(simd, &[-1, 0]);
    assert!(!simd.all_false_mask64x2(one_neg));
}

#[simd_test]
fn all_false_mask8x64<S: Simd>(simd: S) {
    let all_zero = mask8x64::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ],
    );
    assert!(simd.all_false_mask8x64(all_zero));

    let all_neg = mask8x64::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.all_false_mask8x64(all_neg));

    let one_neg = mask8x64::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ],
    );
    assert!(!simd.all_false_mask8x64(one_neg));
}

#[simd_test]
fn all_false_mask16x32<S: Simd>(simd: S) {
    let all_zero = mask16x32::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
    );
    assert!(simd.all_false_mask16x32(all_zero));

    let all_neg = mask16x32::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.all_false_mask16x32(all_neg));

    let one_neg = mask16x32::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
    );
    assert!(!simd.all_false_mask16x32(one_neg));
}

#[simd_test]
fn all_false_mask32x16<S: Simd>(simd: S) {
    let all_zero = mask32x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(simd.all_false_mask32x16(all_zero));

    let all_neg = mask32x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.all_false_mask32x16(all_neg));

    let one_neg = mask32x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]);
    assert!(!simd.all_false_mask32x16(one_neg));
}

// These rows were split out of pre-existing bundled tests; they do not add new vector/type coverage.

#[simd_test]
fn all_false_mask64x4<S: Simd>(simd: S) {
    assert!(simd.all_false_mask64x4(simd.splat_mask64x4(false)));
    assert!(!simd.all_false_mask64x4(simd.from_bitmask_mask64x4(1)));
}

#[simd_test]
fn all_false_mask64x8<S: Simd>(simd: S) {
    assert!(simd.all_false_mask64x8(simd.splat_mask64x8(false)));
    assert!(!simd.all_false_mask64x8(simd.from_bitmask_mask64x8(1)));
}
