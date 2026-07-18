// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn all_true_mask8x16<S: Simd>(simd: S) {
    let all_zero = mask8x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(!simd.all_true_mask8x16(all_zero));

    let all_neg = mask8x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.all_true_mask8x16(all_neg));

    let one_pos = mask8x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.all_true_mask8x16(one_pos));
}

#[simd_test]
fn all_true_mask16x8<S: Simd>(simd: S) {
    let all_zero = mask16x8::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(!simd.all_true_mask16x8(all_zero));

    let all_neg = mask16x8::from_slice(simd, &[-1, -1, -1, -1, -1, -1, -1, -1]);
    assert!(simd.all_true_mask16x8(all_neg));

    let one_pos = mask16x8::from_slice(simd, &[-1, -1, -1, 0, -1, -1, -1, -1]);
    assert!(!simd.all_true_mask16x8(one_pos));
}

#[simd_test]
fn all_true_mask32x4<S: Simd>(simd: S) {
    let all_zero = mask32x4::from_slice(simd, &[0, 0, 0, 0]);
    assert!(!simd.all_true_mask32x4(all_zero));

    let all_neg = mask32x4::from_slice(simd, &[-1, -1, -1, -1]);
    assert!(simd.all_true_mask32x4(all_neg));

    let one_pos = mask32x4::from_slice(simd, &[-1, 0, -1, -1]);
    assert!(!simd.all_true_mask32x4(one_pos));
}

#[simd_test]
fn all_true_mask64x2<S: Simd>(simd: S) {
    let all_zero = mask64x2::from_slice(simd, &[0, 0]);
    assert!(!simd.all_true_mask64x2(all_zero));

    let all_neg = mask64x2::from_slice(simd, &[-1, -1]);
    assert!(simd.all_true_mask64x2(all_neg));

    let one_pos = mask64x2::from_slice(simd, &[-1, 0]);
    assert!(!simd.all_true_mask64x2(one_pos));
}

#[simd_test]
fn all_true_mask8x64<S: Simd>(simd: S) {
    let all_zero = mask8x64::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ],
    );
    assert!(!simd.all_true_mask8x64(all_zero));

    let all_neg = mask8x64::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.all_true_mask8x64(all_neg));

    let one_pos = mask8x64::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.all_true_mask8x64(one_pos));
}

#[simd_test]
fn all_true_mask16x32<S: Simd>(simd: S) {
    let all_zero = mask16x32::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
    );
    assert!(!simd.all_true_mask16x32(all_zero));

    let all_neg = mask16x32::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.all_true_mask16x32(all_neg));

    let one_pos = mask16x32::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.all_true_mask16x32(one_pos));
}

#[simd_test]
fn all_true_mask32x16<S: Simd>(simd: S) {
    let all_zero = mask32x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(!simd.all_true_mask32x16(all_zero));

    let all_neg = mask32x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.all_true_mask32x16(all_neg));

    let one_pos = mask32x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.all_true_mask32x16(one_pos));
}

// Additional concrete rows for this operation.

#[simd_test]
fn all_true_mask64x4<S: Simd>(simd: S) {
    assert!(simd.all_true_mask64x4(simd.splat_mask64x4(true)));
    assert!(!simd.all_true_mask64x4(simd.from_bitmask_mask64x4(1)));
}

#[simd_test]
fn all_true_mask64x8<S: Simd>(simd: S) {
    assert!(simd.all_true_mask64x8(simd.splat_mask64x8(true)));
    assert!(!simd.all_true_mask64x8(simd.from_bitmask_mask64x8(1)));
}

// Generated gap-fill coverage rows.

#[simd_test]
fn all_true_mask8x32<S: Simd>(simd: S) {
    let values: [i8; 32] = core::array::from_fn(|_| -1_i8);
    let mask = mask8x32::from_slice(simd, &values);
    assert!(simd.all_true_mask8x32(mask));
}

#[simd_test]
fn all_true_mask16x16<S: Simd>(simd: S) {
    let values: [i16; 16] = core::array::from_fn(|_| -1_i16);
    let mask = mask16x16::from_slice(simd, &values);
    assert!(simd.all_true_mask16x16(mask));
}

#[simd_test]
fn all_true_mask32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = core::array::from_fn(|_| -1_i32);
    let mask = mask32x8::from_slice(simd, &values);
    assert!(simd.all_true_mask32x8(mask));
}
