// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Extended test suite for 512-bit vector operations
//
// This file contains tests for operations that were not covered in mod_512.rs,
// including mask operations (all/any/true/false), split/combine, zip/unzip,
// shift operations, widen/narrow, and various conversion functions.

#![expect(
    clippy::cast_possible_wrap,
    reason = "from_fn() tests use `usize as i8` and such which will never overflow for 64-long vectors"
)]

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// =============================================================================
// Mask operations tests (512-bit)
// =============================================================================

#[simd_test]
fn any_true_mask8x64<S: Simd>(simd: S) {
    let all_zero = mask8x64::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ],
    );
    assert!(!simd.any_true_mask8x64(all_zero));

    let all_neg = mask8x64::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.any_true_mask8x64(all_neg));

    let one_neg = mask8x64::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ],
    );
    assert!(simd.any_true_mask8x64(one_neg));
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
fn any_false_mask8x64<S: Simd>(simd: S) {
    let all_zero = mask8x64::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ],
    );
    assert!(simd.any_false_mask8x64(all_zero));

    let all_neg = mask8x64::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.any_false_mask8x64(all_neg));

    let one_pos = mask8x64::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.any_false_mask8x64(one_pos));
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
fn any_true_mask16x32<S: Simd>(simd: S) {
    let all_zero = mask16x32::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
    );
    assert!(!simd.any_true_mask16x32(all_zero));

    let all_neg = mask16x32::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.any_true_mask16x32(all_neg));

    let one_neg = mask16x32::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
    );
    assert!(simd.any_true_mask16x32(one_neg));
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
fn any_false_mask16x32<S: Simd>(simd: S) {
    let all_zero = mask16x32::from_slice(
        simd,
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
    );
    assert!(simd.any_false_mask16x32(all_zero));

    let all_neg = mask16x32::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.any_false_mask16x32(all_neg));

    let one_pos = mask16x32::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.any_false_mask16x32(one_pos));
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
fn any_true_mask32x16<S: Simd>(simd: S) {
    let all_zero = mask32x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(!simd.any_true_mask32x16(all_zero));

    let all_neg = mask32x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.any_true_mask32x16(all_neg));

    let one_neg = mask32x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]);
    assert!(simd.any_true_mask32x16(one_neg));
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

#[simd_test]
fn any_false_mask32x16<S: Simd>(simd: S) {
    let all_zero = mask32x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(simd.any_false_mask32x16(all_zero));

    let all_neg = mask32x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.any_false_mask32x16(all_neg));

    let one_pos = mask32x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.any_false_mask32x16(one_pos));
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

// =============================================================================
// Split and combine tests (512-bit)
// =============================================================================

#[simd_test]
fn split_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let (lo, hi) = a.split();
    assert_eq!(*lo, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*hi, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
}

#[simd_test]
fn split_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ],
    );
    let (lo, hi) = a.split();
    assert_eq!(
        *lo,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        ]
    );
    assert_eq!(
        *hi,
        [
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64
        ]
    );
}

#[simd_test]
fn split_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    );
    let (lo, hi) = a.split();
    assert_eq!(
        *lo,
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
        ]
    );
    assert_eq!(
        *hi,
        [
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63
        ]
    );
}

#[simd_test]
fn split_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let (lo, hi) = a.split();
    assert_eq!(*lo, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(
        *hi,
        [
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ]
    );
}

#[simd_test]
fn split_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    let (lo, hi) = a.split();
    assert_eq!(*lo, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(
        *hi,
        [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ]
    );
}

#[simd_test]
fn split_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let (lo, hi) = a.split();
    assert_eq!(*lo, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*hi, [9, 10, 11, 12, 13, 14, 15, 16]);
}

#[simd_test]
fn split_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let (lo, hi) = a.split();
    assert_eq!(*lo, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(*hi, [8, 9, 10, 11, 12, 13, 14, 15]);
}

// =============================================================================
// Fract tests (512-bit floats)
// =============================================================================

#[simd_test]
fn fract_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.7, -2.3, 3.9, -4.1, 5.5, -6.6, 7.2, -8.8, 1.25, -2.75, 0.0, -0.5, 10.125, -10.875,
            100.0, -100.0,
        ],
    );
    let result = simd.fract_f32x16(a);
    assert_eq!(
        *result,
        [
            0.70000005,
            -0.29999995,
            0.9000001,
            -0.099999905,
            0.5,
            -0.5999999,
            0.19999981,
            -0.8000002,
            0.25,
            -0.75,
            0.0,
            -0.5,
            0.125,
            -0.875,
            0.0,
            0.0
        ]
    );
}

// =============================================================================
// max_precise and min_precise tests (512-bit floats)
// =============================================================================

#[simd_test]
fn max_precise_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0, 2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0, 1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0,
        ],
    );
    assert_eq!(
        *a.max_precise(b),
        [
            2.0, -2.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0, 2.0, -2.0, 7.0, 3.0, 2.0, 5.0, 6.0, 7.0
        ]
    );
}

#[simd_test]
fn min_precise_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0, 2.0, -3.0, 0.0, 0.5, 1.0, 5.0, 3.0, 7.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0, 1.0, -2.0, 7.0, 3.0, 2.0, 4.0, 6.0, 5.0,
        ],
    );
    assert_eq!(
        *a.min_precise(b),
        [
            1.0, -3.0, 0.0, 0.5, 1.0, 4.0, 3.0, 5.0, 1.0, -3.0, 0.0, 0.5, 1.0, 4.0, 3.0, 5.0
        ]
    );
}

#[simd_test]
fn max_precise_f32x16_with_nan<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            f32::NAN,
            -3.0,
            f32::INFINITY,
            0.5,
            1.0,
            f32::NAN,
            3.0,
            7.0,
            f32::NAN,
            -3.0,
            f32::INFINITY,
            0.5,
            1.0,
            f32::NAN,
            3.0,
            7.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0,
            f32::NAN,
            7.0,
            f32::NEG_INFINITY,
            f32::NAN,
            4.0,
            6.0,
            5.0,
            1.0,
            f32::NAN,
            7.0,
            f32::NEG_INFINITY,
            f32::NAN,
            4.0,
            6.0,
            5.0,
        ],
    );
    let result = a.max_precise(b);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -3.0);
    assert_eq!(result[2], f32::INFINITY);
    assert_eq!(result[3], 0.5);
    assert_eq!(result[4], 1.0);
    assert_eq!(result[5], 4.0);
    assert_eq!(result[6], 6.0);
    assert_eq!(result[7], 7.0);
    assert_eq!(result[8], 1.0);
    assert_eq!(result[9], -3.0);
    assert_eq!(result[10], f32::INFINITY);
    assert_eq!(result[11], 0.5);
    assert_eq!(result[12], 1.0);
    assert_eq!(result[13], 4.0);
    assert_eq!(result[14], 6.0);
    assert_eq!(result[15], 7.0);
}

#[simd_test]
fn min_precise_f32x16_with_nan<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            f32::NAN,
            -3.0,
            f32::INFINITY,
            0.5,
            1.0,
            f32::NAN,
            3.0,
            7.0,
            f32::NAN,
            -3.0,
            f32::INFINITY,
            0.5,
            1.0,
            f32::NAN,
            3.0,
            7.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            1.0,
            f32::NAN,
            7.0,
            f32::NEG_INFINITY,
            f32::NAN,
            4.0,
            6.0,
            5.0,
            1.0,
            f32::NAN,
            7.0,
            f32::NEG_INFINITY,
            f32::NAN,
            4.0,
            6.0,
            5.0,
        ],
    );
    let result = a.min_precise(b);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -3.0);
    assert_eq!(result[2], 7.0);
    assert_eq!(result[3], f32::NEG_INFINITY);
    assert_eq!(result[4], 1.0);
    assert_eq!(result[5], 4.0);
    assert_eq!(result[6], 3.0);
    assert_eq!(result[7], 5.0);
    assert_eq!(result[8], 1.0);
    assert_eq!(result[9], -3.0);
    assert_eq!(result[10], 7.0);
    assert_eq!(result[11], f32::NEG_INFINITY);
    assert_eq!(result[12], 1.0);
    assert_eq!(result[13], 4.0);
    assert_eq!(result[14], 3.0);
    assert_eq!(result[15], 5.0);
}

// =============================================================================
// Shift operations tests (512-bit)
// =============================================================================

#[simd_test]
fn shr_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            -128, -64, -32, -16, -8, -4, -2, -1, 127, 64, 32, 16, 8, 4, 2, 1, -128, -64, -32, -16,
            -8, -4, -2, -1, 127, 64, 32, 16, 8, 4, 2, 1, -128, -64, -32, -16, -8, -4, -2, -1, 127,
            64, 32, 16, 8, 4, 2, 1, -128, -64, -32, -16, -8, -4, -2, -1, 127, 64, 32, 16, 8, 4, 2,
            1,
        ],
    );
    assert_eq!(
        *(a >> 2),
        [
            -32, -16, -8, -4, -2, -1, -1, -1, 31, 16, 8, 4, 2, 1, 0, 0, -32, -16, -8, -4, -2, -1,
            -1, -1, 31, 16, 8, 4, 2, 1, 0, 0, -32, -16, -8, -4, -2, -1, -1, -1, 31, 16, 8, 4, 2, 1,
            0, 0, -32, -16, -8, -4, -2, -1, -1, -1, 31, 16, 8, 4, 2, 1, 0, 0
        ]
    );
}

#[simd_test]
fn shr_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            255, 128, 64, 32, 16, 8, 4, 2, 254, 127, 63, 31, 15, 7, 3, 1, 255, 128, 64, 32, 16, 8,
            4, 2, 254, 127, 63, 31, 15, 7, 3, 1, 255, 128, 64, 32, 16, 8, 4, 2, 254, 127, 63, 31,
            15, 7, 3, 1, 255, 128, 64, 32, 16, 8, 4, 2, 254, 127, 63, 31, 15, 7, 3, 1,
        ],
    );
    assert_eq!(
        *(a >> 2),
        [
            63, 32, 16, 8, 4, 2, 1, 0, 63, 31, 15, 7, 3, 1, 0, 0, 63, 32, 16, 8, 4, 2, 1, 0, 63,
            31, 15, 7, 3, 1, 0, 0, 63, 32, 16, 8, 4, 2, 1, 0, 63, 31, 15, 7, 3, 1, 0, 0, 63, 32,
            16, 8, 4, 2, 1, 0, 63, 31, 15, 7, 3, 1, 0, 0
        ]
    );
}

#[simd_test]
fn shl_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ],
    );
    assert_eq!(
        *(a << 2),
        [
            4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 4, 8, 12, 16, 20, 24, 28,
            32, 36, 40, 44, 48, 52, 56, 60, 64, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52,
            56, 60, 64, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64
        ]
    );
}

#[simd_test]
fn shl_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ],
    );
    assert_eq!(
        *(a << 2),
        [
            4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 4, 8, 12, 16, 20, 24, 28,
            32, 36, 40, 44, 48, 52, 56, 60, 64, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52,
            56, 60, 64, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64
        ]
    );
}

#[simd_test]
fn shr_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            -32768, -16384, -1024, -1, 32767, 16384, 1024, 1, -32768, -16384, -1024, -1, 32767,
            16384, 1024, 1, -32768, -16384, -1024, -1, 32767, 16384, 1024, 1, -32768, -16384,
            -1024, -1, 32767, 16384, 1024, 1,
        ],
    );
    assert_eq!(
        *(a >> 4),
        [
            -2048, -1024, -64, -1, 2047, 1024, 64, 0, -2048, -1024, -64, -1, 2047, 1024, 64, 0,
            -2048, -1024, -64, -1, 2047, 1024, 64, 0, -2048, -1024, -64, -1, 2047, 1024, 64, 0
        ]
    );
}

#[simd_test]
fn shr_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            65535, 32768, 16384, 8192, 4096, 2048, 1024, 512, 65535, 32768, 16384, 8192, 4096,
            2048, 1024, 512, 65535, 32768, 16384, 8192, 4096, 2048, 1024, 512, 65535, 32768, 16384,
            8192, 4096, 2048, 1024, 512,
        ],
    );
    assert_eq!(
        *(a >> 4),
        [
            4095, 2048, 1024, 512, 256, 128, 64, 32, 4095, 2048, 1024, 512, 256, 128, 64, 32, 4095,
            2048, 1024, 512, 256, 128, 64, 32, 4095, 2048, 1024, 512, 256, 128, 64, 32
        ]
    );
}

#[simd_test]
fn shl_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8,
        ],
    );
    assert_eq!(
        *(a << 4),
        [
            16, 32, 48, 64, 80, 96, 112, 128, 16, 32, 48, 64, 80, 96, 112, 128, 16, 32, 48, 64, 80,
            96, 112, 128, 16, 32, 48, 64, 80, 96, 112, 128
        ]
    );
}

#[simd_test]
fn shl_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8,
        ],
    );
    assert_eq!(
        *(a << 4),
        [
            16, 32, 48, 64, 80, 96, 112, 128, 16, 32, 48, 64, 80, 96, 112, 128, 16, 32, 48, 64, 80,
            96, 112, 128, 16, 32, 48, 64, 80, 96, 112, 128
        ]
    );
}

#[simd_test]
fn shr_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
        ],
    );
    assert_eq!(
        *(a >> 8),
        [
            -8388608, -256, 256, 8388607, -8388608, -256, 256, 8388607, -8388608, -256, 256,
            8388607, -8388608, -256, 256, 8388607
        ]
    );
}

#[simd_test]
fn shr_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            u32::MAX,
            2147483648,
            65536,
            256,
            u32::MAX,
            2147483648,
            65536,
            256,
            u32::MAX,
            2147483648,
            65536,
            256,
            u32::MAX,
            2147483648,
            65536,
            256,
        ],
    );
    assert_eq!(
        *(a >> 8),
        [
            16777215, 8388608, 256, 1, 16777215, 8388608, 256, 1, 16777215, 8388608, 256, 1,
            16777215, 8388608, 256, 1
        ]
    );
}

#[simd_test]
fn shl_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        *(a << 4),
        [
            16, 32, 48, 64, 80, 96, 112, 128, 16, 32, 48, 64, 80, 96, 112, 128
        ]
    );
}

#[simd_test]
fn shl_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            0xFFFFFFFF, 0xFFFF, 0xFF, 0, 0xFFFFFFFF, 0xFFFF, 0xFF, 0, 0xFFFFFFFF, 0xFFFF, 0xFF, 0,
            0xFFFFFFFF, 0xFFFF, 0xFF, 0,
        ],
    );
    assert_eq!(
        *(a << 4),
        [
            0xFFFFFFF0, 0xFFFF0, 0xFF0, 0, 0xFFFFFFF0, 0xFFFF0, 0xFF0, 0, 0xFFFFFFF0, 0xFFFF0,
            0xFF0, 0, 0xFFFFFFF0, 0xFFFF0, 0xFF0, 0
        ]
    );
}

// Vector shift tests (shlv/shrv)
#[simd_test]
fn shrv_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
            i32::MIN,
            -65536,
            65536,
            i32::MAX,
        ],
    );
    assert_eq!(
        *(a >> i32x16::splat(simd, 8)),
        [
            -8388608, -256, 256, 8388607, -8388608, -256, 256, 8388607, -8388608, -256, 256,
            8388607, -8388608, -256, 256, 8388607
        ]
    );
}

#[simd_test]
fn shrv_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            u32::MAX,
            2147483648,
            65536,
            256,
            u32::MAX,
            2147483648,
            65536,
            256,
            u32::MAX,
            2147483648,
            65536,
            256,
            u32::MAX,
            2147483648,
            65536,
            256,
        ],
    );
    assert_eq!(
        *(a >> u32x16::splat(simd, 8)),
        [
            16777215, 8388608, 256, 1, 16777215, 8388608, 256, 1, 16777215, 8388608, 256, 1,
            16777215, 8388608, 256, 1
        ]
    );
}

#[simd_test]
fn shlv_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[
            0xFFFFFFFF, 0xFFFF, 0xFF, 0, 0xFFFFFFFF, 0xFFFF, 0xFF, 0, 0xFFFFFFFF, 0xFFFF, 0xFF, 0,
            0xFFFFFFFF, 0xFFFF, 0xFF, 0,
        ],
    );
    assert_eq!(
        *(a << u32x16::splat(simd, 4)),
        [
            0xFFFFFFF0, 0xFFFF0, 0xFF0, 0, 0xFFFFFFF0, 0xFFFF0, 0xFF0, 0, 0xFFFFFFF0, 0xFFFF0,
            0xFF0, 0, 0xFFFFFFF0, 0xFFFF0, 0xFF0, 0
        ]
    );
}

#[simd_test]
fn shrv_u32x16_varied<S: Simd>(simd: S) {
    let a = u32x16::splat(simd, u32::MAX);
    const SHIFTS: [u32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    assert_eq!(
        *(a >> u32x16::from_slice(simd, &SHIFTS)),
        SHIFTS.map(|x| u32::MAX >> x)
    );
}

#[simd_test]
fn shlv_u32x16_varied<S: Simd>(simd: S) {
    let a = u32x16::splat(simd, u32::MAX);
    const SHIFTS: [u32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    assert_eq!(
        *(a << u32x16::from_slice(simd, &SHIFTS)),
        SHIFTS.map(|x| u32::MAX << x)
    );
}

// =============================================================================
// Zip and unzip tests (512-bit)
// =============================================================================

#[simd_test]
fn zip_low_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0,
        ],
    );
    // zip_low interleaves the first half of each 256-bit lane
    assert_eq!(
        *simd.zip_low_f32x16(a, b),
        [
            0.0, 16.0, 1.0, 17.0, 2.0, 18.0, 3.0, 19.0, 4.0, 20.0, 5.0, 21.0, 6.0, 22.0, 7.0, 23.0
        ]
    );
}

#[simd_test]
fn zip_high_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0,
        ],
    );
    // zip_high interleaves the second half of each 256-bit lane
    assert_eq!(
        *simd.zip_high_f32x16(a, b),
        [
            8.0, 24.0, 9.0, 25.0, 10.0, 26.0, 11.0, 27.0, 12.0, 28.0, 13.0, 29.0, 14.0, 30.0, 15.0,
            31.0
        ]
    );
}

#[simd_test]
fn unzip_low_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ],
    );
    // unzip_low takes all even-indexed elements
    assert_eq!(
        *simd.unzip_low_f32x16(a, b),
        [
            1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
            31.0
        ]
    );
}

#[simd_test]
fn unzip_high_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ],
    );
    // unzip_high takes all odd-indexed elements
    assert_eq!(
        *simd.unzip_high_f32x16(a, b),
        [
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
            32.0
        ]
    );
}

#[simd_test]
fn zip_low_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
            106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
            123, 124, 125, 126, 127,
        ],
    );
    // zip_low takes the first half of the 512-bit vectors and interleaves them
    assert_eq!(
        *simd.zip_low_i8x64(a, b),
        [
            0, 64, 1, 65, 2, 66, 3, 67, 4, 68, 5, 69, 6, 70, 7, 71, 8, 72, 9, 73, 10, 74, 11, 75,
            12, 76, 13, 77, 14, 78, 15, 79, 16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85, 22, 86,
            23, 87, 24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95
        ]
    );
}

#[simd_test]
fn zip_high_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    );
    let b = i8x64::from_slice(
        simd,
        &[
            64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
            106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
            123, 124, 125, 126, 127,
        ],
    );
    // zip_high takes the second half of the 512-bit vectors and interleaves them
    assert_eq!(
        *simd.zip_high_i8x64(a, b),
        [
            32, 96, 33, 97, 34, 98, 35, 99, 36, 100, 37, 101, 38, 102, 39, 103, 40, 104, 41, 105,
            42, 106, 43, 107, 44, 108, 45, 109, 46, 110, 47, 111, 48, 112, 49, 113, 50, 114, 51,
            115, 52, 116, 53, 117, 54, 118, 55, 119, 56, 120, 57, 121, 58, 122, 59, 123, 60, 124,
            61, 125, 62, 126, 63, 127
        ]
    );
}

#[simd_test]
fn zip_low_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    // zip_low takes the first half of each vector and interleaves them
    assert_eq!(
        *simd.zip_low_i32x16(a, b),
        [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
    );
}

#[simd_test]
fn zip_high_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    // zip_high takes the second half of each vector and interleaves them
    assert_eq!(
        *simd.zip_high_i32x16(a, b),
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    );
}

#[simd_test]
fn zip_low_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u32x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    // zip_low takes the first half of each vector and interleaves them
    assert_eq!(
        *simd.zip_low_u32x16(a, b),
        [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
    );
}

#[simd_test]
fn zip_high_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u32x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    // zip_high takes the second half of each vector and interleaves them
    assert_eq!(
        *simd.zip_high_u32x16(a, b),
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    );
}

#[simd_test]
fn unzip_low_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    // unzip_low takes even-indexed elements from each vector
    assert_eq!(
        *simd.unzip_low_i32x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_high_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i32x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    // unzip_high takes odd-indexed elements from each vector
    assert_eq!(
        *simd.unzip_high_i32x16(a, b),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    );
}

#[simd_test]
fn unzip_low_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u32x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    // unzip_low takes even-indexed elements from each vector
    assert_eq!(
        *simd.unzip_low_u32x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_high_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u32x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    // unzip_high takes odd-indexed elements from each vector
    assert_eq!(
        *simd.unzip_high_u32x16(a, b),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    );
}

// Note: widen/narrow operations are not available for 512-bit vectors (u8x64, u16x64).
// The widen_u8x32 -> u16x32 and narrow_u16x32 -> u8x32 operations exist for 256-bit vectors.

// =============================================================================
// from_fn tests (512-bit)
// =============================================================================

#[simd_test]
fn from_fn_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_fn(simd, |i| i as f32 * 2.0);
    assert_eq!(
        *a,
        [
            0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0,
            30.0
        ]
    );
}

#[simd_test]
fn from_fn_i8x64<S: Simd>(simd: S) {
    let a = i8x64::from_fn(simd, |i| i as i8);
    assert_eq!(
        *a,
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
        ]
    );
}

#[simd_test]
fn from_fn_u8x64<S: Simd>(simd: S) {
    let a = u8x64::from_fn(simd, |i| (i * 2) as u8);
    assert_eq!(
        *a,
        [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44,
            46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
            90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124,
            126
        ]
    );
}

#[simd_test]
fn from_fn_i16x32<S: Simd>(simd: S) {
    let a = i16x32::from_fn(simd, |i| (i as i16) * 100);
    assert_eq!(
        *a,
        [
            0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
            1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
            3000, 3100
        ]
    );
}

#[simd_test]
fn from_fn_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_fn(simd, |i| i as u16 + 1000);
    assert_eq!(
        *a,
        [
            1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
            1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027,
            1028, 1029, 1030, 1031
        ]
    );
}

#[simd_test]
fn from_fn_i32x16<S: Simd>(simd: S) {
    let a = i32x16::from_fn(simd, |i| (i as i32) * (i as i32));
    assert_eq!(
        *a,
        [
            0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225
        ]
    );
}

#[simd_test]
fn from_fn_u32x16<S: Simd>(simd: S) {
    let a = u32x16::from_fn(simd, |i| 1_u32 << i);
    assert_eq!(
        *a,
        [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
        ]
    );
}

// =============================================================================
// block_splat tests (512-bit)
// =============================================================================

#[simd_test]
fn block_splat_f32x16<S: Simd>(simd: S) {
    let block = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let a = f32x16::block_splat(block);
    assert_eq!(
        *a,
        [
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0
        ]
    );
}

#[simd_test]
fn block_splat_i8x64<S: Simd>(simd: S) {
    let block = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let a = i8x64::block_splat(block);
    assert_eq!(
        *a,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ]
    );
}

#[simd_test]
fn block_splat_u8x64<S: Simd>(simd: S) {
    let block = u8x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    let a = u8x64::block_splat(block);
    assert_eq!(
        *a,
        [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40,
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40, 50, 60, 70, 80,
            90, 100, 110, 120, 130, 140, 150, 160, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
            120, 130, 140, 150, 160
        ]
    );
}

#[simd_test]
fn block_splat_i16x32<S: Simd>(simd: S) {
    let block = i16x8::from_slice(simd, &[100, 200, 300, 400, 500, 600, 700, 800]);
    let a = i16x32::block_splat(block);
    assert_eq!(
        *a,
        [
            100, 200, 300, 400, 500, 600, 700, 800, 100, 200, 300, 400, 500, 600, 700, 800, 100,
            200, 300, 400, 500, 600, 700, 800, 100, 200, 300, 400, 500, 600, 700, 800
        ]
    );
}

#[simd_test]
fn block_splat_u16x32<S: Simd>(simd: S) {
    let block = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let a = u16x32::block_splat(block);
    assert_eq!(
        *a,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8
        ]
    );
}

#[simd_test]
fn block_splat_i32x16<S: Simd>(simd: S) {
    let block = i32x4::from_slice(simd, &[11, 22, 33, 44]);
    let a = i32x16::block_splat(block);
    assert_eq!(
        *a,
        [
            11, 22, 33, 44, 11, 22, 33, 44, 11, 22, 33, 44, 11, 22, 33, 44
        ]
    );
}

#[simd_test]
fn block_splat_u32x16<S: Simd>(simd: S) {
    let block = u32x4::from_slice(simd, &[0xDEAD, 0xBEEF, 0xCAFE, 0xBABE]);
    let a = u32x16::block_splat(block);
    assert_eq!(
        *a,
        [
            0xDEAD, 0xBEEF, 0xCAFE, 0xBABE, 0xDEAD, 0xBEEF, 0xCAFE, 0xBABE, 0xDEAD, 0xBEEF, 0xCAFE,
            0xBABE, 0xDEAD, 0xBEEF, 0xCAFE, 0xBABE
        ]
    );
}

// =============================================================================
// Conversion tests (512-bit)
// =============================================================================

#[simd_test]
fn cvt_i32_f32x16<S: Simd>(simd: S) {
    use crate::harness::SimdCvtTruncate;
    let a = f32x16::from_slice(
        simd,
        &[
            1.7, -2.3, 3.9, -4.1, 5.5, -6.6, 7.2, -8.8, 10.0, -11.5, 12.9, -13.1, 14.0, -15.0, 0.0,
            100.5,
        ],
    );
    let result = i32x16::truncate_from(a);
    assert_eq!(
        *result,
        [
            1, -2, 3, -4, 5, -6, 7, -8, 10, -11, 12, -13, 14, -15, 0, 100
        ]
    );
}

#[simd_test]
fn cvt_u32_f32x16<S: Simd>(simd: S) {
    use crate::harness::SimdCvtTruncate;
    let a = f32x16::from_slice(
        simd,
        &[
            1.7, 2.3, 3.9, 4.1, 5.5, 6.6, 7.2, 8.8, 10.0, 11.5, 12.9, 13.1, 14.0, 15.0, 0.0, 100.5,
        ],
    );
    let result = u32x16::truncate_from(a);
    assert_eq!(
        *result,
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 0, 100]
    );
}

#[simd_test]
fn cvt_f32_i32x16<S: Simd>(simd: S) {
    use crate::harness::SimdCvtFloat;
    let a = i32x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 10, -11, 12, -13, 14, -15, 0, 100,
        ],
    );
    let result = f32x16::float_from(a);
    assert_eq!(
        *result,
        [
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 0.0,
            100.0
        ]
    );
}

#[simd_test]
fn cvt_f32_u32x16<S: Simd>(simd: S) {
    use crate::harness::SimdCvtFloat;
    let a = u32x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 0, 100],
    );
    let result = f32x16::float_from(a);
    assert_eq!(
        *result,
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 100.0
        ]
    );
}

#[simd_test]
fn cvt_i32_precise_f32x16<S: Simd>(simd: S) {
    use crate::harness::SimdCvtTruncate;
    // Test precise truncation with special values
    let a = f32x16::from_slice(
        simd,
        &[
            1.7,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1e20,
            1e20,
            0.0,
            -0.0,
            i32::MAX as f32,
            i32::MIN as f32,
            0.5,
            -0.5,
            0.9999,
            -0.9999,
            2.5,
            -2.5,
        ],
    );
    let result = i32x16::truncate_from_precise(a);
    // NaN -> 0, infinity -> saturated
    assert_eq!(result[0], 1);
    assert_eq!(result[1], 0); // NaN
    assert_eq!(result[2], i32::MAX); // +inf saturates
    assert_eq!(result[3], i32::MIN); // -inf saturates
    assert_eq!(result[4], i32::MIN); // -1e20 saturates to MIN
    assert_eq!(result[5], i32::MAX); // 1e20 saturates to MAX
    assert_eq!(result[6], 0);
    assert_eq!(result[7], 0);
}

#[simd_test]
fn cvt_u32_precise_f32x16<S: Simd>(simd: S) {
    use crate::harness::SimdCvtTruncate;
    // Test precise truncation with special values
    let a = f32x16::from_slice(
        simd,
        &[
            1.7,
            f32::NAN,
            f32::INFINITY,
            0.0,
            1e20,
            0.5,
            -1.0,
            u32::MAX as f32,
            2.5,
            3.9,
            100.1,
            200.9,
            0.001,
            999.999,
            1.0,
            2.0,
        ],
    );
    let result = u32x16::truncate_from_precise(a);
    // NaN -> 0, infinity -> saturated, negative -> 0
    assert_eq!(result[0], 1);
    assert_eq!(result[1], 0); // NaN
    assert_eq!(result[2], u32::MAX); // +inf saturates
    assert_eq!(result[3], 0);
    assert_eq!(result[4], u32::MAX); // 1e20 saturates to MAX
    assert_eq!(result[5], 0); // 0.5 truncates to 0
    assert_eq!(result[6], 0); // -1.0 clamps to 0
}
