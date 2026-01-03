// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![expect(
    clippy::missing_assert_message,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

//! Tests for `fearless_simd`.

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn splat_f32x4<S: Simd>(simd: S) {
    let a = f32x4::splat(simd, 4.2);
    assert_eq!(*a, [4.2, 4.2, 4.2, 4.2]);
}

#[simd_test]
fn abs_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-1.0, 2.0, -3.0, 4.0]);
    assert_eq!(*a.abs(), [1.0, 2.0, 3.0, 4.0]);
}

#[simd_test]
fn neg_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, -2.0, 3.0, -4.0]);
    assert_eq!(*(-a), [-1.0, 2.0, -3.0, 4.0]);
}

#[simd_test]
fn add_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*(a + b), [6.0, 8.0, 10.0, 12.0]);
}

#[simd_test]
fn sub_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[10.0, 20.0, 30.0, 40.0]);
    let b = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(*(a - b), [9.0, 18.0, 27.0, 36.0]);
}

#[simd_test]
fn sqrt_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 0.0, 1.0, 2.0]);
    assert_eq!(*f32x4::sqrt(a), [2.0, 0.0, 1.0, f32::sqrt(2.0)]);
}

#[simd_test]
fn div_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 2.0, 1.0, 0.0]);
    let b = f32x4::from_slice(simd, &[4.0, 1.0, 3.0, 0.1]);
    assert_eq!(*(a / b), [1.0, 2.0, 1.0 / 3.0, 0.0]);
}

#[simd_test]
fn copysign_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, -2.0, -3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[-1.0, 1.0, -1.0, 1.0]);
    assert_eq!(*a.copysign(b), [-1.0, 2.0, -3.0, 4.0]);
}

#[simd_test]
fn simd_eq_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 2.0, 1.0, 0.0]);
    let b = f32x4::from_slice(simd, &[4.0, 3.1, 1.0, 0.0]);
    assert_eq!(*a.simd_eq(b), [-1, 0, -1, -1]);
}

#[simd_test]
fn simd_lt_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 3.0, 2.0, 1.0]);
    let b = f32x4::from_slice(simd, &[1.0, 2.0, 2.0, 4.0]);
    assert_eq!(*a.simd_lt(b), [0, 0, 0, -1]);
}

#[simd_test]
fn simd_le_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 3.0, 2.0, 1.0]);
    let b = f32x4::from_slice(simd, &[1.0, 2.0, 2.0, 4.0]);
    assert_eq!(*a.simd_le(b), [0, 0, -1, -1]);
}

#[simd_test]
fn simd_ge_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 3.0, 2.0, 1.0]);
    let b = f32x4::from_slice(simd, &[1.0, 2.0, 2.0, 4.0]);
    assert_eq!(*a.simd_ge(b), [-1, -1, -1, 0]);
}

#[simd_test]
fn simd_gt_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[4.0, 3.0, 2.0, 1.0]);
    let b = f32x4::from_slice(simd, &[1.0, 2.0, 2.0, 4.0]);
    assert_eq!(*a.simd_gt(b), [-1, -1, 0, 0]);
}

#[simd_test]
fn madd_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, -2.0, 7.0, 3.0]);
    let b = f32x4::from_slice(simd, &[5.0, 4.0, 100.0, 8.0]);
    let c = f32x4::from_slice(simd, &[2.0, -3.0, 0.0, 0.5]);
    assert_eq!(*a.mul_add(b, c), [7.0, -11.0, 700.0, 24.5]);
}

#[simd_test]
fn max_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.0, -3.0, 0.0, 0.5]);
    let b = f32x4::from_slice(simd, &[1.0, -2.0, 7.0, 3.0]);
    assert_eq!(*a.max(b), [2.0, -2.0, 7.0, 3.0]);
}

#[simd_test]
fn min_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.0, -3.0, 0.0, 0.5]);
    let b = f32x4::from_slice(simd, &[1.0, -2.0, 7.0, 3.0]);
    assert_eq!(*a.min(b), [1.0, -3.0, 0.0, 0.5]);
}

#[simd_test]
fn max_precise_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.0, -3.0, 0.0, 0.5]);
    let b = f32x4::from_slice(simd, &[1.0, -2.0, 7.0, 3.0]);
    assert_eq!(*a.max_precise(b), [2.0, -2.0, 7.0, 3.0]);
}

#[simd_test]
fn min_precise_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.0, -3.0, 0.0, 0.5]);
    let b = f32x4::from_slice(simd, &[1.0, -2.0, 7.0, 3.0]);
    assert_eq!(*a.min_precise(b), [1.0, -3.0, 0.0, 0.5]);
}

#[simd_test]
fn msub_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.0, 3.0, 4.0, 5.0]);
    let b = f32x4::from_slice(simd, &[10.0, 10.0, 10.0, 10.0]);
    let c = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(*a.mul_sub(b, c), [19.0, 28.0, 37.0, 46.0]);
}

#[simd_test]
fn max_precise_f32x4_with_nan<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[f32::NAN, -3.0, f32::INFINITY, 0.5]);
    let b = f32x4::from_slice(simd, &[1.0, f32::NAN, 7.0, f32::NEG_INFINITY]);
    let result = a.max_precise(b);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -3.0);
    assert_eq!(result[2], f32::INFINITY);
    assert_eq!(result[3], 0.5);
}

#[simd_test]
fn min_precise_f32x4_with_nan<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[f32::NAN, -3.0, f32::INFINITY, 0.5]);
    let b = f32x4::from_slice(simd, &[1.0, f32::NAN, 7.0, f32::NEG_INFINITY]);
    let result = a.min_precise(b);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], -3.0);
    assert_eq!(result[2], 7.0);
    assert_eq!(result[3], f32::NEG_INFINITY);
}

#[simd_test]
fn floor_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.0, -3.2, 0.0, 0.5]);
    assert_eq!(*a.floor(), [2.0, -4.0, 0.0, 0.0]);
}

#[simd_test]
fn ceil_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.1, -3.2, -2.7, 0.5]);
    assert_eq!(*a.ceil(), [3.0, -3.0, -2.0, 1.0]);
}

#[simd_test]
fn round_ties_even_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.3, -3.2, 2.7, -3.6]);
    assert_eq!(*a.round_ties_even(), [2.0, -3.0, 3.0, -4.0]);
    let b = f32x4::from_slice(simd, &[-3.5, -2.5, 1.5, 0.5]);
    assert_eq!(*b.round_ties_even(), [-4.0, -2.0, 2.0, 0.0]);
}

#[simd_test]
fn fract_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.7, -2.3, 3.9, -4.1]);
    assert_eq!(
        *simd.fract_f32x4(a),
        [0.70000005, -0.29999995, 0.9000001, -0.099999905]
    );
}

#[simd_test]
fn trunc_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[2.9, -3.2, 0.0, 0.5]);
    assert_eq!(*a.trunc(), [2.0, -3.0, 0.0, 0.0]);
}

#[simd_test]
fn trunc_f32x4_special_values<S: Simd>(simd: S) {
    let a = f32x4::from_slice(
        simd,
        &[f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -f32::NAN],
    );
    let result = a.trunc();

    // Note: f32::NAN != f32::NAN hence we transmute to compare the bit pattern
    unsafe {
        assert_eq!(
            std::mem::transmute::<[f32; 4], [u32; 4]>(*result),
            std::mem::transmute::<[f32; 4], [u32; 4]>([
                f32::NAN,
                f32::NEG_INFINITY,
                f32::INFINITY,
                -f32::NAN
            ])
        );
    }
}

#[simd_test]
fn combine_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*a.combine(b), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[simd_test]
fn cvt_u32_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 42.7, 3e9, -0.3]);
    assert_eq!(*a.to_int::<u32x4<_>>(), [1, 42, 3000000000, 0]);
}

#[simd_test]
fn cvt_u32_precise_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-1.0, 42.7, 5e9, f32::NAN]);
    assert_eq!(*a.to_int_precise::<u32x4<_>>(), [0, 42, u32::MAX, 0]);
}

#[simd_test]
fn cvt_f32_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0, 42, 1000000, u32::MAX]);
    assert_eq!(
        *a.to_float::<f32x4<_>>(),
        [0.0, 42.0, 1000000.0, u32::MAX as f32]
    );
}

#[simd_test]
fn cvt_u32_f32x4_rounding<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[0.0, 0.49, 0.51, 0.99]);
    assert_eq!(*a.to_int::<u32x4<_>>(), [0, 0, 0, 0]);
}

#[simd_test]
fn cvt_u32_precise_f32x4_sat<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, 3000000000.0, 5e9, -5e9]);
    assert_eq!(
        *a.to_int_precise::<u32x4<_>>(),
        [0, 3000000000, u32::MAX, 0]
    );
}

#[simd_test]
fn cvt_u32_precise_f32x4_inf<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);

    assert_eq!(*a.to_int_precise::<u32x4<_>>(), [0, 0, u32::MAX, u32::MIN]);
}

#[simd_test]
fn cvt_i32_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, -0.9, 13.34, 234234.8]);

    assert_eq!(*a.to_int::<i32x4<_>>(), [-10, 0, 13, 234234]);
}

#[simd_test]
fn cvt_i32_precise_f32x4_sat<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, f32::NAN, 5e9, -5e9]);

    assert_eq!(
        *a.to_int_precise::<i32x4<_>>(),
        [-10, 0, i32::MAX, i32::MIN]
    );
}

#[simd_test]
fn cvt_i32_precise_f32x4_inf<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);

    assert_eq!(
        *a.to_int_precise::<i32x4<_>>(),
        [-10, 0, i32::MAX, i32::MIN]
    );
}

#[simd_test]
fn cvt_f32_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[-1, 42, 1000000, i32::MAX]);
    assert_eq!(
        *a.to_float::<f32x4<_>>(),
        [-1.0, 42.0, 1000000.0, i32::MAX as f32]
    );
}

#[simd_test]
fn and_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
        ],
    );
    assert_eq!(
        *(a & b),
        [85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0, 85, 0]
    );
}

#[simd_test]
fn or_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8],
    );
    let b = i8x16::from_slice(simd, &[1, 1, 1, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        *(a | b),
        [1, 1, 3, 3, 6, 7, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8]
    );
}

#[simd_test]
fn xor_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, 0, 0, 0, 0]);
    let b = i8x16::from_slice(
        simd,
        &[-1, -1, 0, 0, 5, 4, 7, 6, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    assert_eq!(
        *(a ^ b),
        [-1, -2, 2, 3, 1, 1, 1, 1, 0, -1, 0, -1, -1, 0, -1, 0]
    );
}

#[simd_test]
fn not_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8],
    );
    assert_eq!(
        *(!a),
        [-1, -2, -3, -4, -5, -6, -7, -8, 0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[simd_test]
fn add_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8],
    );
    let b = i8x16::from_slice(
        simd,
        &[10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8],
    );
    assert_eq!(
        *(a + b),
        [11, 22, 33, 44, 55, 66, 77, 88, 0, 0, 0, 0, 0, 0, 0, 0]
    );
}

#[simd_test]
fn sub_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[10, 20, 30, 40, 50, 60, 70, 80, 0, 0, 0, 0, 0, 0, 0, 0],
    );
    let b = i8x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        *(a - b),
        [
            9, 18, 27, 36, 45, 54, 63, 72, -1, -2, -3, -4, -5, -6, -7, -8
        ]
    );
}

#[simd_test]
fn neg_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    assert_eq!(
        *(-a),
        [
            -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16
        ]
    );
}

#[simd_test]
fn simd_eq_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i8x16::from_slice(simd, &[1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0]);
    assert_eq!(
        *a.simd_eq(b),
        [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]
    );
}

#[simd_test]
fn simd_lt_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, -1, -2, -3, -4, 10, 20, 30, 40, 50, 60, 70, 80],
    );
    let b = i8x16::from_slice(
        simd,
        &[2, 2, 2, 5, 0, 0, 0, 0, 5, 25, 25, 45, 45, 65, 65, 85],
    );
    assert_eq!(
        *a.simd_lt(b),
        [-1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1]
    );
}

#[simd_test]
fn simd_gt_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[2, 2, 2, 5, 0, 0, 0, 0, 5, 25, 25, 45, 45, 65, 65, 85],
    );
    let b = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, -1, -2, -3, -4, 10, 20, 30, 40, 50, 60, 70, 80],
    );
    assert_eq!(
        *a.simd_gt(b),
        [-1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1]
    );
}

#[simd_test]
fn min_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16
        ]
    );
}

#[simd_test]
fn max_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15,
        ],
    );
    assert_eq!(
        *a.max(b),
        [
            2, -1, 4, -3, 6, -5, 8, -7, 10, -9, 12, -11, 14, -13, 16, -15
        ]
    );
}

#[simd_test]
fn combine_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
        ],
    );
    assert_eq!(
        *a.combine(b),
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1, -2, -3, -4, -5, -6, -7, -8,
            -9, -10, -11, -12, -13, -14, -15, -16
        ]
    );
}

#[simd_test]
fn and_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(simd, &[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
    let b = u8x16::from_slice(
        simd,
        &[
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
        ],
    );
    assert_eq!(*(a & b), [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
}

#[simd_test]
fn or_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u8x16::from_slice(simd, &[1, 1, 1, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(*(a | b), [1, 1, 3, 3, 6, 7, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn xor_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 0, 0, 0, 0]);
    let b = u8x16::from_slice(simd, &[1, 1, 0, 0, 5, 4, 7, 6, 1, 0, 1, 0, 1, 0, 1, 0]);
    assert_eq!(*(a ^ b), [1, 0, 2, 3, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]);
}

#[simd_test]
fn not_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        *(!a),
        [
            255, 254, 253, 252, 251, 250, 249, 248, 254, 253, 252, 251, 250, 249, 248, 247
        ]
    );
}

#[simd_test]
fn add_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    assert_eq!(
        *(a + b),
        [
            11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176
        ]
    );
}

#[simd_test]
fn sub_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        ],
    );
    let b = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    assert_eq!(
        *(a - b),
        [
            99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84
        ]
    );
}

#[simd_test]
fn min_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            15, 15, 35, 35, 45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165,
        ],
    );
    assert_eq!(
        *a.min(b),
        [
            10, 15, 30, 35, 45, 60, 65, 80, 85, 100, 105, 120, 125, 140, 145, 160
        ]
    );
}

#[simd_test]
fn max_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            15, 15, 35, 35, 45, 65, 65, 85, 85, 105, 105, 125, 125, 145, 145, 165,
        ],
    );
    assert_eq!(
        *a.max(b),
        [
            15, 20, 35, 40, 50, 65, 70, 85, 90, 105, 110, 125, 130, 145, 150, 165
        ]
    );
}

#[simd_test]
fn combine_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *a.combine(b),
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        ]
    );
}

#[simd_test]
fn and_mask8x16<S: Simd>(simd: S) {
    let a = mask8x16::from_slice(simd, &[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
    let b = mask8x16::from_slice(
        simd,
        &[
            85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
        ],
    );
    assert_eq!(*(a & b), [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
}

#[simd_test]
fn or_mask8x16<S: Simd>(simd: S) {
    let a = mask8x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
    let b = mask8x16::from_slice(simd, &[1, 1, 1, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(*(a | b), [1, 1, 3, 3, 6, 7, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn xor_mask8x16<S: Simd>(simd: S) {
    let a = mask8x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 1, 1, 0, 0, 0, 0]);
    let b = mask8x16::from_slice(simd, &[1, 1, 0, 0, 5, 4, 7, 6, 1, 0, 1, 0, 1, 0, 1, 0]);
    assert_eq!(*(a ^ b), [1, 0, 2, 3, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]);
}

#[simd_test]
fn not_mask8x16<S: Simd>(simd: S) {
    let a = mask8x16::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        *(!a),
        [
            -1, -2, -3, -4, -5, -6, -7, -8, -2, -3, -4, -5, -6, -7, -8, -9
        ]
    );
}

#[simd_test]
fn any_true_mask8x16<S: Simd>(simd: S) {
    let all_zero = mask8x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(!simd.any_true_mask8x16(all_zero));

    let all_neg = mask8x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.any_true_mask8x16(all_neg));

    let one_neg = mask8x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]);
    assert!(simd.any_true_mask8x16(one_neg));
}

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
fn any_false_mask8x16<S: Simd>(simd: S) {
    let all_zero = mask8x16::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(simd.any_false_mask8x16(all_zero));

    let all_neg = mask8x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(!simd.any_false_mask8x16(all_neg));

    let one_pos = mask8x16::from_slice(
        simd,
        &[
            -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1,
        ],
    );
    assert!(simd.any_false_mask8x16(one_pos));
}

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
fn load_interleaved_128_u32x16<S: Simd>(simd: S) {
    #[rustfmt::skip]
    let data: [u32; 16] = [
        1, 2, 3, 4,
        10, 20, 30, 40,
        100, 200, 300, 400,
        1000, 2000, 3000, 4000,
    ];
    assert_eq!(
        *simd.load_interleaved_128_u32x16(&data),
        [
            1, 10, 100, 1000, 2, 20, 200, 2000, 3, 30, 300, 3000, 4, 40, 400, 4000
        ]
    );
}

#[simd_test]
fn load_interleaved_128_u16x32<S: Simd>(simd: S) {
    #[rustfmt::skip]
    let data: [u16; 32] = [
        1, 2, 3, 4,
        5, 6, 7, 8,

        10, 20, 30, 40,
        50, 60, 70, 80,

        100, 200, 300, 400,
        500, 600, 700, 800,

        1000, 2000, 3000, 4000,
        5000, 6000, 7000, 8000,
    ];
    assert_eq!(
        *simd.load_interleaved_128_u16x32(&data),
        [
            1, 5, 10, 50, 100, 500, 1000, 5000, 2, 6, 20, 60, 200, 600, 2000, 6000, 3, 7, 30, 70,
            300, 700, 3000, 7000, 4, 8, 40, 80, 400, 800, 4000, 8000
        ]
    );
}

#[simd_test]
fn load_interleaved_128_u8x64<S: Simd>(simd: S) {
    #[rustfmt::skip]
    let data: [u8; 64] = [
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,

        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31,

        32, 33, 34, 35,
        36, 37, 38, 39,
        40, 41, 42, 43,
        44, 45, 46, 47,

        48, 49, 50, 51,
        52, 53, 54, 55,
        56, 57, 58, 59,
        60, 61, 62, 63,
    ];
    assert_eq!(
        *simd.load_interleaved_128_u8x64(&data),
        [
            0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25,
            29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50,
            54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
        ]
    );
}

#[simd_test]
fn store_interleaved_128_f32x16<S: Simd>(simd: S) {
    let input = [
        0.0,
        f32::NAN,
        f32::INFINITY,
        -3.0,
        4.0,
        -0.0,
        6.0,
        f32::NEG_INFINITY,
        8.0,
        9.0,
        -10.0,
        11.0,
        f32::MIN,
        13.0,
        f32::MAX,
        15.0,
    ];
    let a = f32x16::from_slice(simd, &input);
    let mut dest = [0.0_f32; 16];
    simd.store_interleaved_128_f32x16(a, &mut dest);

    let expected = [
        0.0,
        4.0,
        8.0,
        f32::MIN,
        f32::NAN,
        -0.0,
        9.0,
        13.0,
        f32::INFINITY,
        6.0,
        -10.0,
        f32::MAX,
        -3.0,
        f32::NEG_INFINITY,
        11.0,
        15.0,
    ];

    // Note: f32::NAN != f32::NAN hence we transmute to compare the bit pattern
    unsafe {
        assert_eq!(
            std::mem::transmute::<[f32; 16], [u32; 16]>(dest),
            std::mem::transmute::<[f32; 16], [u32; 16]>(expected)
        );
    }
}

#[simd_test]
fn store_interleaved_128_u8x64<S: Simd>(simd: S) {
    let input: [u8; 64] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ];
    let a = u8x64::from_slice(simd, &input);
    let mut dest = [0_u8; 64];
    simd.store_interleaved_128_u8x64(a, &mut dest);

    let expected = [
        0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53,
        6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63,
    ];

    assert_eq!(dest, expected);
}

#[simd_test]
fn store_interleaved_128_u16x32<S: Simd>(simd: S) {
    let input: [u16; 32] = [
        0, 1, 2, 3, 4, 5, 6, 7, 100, 101, 102, 103, 104, 105, 106, 107, 200, 201, 202, 203, 204,
        205, 206, 207, 300, 301, 302, 303, 304, 305, 306, 307,
    ];
    let a = u16x32::from_slice(simd, &input);
    let mut dest = [0_u16; 32];
    simd.store_interleaved_128_u16x32(a, &mut dest);

    let expected = [
        0, 100, 200, 300, 1, 101, 201, 301, 2, 102, 202, 302, 3, 103, 203, 303, 4, 104, 204, 304,
        5, 105, 205, 305, 6, 106, 206, 306, 7, 107, 207, 307,
    ];

    assert_eq!(dest, expected);
}

#[simd_test]
fn store_interleaved_128_u32x16<S: Simd>(simd: S) {
    let input: [u32; 16] = [
        0,
        1,
        u32::MAX,
        3,
        1000,
        1001,
        1002,
        1003,
        2000,
        2001,
        2002,
        2003,
        u32::MIN,
        3001,
        3002,
        u32::MAX - 1,
    ];
    let a = u32x16::from_slice(simd, &input);
    let mut dest = [0_u32; 16];
    simd.store_interleaved_128_u32x16(a, &mut dest);

    let expected = [
        0,
        1000,
        2000,
        u32::MIN,
        1,
        1001,
        2001,
        3001,
        u32::MAX,
        1002,
        2002,
        3002,
        3,
        1003,
        2003,
        u32::MAX - 1,
    ];

    assert_eq!(dest, expected);
}

#[simd_test]
fn zip_low_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[0.0, 1.0, 2.0, 3.0]);
    let b = f32x4::from_slice(simd, &[4.0, 5.0, 6.0, 7.0]);
    assert_eq!(*simd.zip_low_f32x4(a, b), [0.0, 4.0, 1.0, 5.0]);
}

#[simd_test]
fn zip_low_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b = f32x8::from_slice(simd, &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    assert_eq!(
        *simd.zip_low_f32x8(a, b),
        [0.0, 8.0, 1.0, 9.0, 2.0, 10.0, 3.0, 11.0]
    );
}

#[simd_test]
fn zip_high_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[0.0, 1.0, 2.0, 3.0]);
    let b = f32x4::from_slice(simd, &[4.0, 5.0, 6.0, 7.0]);
    assert_eq!(*simd.zip_high_f32x4(a, b), [2.0, 6.0, 3.0, 7.0]);
}

#[simd_test]
fn zip_high_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b = f32x8::from_slice(simd, &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    assert_eq!(
        *simd.zip_high_f32x8(a, b),
        [4.0, 12.0, 5.0, 13.0, 6.0, 14.0, 7.0, 15.0]
    );
}

#[simd_test]
fn zip_low_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    assert_eq!(
        *simd.zip_low_i8x16(a, b),
        [
            1, 17, -2, -18, 3, 19, -4, -20, 5, 21, -6, -22, 7, 23, -8, -24
        ]
    );
}

#[simd_test]
fn zip_high_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    assert_eq!(
        *simd.zip_high_i8x16(a, b),
        [
            9, 25, -10, -26, 11, 27, -12, -28, 13, 29, -14, -30, 15, 31, -16, -32
        ]
    );
}

#[simd_test]
fn zip_low_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            33, -34, 35, -36, 37, -38, 39, -40, 41, -42, 43, -44, 45, -46, 47, -48, 49, -50, 51,
            -52, 53, -54, 55, -56, 57, -58, 59, -60, 61, -62, 63, -64,
        ],
    );
    assert_eq!(
        *simd.zip_low_i8x32(a, b),
        [
            1, 33, -2, -34, 3, 35, -4, -36, 5, 37, -6, -38, 7, 39, -8, -40, 9, 41, -10, -42, 11,
            43, -12, -44, 13, 45, -14, -46, 15, 47, -16, -48
        ]
    );
}

#[simd_test]
fn zip_high_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20, 21,
            -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            33, -34, 35, -36, 37, -38, 39, -40, 41, -42, 43, -44, 45, -46, 47, -48, 49, -50, 51,
            -52, 53, -54, 55, -56, 57, -58, 59, -60, 61, -62, 63, -64,
        ],
    );
    assert_eq!(
        *simd.zip_high_i8x32(a, b),
        [
            17, 49, -18, -50, 19, 51, -20, -52, 21, 53, -22, -54, 23, 55, -24, -56, 25, 57, -26,
            -58, 27, 59, -28, -60, 29, 61, -30, -62, 31, 63, -32, -64
        ]
    );
}

#[simd_test]
fn zip_low_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    assert_eq!(
        *simd.zip_low_u8x16(a, b),
        [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
    );
}

#[simd_test]
fn zip_high_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    assert_eq!(
        *simd.zip_high_u8x16(a, b),
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    );
}

#[simd_test]
fn zip_low_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    );
    assert_eq!(
        *simd.zip_low_u8x32(a, b),
        [
            0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43,
            12, 44, 13, 45, 14, 46, 15, 47
        ]
    );
}

#[simd_test]
fn zip_high_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    );
    assert_eq!(
        *simd.zip_high_u8x32(a, b),
        [
            16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58,
            27, 59, 28, 60, 29, 61, 30, 62, 31, 63
        ]
    );
}

#[simd_test]
fn zip_low_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i16x8::from_slice(simd, &[9, -10, 11, -12, 13, -14, 15, -16]);
    assert_eq!(*simd.zip_low_i16x8(a, b), [1, 9, -2, -10, 3, 11, -4, -12]);
}

#[simd_test]
fn zip_high_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i16x8::from_slice(simd, &[9, -10, 11, -12, 13, -14, 15, -16]);
    assert_eq!(*simd.zip_high_i16x8(a, b), [5, 13, -6, -14, 7, 15, -8, -16]);
}

#[simd_test]
fn zip_low_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    assert_eq!(
        *simd.zip_low_i16x16(a, b),
        [
            1, 17, -2, -18, 3, 19, -4, -20, 5, 21, -6, -22, 7, 23, -8, -24
        ]
    );
}

#[simd_test]
fn zip_high_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[
            1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
        ],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            17, -18, 19, -20, 21, -22, 23, -24, 25, -26, 27, -28, 29, -30, 31, -32,
        ],
    );
    assert_eq!(
        *simd.zip_high_i16x16(a, b),
        [
            9, 25, -10, -26, 11, 27, -12, -28, 13, 29, -14, -30, 15, 31, -16, -32
        ]
    );
}

#[simd_test]
fn zip_low_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let b = u16x8::from_slice(simd, &[8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(*simd.zip_low_u16x8(a, b), [0, 8, 1, 9, 2, 10, 3, 11]);
}

#[simd_test]
fn zip_high_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let b = u16x8::from_slice(simd, &[8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(*simd.zip_high_u16x8(a, b), [4, 12, 5, 13, 6, 14, 7, 15]);
}

#[simd_test]
fn zip_low_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    assert_eq!(
        *simd.zip_low_u16x16(a, b),
        [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
    );
}

#[simd_test]
fn zip_high_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    assert_eq!(
        *simd.zip_high_u16x16(a, b),
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    );
}

#[simd_test]
fn zip_low_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, -2, 3, -4]);
    let b = i32x4::from_slice(simd, &[5, -6, 7, -8]);
    assert_eq!(*simd.zip_low_i32x4(a, b), [1, 5, -2, -6]);
}

#[simd_test]
fn zip_high_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, -2, 3, -4]);
    let b = i32x4::from_slice(simd, &[5, -6, 7, -8]);
    assert_eq!(*simd.zip_high_i32x4(a, b), [3, 7, -4, -8]);
}

#[simd_test]
fn zip_low_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i32x8::from_slice(simd, &[9, -10, 11, -12, 13, -14, 15, -16]);
    assert_eq!(*simd.zip_low_i32x8(a, b), [1, 9, -2, -10, 3, 11, -4, -12]);
}

#[simd_test]
fn zip_high_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i32x8::from_slice(simd, &[9, -10, 11, -12, 13, -14, 15, -16]);
    assert_eq!(*simd.zip_high_i32x8(a, b), [5, 13, -6, -14, 7, 15, -8, -16]);
}

#[simd_test]
fn zip_low_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0, 1, 2, 3]);
    let b = u32x4::from_slice(simd, &[4, 5, 6, 7]);
    assert_eq!(*simd.zip_low_u32x4(a, b), [0, 4, 1, 5]);
}

#[simd_test]
fn zip_high_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0, 1, 2, 3]);
    let b = u32x4::from_slice(simd, &[4, 5, 6, 7]);
    assert_eq!(*simd.zip_high_u32x4(a, b), [2, 6, 3, 7]);
}

#[simd_test]
fn zip_low_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let b = u32x8::from_slice(simd, &[8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(*simd.zip_low_u32x8(a, b), [0, 8, 1, 9, 2, 10, 3, 11]);
}

#[simd_test]
fn zip_high_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let b = u32x8::from_slice(simd, &[8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(*simd.zip_high_u32x8(a, b), [4, 12, 5, 13, 6, 14, 7, 15]);
}

#[simd_test]
fn zip_low_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*simd.zip_low_f64x4(a, b), [1.0, 5.0, 2.0, 6.0]);
}

#[simd_test]
fn zip_high_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*simd.zip_high_f64x4(a, b), [3.0, 7.0, 4.0, 8.0]);
}

#[simd_test]
fn unzip_low_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*simd.unzip_low_f32x4(a, b), [1.0, 3.0, 5.0, 7.0]);
}

#[simd_test]
fn unzip_low_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b = f32x8::from_slice(simd, &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    assert_eq!(
        *simd.unzip_low_f32x8(a, b),
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    );
}

#[simd_test]
fn unzip_high_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*simd.unzip_high_f32x4(a, b), [2.0, 4.0, 6.0, 8.0]);
}

#[simd_test]
fn unzip_high_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let b = f32x8::from_slice(simd, &[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    assert_eq!(
        *simd.unzip_high_f32x8(a, b),
        [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    );
}

#[simd_test]
fn unzip_low_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_low_i8x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_high_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_high_i8x16(a, b),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    );
}

#[simd_test]
fn unzip_low_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ],
    );
    assert_eq!(
        *simd.unzip_low_i8x32(a, b),
        [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45,
            47, 49, 51, 53, 55, 57, 59, 61, 63
        ]
    );
}

#[simd_test]
fn unzip_high_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ],
    );
    assert_eq!(
        *simd.unzip_high_i8x32(a, b),
        [
            2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62, 64
        ]
    );
}

#[simd_test]
fn unzip_low_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_low_u8x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_high_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_high_u8x16(a, b),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    );
}

#[simd_test]
fn unzip_low_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ],
    );
    assert_eq!(
        *simd.unzip_low_u8x32(a, b),
        [
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45,
            47, 49, 51, 53, 55, 57, 59, 61, 63
        ]
    );
}

#[simd_test]
fn unzip_high_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ],
    );
    assert_eq!(
        *simd.unzip_high_u8x32(a, b),
        [
            2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62, 64
        ]
    );
}

#[simd_test]
fn unzip_low_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_low_i16x8(a, b), [1, 3, 5, 7, 9, 11, 13, 15]);
}

#[simd_test]
fn unzip_high_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_high_i16x8(a, b), [2, 4, 6, 8, 10, 12, 14, 16]);
}

#[simd_test]
fn unzip_low_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u16x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_low_u16x8(a, b), [1, 3, 5, 7, 9, 11, 13, 15]);
}

#[simd_test]
fn unzip_high_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u16x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_high_u16x8(a, b), [2, 4, 6, 8, 10, 12, 14, 16]);
}

#[simd_test]
fn unzip_low_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_low_i16x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_high_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_high_i16x16(a, b),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    );
}

#[simd_test]
fn unzip_low_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_low_u16x16(a, b),
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    );
}

#[simd_test]
fn unzip_high_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *simd.unzip_high_u16x16(a, b),
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    );
}

#[simd_test]
fn unzip_low_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*simd.unzip_low_i32x4(a, b), [1, 3, 5, 7]);
}

#[simd_test]
fn unzip_high_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*simd.unzip_high_i32x4(a, b), [2, 4, 6, 8]);
}

#[simd_test]
fn unzip_low_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*simd.unzip_low_u32x4(a, b), [1, 3, 5, 7]);
}

#[simd_test]
fn unzip_high_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*simd.unzip_high_u32x4(a, b), [2, 4, 6, 8]);
}

#[simd_test]
fn unzip_low_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_low_i32x8(a, b), [1, 3, 5, 7, 9, 11, 13, 15]);
}

#[simd_test]
fn unzip_high_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_high_i32x8(a, b), [2, 4, 6, 8, 10, 12, 14, 16]);
}

#[simd_test]
fn unzip_low_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_low_u32x8(a, b), [1, 3, 5, 7, 9, 11, 13, 15]);
}

#[simd_test]
fn unzip_high_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(*simd.unzip_high_u32x8(a, b), [2, 4, 6, 8, 10, 12, 14, 16]);
}

#[simd_test]
fn unzip_low_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.0, 2.0]);
    let b = f64x2::from_slice(simd, &[3.0, 4.0]);
    assert_eq!(*simd.unzip_low_f64x2(a, b), [1.0, 3.0]);
}

#[simd_test]
fn unzip_high_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.0, 2.0]);
    let b = f64x2::from_slice(simd, &[3.0, 4.0]);
    assert_eq!(*simd.unzip_high_f64x2(a, b), [2.0, 4.0]);
}

#[simd_test]
fn unzip_low_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*simd.unzip_low_f64x4(a, b), [1.0, 3.0, 5.0, 7.0]);
}

#[simd_test]
fn unzip_high_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*simd.unzip_high_f64x4(a, b), [2.0, 4.0, 6.0, 8.0]);
}

#[simd_test]
fn shr_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            -128, -64, -32, -16, -8, -4, -2, -1, 127, 64, 32, 16, 8, 4, 2, 1,
        ],
    );
    assert_eq!(
        *(a >> 2),
        [-32, -16, -8, -4, -2, -1, -1, -1, 31, 16, 8, 4, 2, 1, 0, 0]
    );
}

#[simd_test]
fn shr_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[255, 128, 64, 32, 16, 8, 4, 2, 254, 127, 63, 31, 15, 7, 3, 1],
    );
    assert_eq!(
        *(a >> 2),
        [63, 32, 16, 8, 4, 2, 1, 0, 63, 31, 15, 7, 3, 1, 0, 0]
    );
}

#[simd_test]
fn shr_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[-32768, -16384, -1024, -1, 32767, 16384, 1024, 1]);
    assert_eq!(*(a >> 4), [-2048, -1024, -64, -1, 2047, 1024, 64, 0]);
}

#[simd_test]
fn shr_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[65535, 32768, 16384, 8192, 4096, 2048, 1024, 512]);
    assert_eq!(*(a >> 4), [4095, 2048, 1024, 512, 256, 128, 64, 32]);
}

#[simd_test]
fn shr_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[i32::MIN, -65536, 65536, i32::MAX]);
    assert_eq!(*(a >> 8), [-8388608, -256, 256, 8388607]);
}

#[simd_test]
fn shrv_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[i32::MIN, -65536, 65536, i32::MAX]);
    assert_eq!(
        *(a >> i32x4::splat(simd, 8)),
        [-8388608, -256, 256, 8388607]
    );
}

#[simd_test]
fn shr_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[u32::MAX, 2147483648, 65536, 256]);
    assert_eq!(*(a >> 8), [16777215, 8388608, 256, 1]);
}

#[simd_test]
fn shrv_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[u32::MAX, 2147483648, 65536, 256]);
    assert_eq!(*(a >> u32x4::splat(simd, 8)), [16777215, 8388608, 256, 1]);
}

#[simd_test]
fn shrv_u32x4_varied<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[u32::MAX; 4]);
    const SHIFTS: [u32; 4] = [0, 1, 2, 3];
    assert_eq!(
        *(a >> u32x4::from_slice(simd, &SHIFTS)),
        SHIFTS.map(|x| u32::MAX >> x)
    );
}

#[simd_test]
fn shl_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0xFFFFFFFF, 0xFFFF, 0xFF, 0]);
    assert_eq!(*(a << 4), [0xFFFFFFF0, 0xFFFF0, 0xFF0, 0]);
}

#[simd_test]
fn shlv_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0xFFFFFFFF, 0xFFFF, 0xFF, 0]);
    assert_eq!(
        *(a << u32x4::splat(simd, 4)),
        [0xFFFFFFF0, 0xFFFF0, 0xFF0, 0]
    );
}

#[simd_test]
fn shlv_u32x4_varied<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[u32::MAX; 4]);
    const SHIFTS: [u32; 4] = [0, 1, 2, 3];
    assert_eq!(
        *(a << u32x4::from_slice(simd, &SHIFTS)),
        SHIFTS.map(|x| u32::MAX << x)
    );
}

#[simd_test]
fn add_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    assert_eq!(*(a + b), [11, 22, 33, 44, 55, 66, 77, 88]);
}

#[simd_test]
fn sub_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[100, 200, 300, 400, 500, 600, 700, 800]);
    let b = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*(a - b), [99, 198, 297, 396, 495, 594, 693, 792]);
}

#[simd_test]
fn neg_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    assert_eq!(*(-a), [-1, 2, -3, 4, -5, 6, -7, 8]);
}

#[simd_test]
fn simd_eq_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x8::from_slice(simd, &[1, 0, 3, 0, 5, 0, 7, 0]);
    assert_eq!(*a.simd_eq(b), [-1, 0, -1, 0, -1, 0, -1, 0]);
}

#[simd_test]
fn simd_lt_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, -1, -2, -3, -4]);
    let b = i16x8::from_slice(simd, &[2, 2, 2, 5, 0, 0, 0, 0]);
    assert_eq!(*a.simd_lt(b), [-1, 0, 0, -1, -1, -1, -1, -1]);
}

#[simd_test]
fn simd_gt_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[2, 2, 2, 5, 0, 0, 0, 0]);
    let b = i16x8::from_slice(simd, &[1, 2, 3, 4, -1, -2, -3, -4]);
    assert_eq!(*a.simd_gt(b), [-1, 0, 0, -1, -1, -1, -1, -1]);
}

#[simd_test]
fn min_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i16x8::from_slice(simd, &[2, -1, 4, -3, 6, -5, 8, -7]);
    assert_eq!(*a.min(b), [1, -2, 3, -4, 5, -6, 7, -8]);
}

#[simd_test]
fn max_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, -2, 3, -4, 5, -6, 7, -8]);
    let b = i16x8::from_slice(simd, &[2, -1, 4, -3, 6, -5, 8, -7]);
    assert_eq!(*a.max(b), [2, -1, 4, -3, 6, -5, 8, -7]);
}

#[simd_test]
fn combine_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(
        *a.combine(b),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
}

#[simd_test]
fn add_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u16x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    assert_eq!(*(a + b), [11, 22, 33, 44, 55, 66, 77, 88]);
}

#[simd_test]
fn sub_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[100, 200, 300, 400, 500, 600, 700, 800]);
    let b = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*(a - b), [99, 198, 297, 396, 495, 594, 693, 792]);
}

#[simd_test]
fn simd_eq_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 32768, 40000, 65535, 6, 7, 8]);
    let b = u16x8::from_slice(simd, &[1, 0, 32768, 0, 65535, 0, 7, 0]);
    assert_eq!(*a.simd_eq(b), [-1, 0, -1, 0, -1, 0, -1, 0]);
}

#[simd_test]
fn simd_lt_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 3, 4, 100, 200, 300, 400]);
    let b = u16x8::from_slice(simd, &[2, 2, 2, 5, 40000, 150, 50000, 350]);
    assert_eq!(*a.simd_lt(b), [-1, 0, 0, -1, -1, 0, -1, 0]);
}

#[simd_test]
fn simd_gt_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[2, 2, 2, 5, 40000, 150, 50000, 350]);
    let b = u16x8::from_slice(simd, &[1, 2, 3, 4, 100, 200, 300, 400]);
    assert_eq!(*a.simd_gt(b), [-1, 0, 0, -1, -1, 0, -1, 0]);
}

#[simd_test]
fn min_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    let b = u16x8::from_slice(simd, &[15, 15, 35, 35, 45, 65, 65, 85]);
    assert_eq!(*a.min(b), [10, 15, 30, 35, 45, 60, 65, 80]);
}

#[simd_test]
fn max_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[10, 20, 30, 40, 50, 60, 70, 80]);
    let b = u16x8::from_slice(simd, &[15, 15, 35, 35, 45, 65, 65, 85]);
    assert_eq!(*a.max(b), [15, 20, 35, 40, 50, 65, 70, 85]);
}

#[simd_test]
fn combine_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = u16x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(
        *a.combine(b),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
}

#[simd_test]
fn add_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i32x4::from_slice(simd, &[10, 20, 30, 40]);
    assert_eq!(*(a + b), [11, 22, 33, 44]);
}

#[simd_test]
fn sub_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[100, 200, 300, 400]);
    let b = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    assert_eq!(*(a - b), [99, 198, 297, 396]);
}

#[simd_test]
fn simd_eq_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i32x4::from_slice(simd, &[1, 0, 3, 0]);
    assert_eq!(*a.simd_eq(b), [-1, 0, -1, 0]);
}

#[simd_test]
fn simd_lt_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, -3, -4]);
    let b = i32x4::from_slice(simd, &[2, 2, 0, 0]);
    assert_eq!(*a.simd_lt(b), [-1, 0, -1, -1]);
}

#[simd_test]
fn simd_gt_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[2, 2, 0, 0]);
    let b = i32x4::from_slice(simd, &[1, 2, -3, -4]);
    assert_eq!(*a.simd_gt(b), [-1, 0, -1, -1]);
}

#[simd_test]
fn min_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, -2, 3, -4]);
    let b = i32x4::from_slice(simd, &[2, -1, 4, -3]);
    assert_eq!(*a.min(b), [1, -2, 3, -4]);
}

#[simd_test]
fn max_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, -2, 3, -4]);
    let b = i32x4::from_slice(simd, &[2, -1, 4, -3]);
    assert_eq!(*a.max(b), [2, -1, 4, -3]);
}

#[simd_test]
fn combine_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = i32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.combine(b), [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn add_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u32x4::from_slice(simd, &[10, 20, 30, 40]);
    assert_eq!(*(a + b), [11, 22, 33, 44]);
}

#[simd_test]
fn sub_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[100, 200, 300, 400]);
    let b = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    assert_eq!(*(a - b), [99, 198, 297, 396]);
}

#[simd_test]
fn simd_eq_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 2147483648, 4294967295]);
    let b = u32x4::from_slice(simd, &[1, 0, 2147483648, 0]);
    assert_eq!(*a.simd_eq(b), [-1, 0, -1, 0]);
}

#[simd_test]
fn simd_lt_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 100, 200]);
    let b = u32x4::from_slice(simd, &[2, 2, 3000000000, 150]);
    assert_eq!(*a.simd_lt(b), [-1, 0, -1, 0]);
}

#[simd_test]
fn simd_gt_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[2, 2, 3000000000, 150]);
    let b = u32x4::from_slice(simd, &[1, 2, 100, 200]);
    assert_eq!(*a.simd_gt(b), [-1, 0, -1, 0]);
}

#[simd_test]
fn min_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[10, 20, 30, 40]);
    let b = u32x4::from_slice(simd, &[15, 15, 35, 35]);
    assert_eq!(*a.min(b), [10, 15, 30, 35]);
}

#[simd_test]
fn max_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[10, 20, 30, 40]);
    let b = u32x4::from_slice(simd, &[15, 15, 35, 35]);
    assert_eq!(*a.max(b), [15, 20, 35, 40]);
}

#[simd_test]
fn combine_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    let b = u32x4::from_slice(simd, &[5, 6, 7, 8]);
    assert_eq!(*a.combine(b), [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn combine_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::from_slice(simd, &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    assert_eq!(
        *a.combine(b),
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
        ]
    );
}

#[simd_test]
fn combine_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let b = i8x32::from_slice(
        simd,
        &[
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ],
    );
    assert_eq!(
        *a.combine(b),
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
        ]
    );
}

#[simd_test]
fn combine_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    let b = u8x32::from_slice(
        simd,
        &[
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        ],
    );
    assert_eq!(
        *a.combine(b),
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
        ]
    );
}

#[simd_test]
fn combine_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let b = i16x16::from_slice(
        simd,
        &[
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    assert_eq!(
        *a.combine(b),
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        ]
    );
}

#[simd_test]
fn combine_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let b = u16x16::from_slice(
        simd,
        &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    assert_eq!(
        *a.combine(b),
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
        ]
    );
}

#[simd_test]
fn combine_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::from_slice(simd, &[9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(
        *a.combine(b),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
}

#[simd_test]
fn combine_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let b = u32x8::from_slice(simd, &[8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(
        *a.combine(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    );
}

#[simd_test]
fn combine_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let b = f64x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*a.combine(b), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[simd_test]
fn split_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let (lo, hi) = simd.split_f32x8(a);
    assert_eq!(*lo, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(*hi, [5.0, 6.0, 7.0, 8.0]);
}

#[simd_test]
fn split_i8x32<S: Simd>(simd: S) {
    let a = i8x32::from_slice(
        simd,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ],
    );
    let (lo, hi) = simd.split_i8x32(a);
    assert_eq!(*lo, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(
        *hi,
        [
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ]
    );
}

#[simd_test]
fn split_u8x32<S: Simd>(simd: S) {
    let a = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ],
    );
    let (lo, hi) = simd.split_u8x32(a);
    assert_eq!(*lo, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(
        *hi,
        [
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
        ]
    );
}

#[simd_test]
fn split_i16x16<S: Simd>(simd: S) {
    let a = i16x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    let (lo, hi) = simd.split_i16x16(a);
    assert_eq!(*lo, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(*hi, [9, 10, 11, 12, 13, 14, 15, 16]);
}

#[simd_test]
fn split_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    );
    let (lo, hi) = simd.split_u16x16(a);
    assert_eq!(*lo, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(*hi, [8, 9, 10, 11, 12, 13, 14, 15]);
}

#[simd_test]
fn split_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let (lo, hi) = simd.split_i32x8(a);
    assert_eq!(*lo, [1, 2, 3, 4]);
    assert_eq!(*hi, [5, 6, 7, 8]);
}

#[simd_test]
fn split_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let (lo, hi) = simd.split_u32x8(a);
    assert_eq!(*lo, [0, 1, 2, 3]);
    assert_eq!(*hi, [4, 5, 6, 7]);
}

#[simd_test]
fn split_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let (lo, hi) = simd.split_f64x4(a);
    assert_eq!(*lo, [1.0, 2.0]);
    assert_eq!(*hi, [3.0, 4.0]);
}

#[simd_test]
fn select_f32x4<S: Simd>(simd: S) {
    let mask = mask32x4::from_slice(simd, &[-1, 0, -1, 0]);
    let b = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let c = f32x4::from_slice(simd, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(*mask.select(b, c), [1.0, 6.0, 3.0, 8.0]);
}

#[simd_test]
fn select_i8x16<S: Simd>(simd: S) {
    let mask = mask8x16::from_slice(
        simd,
        &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    let b = i8x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, -10, -20, -30, -40,
        ],
    );
    let c = i8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -2, -3, -4],
    );
    assert_eq!(
        *mask.select(b, c),
        [
            10, 2, 30, 4, 50, 6, 70, 8, 90, 10, 110, 12, -10, -2, -30, -4
        ]
    );
}

#[simd_test]
fn select_u8x16<S: Simd>(simd: S) {
    let mask = mask8x16::from_slice(
        simd,
        &[0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    );
    let b = u8x16::from_slice(
        simd,
        &[
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ],
    );
    let c = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    );
    assert_eq!(
        *mask.select(b, c),
        [
            1, 20, 3, 40, 5, 60, 7, 80, 9, 100, 11, 120, 13, 140, 15, 160
        ]
    );
}

#[simd_test]
fn select_mask8x16<S: Simd>(simd: S) {
    let mask = mask8x16::from_slice(
        simd,
        &[-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0],
    );
    let b = mask8x16::from_slice(
        simd,
        &[-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    );
    let c = mask8x16::from_slice(
        simd,
        &[0, -1, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
    );
    let result: mask8x16<_> = mask.select(b, c);
    assert_eq!(
        *result,
        [-1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1]
    );
}

#[simd_test]
fn select_i16x8<S: Simd>(simd: S) {
    let mask = mask16x8::from_slice(simd, &[-1, 0, -1, 0, -1, 0, -1, 0]);
    let b = i16x8::from_slice(simd, &[100, 200, 300, 400, -100, -200, -300, -400]);
    let c = i16x8::from_slice(simd, &[10, 20, 30, 40, -10, -20, -30, -40]);
    assert_eq!(*mask.select(b, c), [100, 20, 300, 40, -100, -20, -300, -40]);
}

#[simd_test]
fn select_u16x8<S: Simd>(simd: S) {
    let mask = mask16x8::from_slice(simd, &[0, -1, 0, -1, 0, -1, 0, -1]);
    let b = u16x8::from_slice(simd, &[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]);
    let c = u16x8::from_slice(simd, &[100, 200, 300, 400, 500, 600, 700, 800]);
    assert_eq!(
        *mask.select(b, c),
        [100, 2000, 300, 4000, 500, 6000, 700, 8000]
    );
}

#[simd_test]
fn select_mask16x8<S: Simd>(simd: S) {
    let mask = mask16x8::from_slice(simd, &[-1, -1, 0, 0, -1, -1, 0, 0]);
    let b = mask16x8::from_slice(simd, &[-1, 0, -1, 0, -1, 0, -1, 0]);
    let c = mask16x8::from_slice(simd, &[0, -1, 0, -1, 0, -1, 0, -1]);
    let result: mask16x8<_> = mask.select(b, c);
    assert_eq!(*result, [-1, 0, 0, -1, -1, 0, 0, -1]);
}

#[simd_test]
fn any_true_mask16x8<S: Simd>(simd: S) {
    let all_zero = mask16x8::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(!simd.any_true_mask16x8(all_zero));

    let all_neg = mask16x8::from_slice(simd, &[-1, -1, -1, -1, -1, -1, -1, -1]);
    assert!(simd.any_true_mask16x8(all_neg));

    let one_neg = mask16x8::from_slice(simd, &[0, 0, 0, -1, 0, 0, 0, 0]);
    assert!(simd.any_true_mask16x8(one_neg));
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
fn any_false_mask16x8<S: Simd>(simd: S) {
    let all_zero = mask16x8::from_slice(simd, &[0, 0, 0, 0, 0, 0, 0, 0]);
    assert!(simd.any_false_mask16x8(all_zero));

    let all_neg = mask16x8::from_slice(simd, &[-1, -1, -1, -1, -1, -1, -1, -1]);
    assert!(!simd.any_false_mask16x8(all_neg));

    let one_pos = mask16x8::from_slice(simd, &[-1, -1, -1, 0, -1, -1, -1, -1]);
    assert!(simd.any_false_mask16x8(one_pos));
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
fn select_i32x4<S: Simd>(simd: S) {
    let mask = mask32x4::from_slice(simd, &[-1, 0, 0, -1]);
    let b = i32x4::from_slice(simd, &[10000, 20000, -30000, -40000]);
    let c = i32x4::from_slice(simd, &[100, 200, -300, -400]);
    assert_eq!(*mask.select(b, c), [10000, 200, -300, -40000]);
}

#[simd_test]
fn select_u32x4<S: Simd>(simd: S) {
    let mask = mask32x4::from_slice(simd, &[0, -1, -1, 0]);
    let b = u32x4::from_slice(simd, &[100000, 200000, 300000, 400000]);
    let c = u32x4::from_slice(simd, &[1000, 2000, 3000, 4000]);
    assert_eq!(*mask.select(b, c), [1000, 200000, 300000, 4000]);
}

#[simd_test]
fn select_mask32x4<S: Simd>(simd: S) {
    let mask = mask32x4::from_slice(simd, &[-1, 0, -1, 0]);
    let b = mask32x4::from_slice(simd, &[-1, -1, 0, 0]);
    let c = mask32x4::from_slice(simd, &[0, 0, -1, -1]);
    let result: mask32x4<_> = mask.select(b, c);
    assert_eq!(*result, [-1, 0, 0, -1]);
}

#[simd_test]
fn any_true_mask32x4<S: Simd>(simd: S) {
    let all_zero = mask32x4::from_slice(simd, &[0, 0, 0, 0]);
    assert!(!simd.any_true_mask32x4(all_zero));

    let all_neg = mask32x4::from_slice(simd, &[-1, -1, -1, -1]);
    assert!(simd.any_true_mask32x4(all_neg));

    let one_neg = mask32x4::from_slice(simd, &[0, -1, 0, 0]);
    assert!(simd.any_true_mask32x4(one_neg));
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
fn any_false_mask32x4<S: Simd>(simd: S) {
    let all_zero = mask32x4::from_slice(simd, &[0, 0, 0, 0]);
    assert!(simd.any_false_mask32x4(all_zero));

    let all_neg = mask32x4::from_slice(simd, &[-1, -1, -1, -1]);
    assert!(!simd.any_false_mask32x4(all_neg));

    let one_pos = mask32x4::from_slice(simd, &[-1, 0, -1, -1]);
    assert!(simd.any_false_mask32x4(one_pos));
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
fn narrow_u16x16<S: Simd>(simd: S) {
    let a = u16x16::from_slice(
        simd,
        &[
            0, 1, 127, 128, 255, 256, 300, 1000, 128, 192, 224, 240, 248, 252, 254, 65535,
        ],
    );
    assert_eq!(
        *simd.narrow_u16x16(a),
        [
            0, 1, 127, 128, 255, 0, 44, 232, 128, 192, 224, 240, 248, 252, 254, 255
        ]
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
fn narrow_u16x32<S: Simd>(simd: S) {
    let a = u16x32::from_slice(
        simd,
        &[
            0, 1, 127, 128, 255, 256, 300, 1000, 128, 192, 224, 240, 248, 252, 254, 255, 100, 200,
            255, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65535, 0, 1, 2, 3,
        ],
    );
    assert_eq!(
        *simd.narrow_u16x32(a),
        [
            0, 1, 127, 128, 255, 0, 44, 232, 128, 192, 224, 240, 248, 252, 254, 255, 100, 200, 255,
            0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 1, 2, 3
        ]
    );
}

#[simd_test]
fn abs_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[-1.5, 2.5]);
    assert_eq!(*a.abs(), [1.5, 2.5]);
}

#[simd_test]
fn neg_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.5, -2.5]);
    assert_eq!(*(-a), [-1.5, 2.5]);
}

#[simd_test]
fn neg_i32x4<S: Simd>(simd: S) {
    assert_eq!(*(-i32x4::from_slice(simd, &[1, -2, 3, -4])), [-1, 2, -3, 4]);
}

#[simd_test]
fn sqrt_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[4.0, 9.0]);
    assert_eq!(*a.sqrt(), [2.0, 3.0]);
}

#[simd_test]
fn copysign_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.5, -2.5]);
    let b = f64x2::from_slice(simd, &[-1.0, 1.0]);
    assert_eq!(*a.copysign(b), [-1.5, 2.5]);
}

#[simd_test]
fn msub_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[2.0, 3.0]);
    let b = f64x2::from_slice(simd, &[4.0, 5.0]);
    let c = f64x2::from_slice(simd, &[1.0, 2.0]);
    assert_eq!(*a.mul_sub(b, c), [7.0, 13.0]);
}

#[simd_test]
fn madd_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.0, 2.0]);
    let b = f64x2::from_slice(simd, &[4.0, 5.0]);
    let c = f64x2::from_slice(simd, &[2.0, 3.0]);
    assert_eq!(*a.mul_add(b, c), [6.0, 13.0]);
}

#[simd_test]
fn floor_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.7, -2.3]);
    assert_eq!(*a.floor(), [1.0, -3.0]);
}

#[simd_test]
fn ceil_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[2.1, -3.2]);
    assert_eq!(*a.ceil(), [3.0, -3.0]);
    let b = f64x2::from_slice(simd, &[-2.7, 0.5]);
    assert_eq!(*b.ceil(), [-2.0, 1.0]);
}

#[simd_test]
fn round_ties_even_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[2.3, -3.2]);
    assert_eq!(*a.round_ties_even(), [2.0, -3.0]);
    let b = f64x2::from_slice(simd, &[2.7, -3.6]);
    assert_eq!(*b.round_ties_even(), [3.0, -4.0]);
    let c = f64x2::from_slice(simd, &[-3.5, -2.5]);
    assert_eq!(*c.round_ties_even(), [-4.0, -2.0]);
    let d = f64x2::from_slice(simd, &[1.5, 0.5]);
    assert_eq!(*d.round_ties_even(), [2.0, 0.0]);
}

#[simd_test]
fn fract_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.7, -2.3]);
    assert_eq!(*a.fract(), [0.7, -0.2999999999999998]);
}

#[simd_test]
fn trunc_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.7, -2.3]);
    assert_eq!(*a.trunc(), [1.0, -2.0]);
}

#[simd_test]
fn any_true_mask64x2<S: Simd>(simd: S) {
    let all_zero = mask64x2::from_slice(simd, &[0, 0]);
    assert!(!simd.any_true_mask64x2(all_zero));

    let all_neg = mask64x2::from_slice(simd, &[-1, -1]);
    assert!(simd.any_true_mask64x2(all_neg));

    let one_neg = mask64x2::from_slice(simd, &[-1, 0]);
    assert!(simd.any_true_mask64x2(one_neg));
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
fn any_false_mask64x2<S: Simd>(simd: S) {
    let all_zero = mask64x2::from_slice(simd, &[0, 0]);
    assert!(simd.any_false_mask64x2(all_zero));

    let all_neg = mask64x2::from_slice(simd, &[-1, -1]);
    assert!(!simd.any_false_mask64x2(all_neg));

    let one_pos = mask64x2::from_slice(simd, &[-1, 0]);
    assert!(simd.any_false_mask64x2(one_pos));
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
fn mul_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 100],
    );
    let b = u8x16::from_slice(
        simd,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2],
    );

    assert_eq!(
        *(a * b),
        [
            0, 2, 6, 12, 20, 30, 70, 120, 180, 250, 74, 164, 8, 188, 132, 200
        ]
    );
}

#[simd_test]
fn mul_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(
        simd,
        &[
            0, 1, -2, 3, -4, 5, 10, -15, 20, -25, 30, 35, -40, 50, -60, 100,
        ],
    );
    let b = i8x16::from_slice(
        simd,
        &[1, 2, 3, -4, 5, -6, 7, 8, 9, 10, -11, 12, 13, -14, 15, 2],
    );

    assert_eq!(
        *(a * b),
        [
            0, 2, -6, -12, -20, -30, 70, -120, -76, 6, -74, -92, -8, 68, 124, -56
        ]
    );
}

#[simd_test]
fn mul_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[0, 5, 10, 30, 500, 0, 0, 0]);
    let b = u16x8::from_slice(simd, &[5, 8, 13, 21, 230, 0, 0, 0]);

    assert_eq!(*(a * b), [0, 40, 130, 630, 49464, 0, 0, 0]);
}

#[simd_test]
fn mul_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 5464, 23234, 456456]);
    let b = u32x4::from_slice(simd, &[23, 34, 565, 34234]);

    assert_eq!(*(a * b), [23, 185776, 13127210, 2741412816]);
}

#[simd_test]
fn mul_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[-10.3, 0.0, 13.34, 234234.0]);
    let b = f32x4::from_slice(simd, &[-8.1, 7.9, -9.8, 3243.6]);

    assert_eq!(*(a * b), [83.43001, 0.0, -130.73201, 759761400.0]);
}

#[simd_test]
fn simd_eq_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(
        simd,
        &[1, 2, 128, 200, 255, 6, 7, 8, 1, 2, 128, 200, 255, 6, 7, 8],
    );
    let b = u8x16::from_slice(
        simd,
        &[1, 0, 128, 0, 255, 0, 7, 0, 1, 0, 128, 0, 255, 0, 7, 0],
    );
    assert_eq!(
        *a.simd_eq(b),
        [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0]
    );
}

#[simd_test]
fn simd_ge_u8x16<S: Simd>(simd: S) {
    let vals = u8x16::from_slice(
        simd,
        &[
            0, 12, 34, 50, 220, 180, 127, 128, 255, 50, 33, 126, 0, 0, 0, 0,
        ],
    );
    let mask = vals.simd_ge(u8x16::splat(simd, 128));

    assert_eq!(*mask, [0, 0, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0]);
}

#[simd_test]
fn simd_gt_u8x16<S: Simd>(simd: S) {
    let vals = u8x16::from_slice(
        simd,
        &[
            0, 12, 34, 50, 220, 180, 127, 128, 255, 50, 33, 126, 0, 0, 0, 0,
        ],
    );
    let mask = vals.simd_gt(u8x16::splat(simd, 128));

    assert_eq!(*mask, [0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]);
}

#[simd_test]
fn simd_le_u8x16<S: Simd>(simd: S) {
    let vals = u8x16::from_slice(
        simd,
        &[
            0, 12, 34, 50, 220, 180, 127, 128, 255, 50, 33, 126, 0, 0, 0, 0,
        ],
    );
    let mask = vals.simd_le(u8x16::splat(simd, 128));

    assert_eq!(
        *mask,
        [-1, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1]
    );
}

#[simd_test]
fn simd_lt_u8x16<S: Simd>(simd: S) {
    let vals = u8x16::from_slice(
        simd,
        &[
            0, 12, 34, 50, 220, 180, 127, 128, 255, 50, 33, 126, 0, 0, 0, 0,
        ],
    );
    let mask = vals.simd_lt(u8x16::splat(simd, 128));

    assert_eq!(
        *mask,
        [-1, -1, -1, -1, 0, 0, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    );
}

#[simd_test]
fn simd_ge_i8x16<S: Simd>(simd: S) {
    let vals = i8x16::from_slice(
        simd,
        &[0, -45, -12, 34, 89, 122, -122, 13, -1, 0, 0, 0, 0, 0, 0, 0],
    );
    let mask = vals.simd_ge(i8x16::splat(simd, -1));

    assert_eq!(
        *mask,
        [-1, 0, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    );
}

#[simd_test]
fn select_native_width_vectors<S: Simd>(simd: S) {
    // Test with native f32 vectors
    let a_f32 = S::f32s::from_slice(simd, &vec![1.0_f32; S::f32s::N]);
    let b_f32 = S::f32s::from_slice(simd, &vec![2.0_f32; S::f32s::N]);
    let mask_f32 = S::mask32s::from_slice(simd, &vec![-1_i32; S::mask32s::N]);
    let result_f32 = mask_f32.select(a_f32, b_f32);
    assert_eq!(result_f32.as_slice(), vec![1.0_f32; S::f32s::N]);

    // Test with native u32 vectors
    let a_u32 = S::u32s::from_slice(simd, &vec![10_u32; S::u32s::N]);
    let b_u32 = S::u32s::from_slice(simd, &vec![20_u32; S::u32s::N]);
    let result_u32 = mask_f32.select(a_u32, b_u32);
    assert_eq!(result_u32.as_slice(), vec![10_u32; S::u32s::N]);

    // Test with native i32 vectors
    let a_i32 = S::i32s::from_slice(simd, &vec![100_i32; S::i32s::N]);
    let b_i32 = S::i32s::from_slice(simd, &vec![-100_i32; S::i32s::N]);
    let result_i32 = mask_f32.select(a_i32, b_i32);
    assert_eq!(result_i32.as_slice(), vec![100_i32; S::i32s::N]);

    // Test with native u8 vectors
    let a_u8 = S::u8s::from_slice(simd, &vec![1_u8; S::u8s::N]);
    let b_u8 = S::u8s::from_slice(simd, &vec![2_u8; S::u8s::N]);
    let mask_u8 = S::mask8s::from_slice(simd, &vec![0_i8; S::u8s::N]);
    let result_u8 = mask_u8.select(a_u8, b_u8);
    assert_eq!(result_u8.as_slice(), vec![2_u8; S::u8s::N]);

    // Test with native i8 vectors
    let a_i8 = S::i8s::from_slice(simd, &vec![10_i8; S::i8s::N]);
    let b_i8 = S::i8s::from_slice(simd, &vec![-10_i8; S::i8s::N]);
    let result_i8 = mask_u8.select(a_i8, b_i8);
    assert_eq!(result_i8.as_slice(), vec![-10_i8; S::i8s::N]);

    // Test with native u16 vectors
    let a_u16 = S::u16s::from_slice(simd, &vec![100_u16; S::u16s::N]);
    let b_u16 = S::u16s::from_slice(simd, &vec![200_u16; S::u16s::N]);
    let mask_u16 = S::mask16s::from_slice(simd, &vec![-1_i16; S::mask16s::N]);
    let result_u16 = mask_u16.select(a_u16, b_u16);
    assert_eq!(result_u16.as_slice(), vec![100_u16; S::u16s::N]);

    // Test with native i16 vectors
    let a_i16 = S::i16s::from_slice(simd, &vec![50_i16; S::i16s::N]);
    let b_i16 = S::i16s::from_slice(simd, &vec![-50_i16; S::i16s::N]);
    let result_i16 = mask_u16.select(a_i16, b_i16);
    assert_eq!(result_i16.as_slice(), vec![50_i16; S::i16s::N]);
}

#[simd_test]
fn bitcast_native<S: Simd>(simd: S) {
    let a_i32 = S::i32s::from_slice(simd, &vec![-1; S::i32s::N]);
    assert_eq!(
        a_i32.bitcast::<S::u32s>().as_slice(),
        &vec![u32::MAX; S::i32s::N]
    );
}

#[simd_test]
fn wrapping_add_u32<S: Simd>(simd: S) {
    assert_eq!(
        (S::u32s::splat(simd, u32::MAX) + 1).as_slice(),
        &vec![0; S::u32s::N]
    );
}

#[simd_test]
fn index_consistency<S: Simd>(simd: S) {
    // We'll call index methods by name to avoid confusing clippy.
    use std::ops::{Index, IndexMut};

    let mut v = u32x4::from_slice(simd, &[0, 1, 2, 3]);
    for i in 0..4 {
        assert_eq!(i, *v.index(i) as usize);
        assert_eq!(i, *v.index_mut(i) as usize);
    }
}
#[simd_test]
fn store_array_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let mut dest = [0.0_f32; 4];
    simd.store_array_f32x4(a, &mut dest);
    assert_eq!(dest, [1.0, 2.0, 3.0, 4.0]);
}

#[simd_test]
fn store_array_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let mut dest = [0.0_f32; 8];
    simd.store_array_f32x8(a, &mut dest);
    assert_eq!(dest, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[simd_test]
fn store_array_f64x2<S: Simd>(simd: S) {
    let a = f64x2::from_slice(simd, &[1.5, 2.5]);
    let mut dest = [0.0_f64; 2];
    simd.store_array_f64x2(a, &mut dest);
    assert_eq!(dest, [1.5, 2.5]);
}

#[simd_test]
fn store_array_f64x4<S: Simd>(simd: S) {
    let a = f64x4::from_slice(simd, &[1.5, 2.5, 3.5, 4.5]);
    let mut dest = [0.0_f64; 4];
    simd.store_array_f64x4(a, &mut dest);
    assert_eq!(dest, [1.5, 2.5, 3.5, 4.5]);
}

#[simd_test]
fn store_array_i8x16<S: Simd>(simd: S) {
    let a = i8x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let mut dest = [0_i8; 16];
    simd.store_array_i8x16(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
}

#[simd_test]
fn store_array_i16x8<S: Simd>(simd: S) {
    let a = i16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut dest = [0_i16; 8];
    simd.store_array_i16x8(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn store_array_i32x4<S: Simd>(simd: S) {
    let a = i32x4::from_slice(simd, &[1, 2, 3, 4]);
    let mut dest = [0_i32; 4];
    simd.store_array_i32x4(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4]);
}

#[simd_test]
fn store_array_u8x16<S: Simd>(simd: S) {
    let a = u8x16::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let mut dest = [0_u8; 16];
    simd.store_array_u8x16(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
}

#[simd_test]
fn store_array_u16x8<S: Simd>(simd: S) {
    let a = u16x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut dest = [0_u16; 8];
    simd.store_array_u16x8(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn store_array_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[1, 2, 3, 4]);
    let mut dest = [0_u32; 4];
    simd.store_array_u32x4(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4]);
}

#[simd_test]
fn store_array_i32x8<S: Simd>(simd: S) {
    let a = i32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut dest = [0_i32; 8];
    simd.store_array_i32x8(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn store_array_u32x8<S: Simd>(simd: S) {
    let a = u32x8::from_slice(simd, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut dest = [0_u32; 8];
    simd.store_array_u32x8(a, &mut dest);
    assert_eq!(dest, [1, 2, 3, 4, 5, 6, 7, 8]);
}

#[simd_test]
fn store_array_f32x16<S: Simd>(simd: S) {
    let data: [f32; 16] = core::array::from_fn(|i| (i + 1) as f32);
    let a = f32x16::from_slice(simd, &data);
    let mut dest = [0.0_f32; 16];
    simd.store_array_f32x16(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_f64x8<S: Simd>(simd: S) {
    let data: [f64; 8] = core::array::from_fn(|i| (i + 1) as f64);
    let a = f64x8::from_slice(simd, &data);
    let mut dest = [0.0_f64; 8];
    simd.store_array_f64x8(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_i8x32<S: Simd>(simd: S) {
    let data: [i8; 32] = core::array::from_fn(|i| (i + 1) as i8);
    let a = i8x32::from_slice(simd, &data);
    let mut dest = [0_i8; 32];
    simd.store_array_i8x32(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_i8x64<S: Simd>(simd: S) {
    let data: [i8; 64] = core::array::from_fn(|i| (i + 1) as i8);
    let a = i8x64::from_slice(simd, &data);
    let mut dest = [0_i8; 64];
    simd.store_array_i8x64(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_i16x16<S: Simd>(simd: S) {
    let data: [i16; 16] = core::array::from_fn(|i| (i + 1) as i16);
    let a = i16x16::from_slice(simd, &data);
    let mut dest = [0_i16; 16];
    simd.store_array_i16x16(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_i16x32<S: Simd>(simd: S) {
    let data: [i16; 32] = core::array::from_fn(|i| (i + 1) as i16);
    let a = i16x32::from_slice(simd, &data);
    let mut dest = [0_i16; 32];
    simd.store_array_i16x32(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_i32x16<S: Simd>(simd: S) {
    let data: [i32; 16] = core::array::from_fn(|i| (i + 1) as i32);
    let a = i32x16::from_slice(simd, &data);
    let mut dest = [0_i32; 16];
    simd.store_array_i32x16(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_u8x32<S: Simd>(simd: S) {
    let data: [u8; 32] = core::array::from_fn(|i| (i + 1) as u8);
    let a = u8x32::from_slice(simd, &data);
    let mut dest = [0_u8; 32];
    simd.store_array_u8x32(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_u8x64<S: Simd>(simd: S) {
    let data: [u8; 64] = core::array::from_fn(|i| (i + 1) as u8);
    let a = u8x64::from_slice(simd, &data);
    let mut dest = [0_u8; 64];
    simd.store_array_u8x64(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_u16x16<S: Simd>(simd: S) {
    let data: [u16; 16] = core::array::from_fn(|i| (i + 1) as u16);
    let a = u16x16::from_slice(simd, &data);
    let mut dest = [0_u16; 16];
    simd.store_array_u16x16(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_u16x32<S: Simd>(simd: S) {
    let data: [u16; 32] = core::array::from_fn(|i| (i + 1) as u16);
    let a = u16x32::from_slice(simd, &data);
    let mut dest = [0_u16; 32];
    simd.store_array_u16x32(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_array_u32x16<S: Simd>(simd: S) {
    let data: [u32; 16] = core::array::from_fn(|i| (i + 1) as u32);
    let a = u32x16::from_slice(simd, &data);
    let mut dest = [0_u32; 16];
    simd.store_array_u32x16(a, &mut dest);
    assert_eq!(dest, data);
}

#[simd_test]
fn store_slice_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0]);
    let mut dest = [0.0_f32; 4];
    a.store_slice(&mut dest);
    assert_eq!(dest, [1.0, 2.0, 3.0, 4.0]);
}