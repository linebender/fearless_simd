// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn cvt_u32_f32x4_regular<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[101.75, 22.5, 33.875, 404.25]);
    assert_eq!(*a.to_int::<u32x4<_>>(), [101, 22, 33, 404]);
}

#[simd_test]
fn cvt_u32_f32x8_regular<S: Simd>(simd: S) {
    let a = f32x8::from_slice(
        simd,
        &[
            801.75, 702.5, 603.25, 504.875, 405.125, 306.5, 207.75, 108.25,
        ],
    );
    assert_eq!(
        *a.to_int::<u32x8<_>>(),
        [801, 702, 603, 504, 405, 306, 207, 108]
    );
}

#[simd_test]
fn cvt_u32_f32x16_regular<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1601.75, 1502.5, 1403.25, 1304.875, 1205.125, 1106.5, 1007.75, 908.25, 809.5, 710.75,
            611.125, 512.875, 413.25, 314.5, 215.75, 116.25,
        ],
    );
    assert_eq!(
        *a.to_int::<u32x16<_>>(),
        [
            1601, 1502, 1403, 1304, 1205, 1106, 1007, 908, 809, 710, 611, 512, 413, 314, 215, 116,
        ]
    );
}

#[simd_test]
fn cvt_u32_f32x4<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[1.0, 42.7, 3e9, -0.3]);
    assert_eq!(*a.to_int::<u32x4<_>>(), [1, 42, 3000000000, 0]);
}

#[simd_test]
fn cvt_u32_f32x4_rounding<S: Simd>(simd: S) {
    let a = f32x4::from_slice(simd, &[0.0, 0.49, 0.51, 0.99]);
    assert_eq!(*a.to_int::<u32x4<_>>(), [0, 0, 0, 0]);
    let a = f32x4::from_slice(simd, &[1.01, 1.99, 2.5, 3.75]);
    assert_eq!(*a.to_int::<u32x4<_>>(), [1, 1, 2, 3]);
}

#[simd_test]
fn cvt_u32_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 42.7, 3e9, -0.3, 0.0, 17.9, 255.99, 1024.1]);
    assert_eq!(
        *a.to_int::<u32x8<_>>(),
        [1, 42, 3000000000, 0, 0, 17, 255, 1024]
    );
}

#[simd_test]
fn cvt_u32_f32x8_rounding<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[0.0, 0.49, 0.51, 0.99, 1.01, 1.99, 2.5, 3.75]);
    assert_eq!(*a.to_int::<u32x8<_>>(), [0, 0, 0, 0, 1, 1, 2, 3]);
}

#[simd_test]
fn cvt_u32_f32x16_rounding<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            0.0, 0.49, 0.51, 0.99, 1.01, 1.99, 2.5, 3.75, 0.0, 0.49, 0.51, 0.99, 1.01, 1.99, 2.5,
            3.75,
        ],
    );
    assert_eq!(
        *a.to_int::<u32x16<_>>(),
        [0, 0, 0, 0, 1, 1, 2, 3, 0, 0, 0, 0, 1, 1, 2, 3]
    );
}

#[simd_test]
fn cvt_u32_f32x16<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtTruncate;
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
