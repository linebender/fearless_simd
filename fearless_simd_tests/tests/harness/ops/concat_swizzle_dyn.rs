// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn concat_swizzle_dyn_u8x16<S: Simd>(simd: S) {
    let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let b = [
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    ];
    let indices = [15, 16, 31, 0, 1, 14, 17, 30, 7, 8, 23, 24, 2, 13, 18, 29];
    let expected = [16, 17, 32, 1, 2, 15, 18, 31, 8, 9, 24, 25, 3, 14, 19, 30];

    let a = u8x16::simd_from(simd, a);
    let b = u8x16::simd_from(simd, b);
    let indices = u8x16::simd_from(simd, indices);

    assert_eq!(*a.concat_swizzle_dyn(b, indices), expected);
}

#[simd_test]
fn concat_swizzle_dyn_u8x32_generic<S: Simd>(simd: S) {
    #[inline(always)]
    fn do_concat_swizzle<S: Simd, V: SimdBase<S>>(a: V, b: V, indices: V::Bytes) -> V {
        a.concat_swizzle_dyn(b, indices)
    }

    let a = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    ];
    let b = [
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64,
    ];
    let indices = [
        0, 15, 16, 31, 32, 47, 48, 63, 1, 14, 17, 30, 33, 46, 49, 62, 7, 8, 23, 24, 39, 40, 55, 56,
        2, 13, 18, 29, 34, 45, 50, 61,
    ];
    let expected = [
        1, 16, 17, 32, 33, 48, 49, 64, 2, 15, 18, 31, 34, 47, 50, 63, 8, 9, 24, 25, 40, 41, 56, 57,
        3, 14, 19, 30, 35, 46, 51, 62,
    ];

    let a = u8x32::simd_from(simd, a);
    let b = u8x32::simd_from(simd, b);
    let indices = u8x32::simd_from(simd, indices);
    let result = do_concat_swizzle::<S, u8x32<S>>(a, b, indices);

    assert_eq!(*result, expected);
}

#[simd_test]
fn concat_swizzle_dyn_u8x64<S: Simd>(simd: S) {
    let a = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    ];
    let b = [
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
        88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        126, 127, 128,
    ];
    let indices = [
        0, 15, 16, 31, 32, 47, 48, 63, 64, 79, 80, 95, 96, 111, 112, 127, 1, 14, 17, 30, 33, 46,
        49, 62, 65, 78, 81, 94, 97, 110, 113, 126, 7, 8, 23, 24, 39, 40, 55, 56, 71, 72, 87, 88,
        103, 104, 119, 120, 2, 13, 18, 29, 34, 45, 50, 61, 66, 77, 82, 93, 98, 109, 114, 125,
    ];
    let expected = [
        1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113, 128, 2, 15, 18, 31, 34, 47,
        50, 63, 66, 79, 82, 95, 98, 111, 114, 127, 8, 9, 24, 25, 40, 41, 56, 57, 72, 73, 88, 89,
        104, 105, 120, 121, 3, 14, 19, 30, 35, 46, 51, 62, 67, 78, 83, 94, 99, 110, 115, 126,
    ];

    let a = u8x64::simd_from(simd, a);
    let b = u8x64::simd_from(simd, b);
    let indices = u8x64::simd_from(simd, indices);

    assert_eq!(*a.concat_swizzle_dyn(b, indices), expected);
}

#[simd_test]
fn concat_swizzle_dyn_f32x8_bytes<S: Simd>(simd: S) {
    let a_bytes = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    ];
    let b_bytes = [
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64,
    ];
    let indices = [
        63, 48, 47, 32, 31, 16, 15, 0, 1, 14, 17, 30, 33, 46, 49, 62, 7, 8, 23, 24, 39, 40, 55, 56,
        2, 13, 18, 29, 34, 45, 50, 61,
    ];
    let expected = [
        64, 49, 48, 33, 32, 17, 16, 1, 2, 15, 18, 31, 34, 47, 50, 63, 8, 9, 24, 25, 40, 41, 56, 57,
        3, 14, 19, 30, 35, 46, 51, 62,
    ];

    let a: f32x8<S> = u8x32::simd_from(simd, a_bytes).bitcast();
    let b: f32x8<S> = u8x32::simd_from(simd, b_bytes).bitcast();
    let indices = u8x32::simd_from(simd, indices);
    let result: u8x32<S> = a.concat_swizzle_dyn(b, indices).bitcast();

    assert_eq!(*result, expected);
}
