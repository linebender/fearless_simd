// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

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
        <[i8; 16]>::from(result),
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
    assert_eq!(<[i16; 8]>::from(result), [-1, 0, 0, -1, -1, 0, 0, -1]);
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
    assert_eq!(<[i32; 4]>::from(result), [-1, 0, 0, -1]);
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
fn select_f32x8<S: Simd>(simd: S) {
    let a = f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::from_slice(simd, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let mask = mask32x8::from_slice(simd, &[-1, 0, -1, 0, 0, -1, 0, -1]);
    assert_eq!(
        *mask.select(a, b),
        [1.0, 20.0, 3.0, 40.0, 50.0, 6.0, 70.0, 8.0]
    );
}

#[simd_test]
fn select_f32x16<S: Simd>(simd: S) {
    let a = f32x16::from_slice(
        simd,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let b = f32x16::from_slice(
        simd,
        &[
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0,
            140.0, 150.0, 160.0,
        ],
    );
    let mask = mask32x16::from_slice(
        simd,
        &[-1, 0, -1, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0, -1, 0, -1],
    );
    assert_eq!(
        *mask.select(a, b),
        [
            1.0, 20.0, 3.0, 40.0, 50.0, 6.0, 70.0, 8.0, 9.0, 100.0, 11.0, 120.0, 130.0, 14.0,
            150.0, 16.0
        ]
    );
}

#[simd_test]
fn select_f64x8<S: Simd>(simd: S) {
    let a = f64x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f64x8::from_slice(simd, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let mask = mask64x8::from_slice(simd, &[-1, 0, -1, 0, 0, -1, 0, -1]);
    assert_eq!(
        *mask.select(a, b),
        [1.0, 20.0, 3.0, 40.0, 50.0, 6.0, 70.0, 8.0]
    );
}

#[simd_test]
fn select_i64x2<S: Simd>(simd: S) {
    let mask = mask64x2::from_slice(simd, &[-1, 0]);
    let a = i64x2::from_slice(simd, &[-9, 18]);
    let b = i64x2::from_slice(simd, &[3, -6]);
    assert_eq!(*mask.select(a, b), [-9, -6]);
}

#[simd_test]
fn select_i64x4<S: Simd>(simd: S) {
    let mask = mask64x4::from_slice(simd, &[-1, 0, -1, 0]);
    let a = i64x4::from_slice(simd, &[-9, 18, i64::MAX - 7, i64::MIN + 9]);
    let b = i64x4::from_slice(simd, &[3, -6, 5, -7]);
    assert_eq!(*mask.select(a, b), [-9, -6, 9223372036854775800, -7]);
}

#[simd_test]
fn select_i64x8<S: Simd>(simd: S) {
    let mask = mask64x8::from_slice(simd, &[-1, 0, -1, 0, -1, 0, -1, 0]);
    let a = i64x8::from_slice(
        simd,
        &[-9, 18, i64::MAX - 7, i64::MIN + 9, 123, -456, 789, -1024],
    );
    let b = i64x8::from_slice(simd, &[3, -6, 5, -7, -11, 13, -17, 19]);
    assert_eq!(
        *mask.select(a, b),
        [-9, -6, 9223372036854775800, -7, 123, 13, 789, 19]
    );
}

#[simd_test]
fn select_u64x2<S: Simd>(simd: S) {
    let mask = mask64x2::from_slice(simd, &[-1, 0]);
    let a = u64x2::from_slice(simd, &[0, 1_u64 << 63]);
    let b = u64x2::from_slice(simd, &[u64::MAX, 7]);
    assert_eq!(*mask.select(a, b), [0, 7]);
}

#[simd_test]
fn select_u64x4<S: Simd>(simd: S) {
    let mask = mask64x4::from_slice(simd, &[-1, 0, -1, 0]);
    let a = u64x4::from_slice(simd, &[0, 1_u64 << 63, u64::MAX - 3, 42]);
    let b = u64x4::from_slice(simd, &[u64::MAX, 7, 13, 999]);
    assert_eq!(*mask.select(a, b), [0, 7, u64::MAX - 3, 999]);
}

#[simd_test]
fn select_u64x8<S: Simd>(simd: S) {
    let mask = mask64x8::from_slice(simd, &[-1, 0, -1, 0, -1, 0, -1, 0]);
    let a = u64x8::from_slice(
        simd,
        &[
            0,
            1_u64 << 63,
            u64::MAX - 3,
            42,
            17,
            99,
            123456789,
            u64::MAX,
        ],
    );
    let b = u64x8::from_slice(simd, &[u64::MAX, 7, 13, 999, 29, 11, 987654321, 1]);
    assert_eq!(
        *mask.select(a, b),
        [0, 7, u64::MAX - 3, 999, 17, 11, 123456789, 1]
    );
}

// Additional concrete rows for this operation.

#[simd_test]
fn select_mask64x2<S: Simd>(simd: S) {
    let mask = mask64x2::from_slice(simd, &[-1_i64, 0_i64]);
    let t = simd.splat_mask64x2(true);
    let f = simd.splat_mask64x2(false);
    assert_eq!(
        <[i64; 2]>::from(simd.select_mask64x2(mask, t, f)),
        [-1_i64, 0_i64]
    );
}

#[simd_test]
fn select_mask64x4<S: Simd>(simd: S) {
    let mask = mask64x4::from_slice(simd, &[-1_i64, 0_i64, -1_i64, 0_i64]);
    let t = simd.splat_mask64x4(true);
    let f = simd.splat_mask64x4(false);
    assert_eq!(
        <[i64; 4]>::from(simd.select_mask64x4(mask, t, f)),
        [-1_i64, 0_i64, -1_i64, 0_i64]
    );
}

#[simd_test]
fn select_mask64x8<S: Simd>(simd: S) {
    let mask = mask64x8::from_slice(
        simd,
        &[-1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64],
    );
    let t = simd.splat_mask64x8(true);
    let f = simd.splat_mask64x8(false);
    assert_eq!(
        <[i64; 8]>::from(simd.select_mask64x8(mask, t, f)),
        [-1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64, -1_i64, 0_i64]
    );
}

// Generated gap-fill coverage rows.

#[simd_test]
fn select_i8x32<S: Simd>(simd: S) {
    let mask_values: [i8; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let true_values: [i8; 32] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let false_values: [i8; 32] = core::array::from_fn(|i| (i % 7) as i8 + 1_i8);
    let mask = mask8x32::from_slice(simd, &mask_values);
    let if_true = i8x32::from_slice(simd, &true_values);
    let if_false = i8x32::from_slice(simd, &false_values);
    let expected: [i8; 32] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_i8x32(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_u8x32<S: Simd>(simd: S) {
    let mask_values: [i8; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let true_values: [u8; 32] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let false_values: [u8; 32] = core::array::from_fn(|i| (i % 7) as u8 + 1_u8);
    let mask = mask8x32::from_slice(simd, &mask_values);
    let if_true = u8x32::from_slice(simd, &true_values);
    let if_false = u8x32::from_slice(simd, &false_values);
    let expected: [u8; 32] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_u8x32(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_i8x64<S: Simd>(simd: S) {
    let mask_values: [i8; 64] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let true_values: [i8; 64] = core::array::from_fn(|i| (i % 23) as i8 + 10_i8);
    let false_values: [i8; 64] = core::array::from_fn(|i| (i % 7) as i8 + 1_i8);
    let mask = mask8x64::from_slice(simd, &mask_values);
    let if_true = i8x64::from_slice(simd, &true_values);
    let if_false = i8x64::from_slice(simd, &false_values);
    let expected: [i8; 64] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_i8x64(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_u8x64<S: Simd>(simd: S) {
    let mask_values: [i8; 64] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let true_values: [u8; 64] = core::array::from_fn(|i| (i % 23) as u8 + 10_u8);
    let false_values: [u8; 64] = core::array::from_fn(|i| (i % 7) as u8 + 1_u8);
    let mask = mask8x64::from_slice(simd, &mask_values);
    let if_true = u8x64::from_slice(simd, &true_values);
    let if_false = u8x64::from_slice(simd, &false_values);
    let expected: [u8; 64] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_u8x64(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_i16x16<S: Simd>(simd: S) {
    let mask_values: [i16; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let true_values: [i16; 16] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let false_values: [i16; 16] = core::array::from_fn(|i| (i % 7) as i16 + 1_i16);
    let mask = mask16x16::from_slice(simd, &mask_values);
    let if_true = i16x16::from_slice(simd, &true_values);
    let if_false = i16x16::from_slice(simd, &false_values);
    let expected: [i16; 16] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_i16x16(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_u16x16<S: Simd>(simd: S) {
    let mask_values: [i16; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let true_values: [u16; 16] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let false_values: [u16; 16] = core::array::from_fn(|i| (i % 7) as u16 + 1_u16);
    let mask = mask16x16::from_slice(simd, &mask_values);
    let if_true = u16x16::from_slice(simd, &true_values);
    let if_false = u16x16::from_slice(simd, &false_values);
    let expected: [u16; 16] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_u16x16(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_i16x32<S: Simd>(simd: S) {
    let mask_values: [i16; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let true_values: [i16; 32] = core::array::from_fn(|i| (i % 23) as i16 + 10_i16);
    let false_values: [i16; 32] = core::array::from_fn(|i| (i % 7) as i16 + 1_i16);
    let mask = mask16x32::from_slice(simd, &mask_values);
    let if_true = i16x32::from_slice(simd, &true_values);
    let if_false = i16x32::from_slice(simd, &false_values);
    let expected: [i16; 32] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_i16x32(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_u16x32<S: Simd>(simd: S) {
    let mask_values: [i16; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let true_values: [u16; 32] = core::array::from_fn(|i| (i % 23) as u16 + 10_u16);
    let false_values: [u16; 32] = core::array::from_fn(|i| (i % 7) as u16 + 1_u16);
    let mask = mask16x32::from_slice(simd, &mask_values);
    let if_true = u16x32::from_slice(simd, &true_values);
    let if_false = u16x32::from_slice(simd, &false_values);
    let expected: [u16; 32] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_u16x32(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_i32x8<S: Simd>(simd: S) {
    let mask_values: [i32; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let true_values: [i32; 8] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let false_values: [i32; 8] = core::array::from_fn(|i| (i % 7) as i32 + 1_i32);
    let mask = mask32x8::from_slice(simd, &mask_values);
    let if_true = i32x8::from_slice(simd, &true_values);
    let if_false = i32x8::from_slice(simd, &false_values);
    let expected: [i32; 8] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_i32x8(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_u32x8<S: Simd>(simd: S) {
    let mask_values: [i32; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let true_values: [u32; 8] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let false_values: [u32; 8] = core::array::from_fn(|i| (i % 7) as u32 + 1_u32);
    let mask = mask32x8::from_slice(simd, &mask_values);
    let if_true = u32x8::from_slice(simd, &true_values);
    let if_false = u32x8::from_slice(simd, &false_values);
    let expected: [u32; 8] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_u32x8(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_i32x16<S: Simd>(simd: S) {
    let mask_values: [i32; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let true_values: [i32; 16] = core::array::from_fn(|i| (i % 23) as i32 + 10_i32);
    let false_values: [i32; 16] = core::array::from_fn(|i| (i % 7) as i32 + 1_i32);
    let mask = mask32x16::from_slice(simd, &mask_values);
    let if_true = i32x16::from_slice(simd, &true_values);
    let if_false = i32x16::from_slice(simd, &false_values);
    let expected: [i32; 16] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_i32x16(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_u32x16<S: Simd>(simd: S) {
    let mask_values: [i32; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let true_values: [u32; 16] = core::array::from_fn(|i| (i % 23) as u32 + 10_u32);
    let false_values: [u32; 16] = core::array::from_fn(|i| (i % 7) as u32 + 1_u32);
    let mask = mask32x16::from_slice(simd, &mask_values);
    let if_true = u32x16::from_slice(simd, &true_values);
    let if_false = u32x16::from_slice(simd, &false_values);
    let expected: [u32; 16] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_u32x16(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_f64x2<S: Simd>(simd: S) {
    let mask_values: [i64; 2] = core::array::from_fn(|i| if i % 2 == 0 { -1_i64 } else { 0_i64 });
    let true_values: [f64; 2] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let false_values: [f64; 2] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let mask = mask64x2::from_slice(simd, &mask_values);
    let if_true = f64x2::from_slice(simd, &true_values);
    let if_false = f64x2::from_slice(simd, &false_values);
    let expected: [f64; 2] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_f64x2(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_f64x4<S: Simd>(simd: S) {
    let mask_values: [i64; 4] = core::array::from_fn(|i| if i % 2 == 0 { -1_i64 } else { 0_i64 });
    let true_values: [f64; 4] = core::array::from_fn(|i| i as f64 + 1.25_f64);
    let false_values: [f64; 4] = core::array::from_fn(|i| (i % 5) as f64 + 2.5_f64);
    let mask = mask64x4::from_slice(simd, &mask_values);
    let if_true = f64x4::from_slice(simd, &true_values);
    let if_false = f64x4::from_slice(simd, &false_values);
    let expected: [f64; 4] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_f64x4(mask, if_true, if_false);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn select_mask8x32<S: Simd>(simd: S) {
    let mask_values: [i8; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let true_values: [i8; 32] = core::array::from_fn(|i| if i % 3 == 0 { -1_i8 } else { 0_i8 });
    let false_values: [i8; 32] = core::array::from_fn(|i| if i % 3 == 0 { 0_i8 } else { -1_i8 });
    let mask = mask8x32::from_slice(simd, &mask_values);
    let if_true = mask8x32::from_slice(simd, &true_values);
    let if_false = mask8x32::from_slice(simd, &false_values);
    let expected: [i8; 32] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_mask8x32(mask, if_true, if_false);
    assert_eq!(<[i8; 32]>::from(result), expected);
}

#[simd_test]
fn select_mask8x64<S: Simd>(simd: S) {
    let mask_values: [i8; 64] = core::array::from_fn(|i| if i % 2 == 0 { -1_i8 } else { 0_i8 });
    let true_values: [i8; 64] = core::array::from_fn(|i| if i % 3 == 0 { -1_i8 } else { 0_i8 });
    let false_values: [i8; 64] = core::array::from_fn(|i| if i % 3 == 0 { 0_i8 } else { -1_i8 });
    let mask = mask8x64::from_slice(simd, &mask_values);
    let if_true = mask8x64::from_slice(simd, &true_values);
    let if_false = mask8x64::from_slice(simd, &false_values);
    let expected: [i8; 64] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_mask8x64(mask, if_true, if_false);
    assert_eq!(<[i8; 64]>::from(result), expected);
}

#[simd_test]
fn select_mask16x16<S: Simd>(simd: S) {
    let mask_values: [i16; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let true_values: [i16; 16] = core::array::from_fn(|i| if i % 3 == 0 { -1_i16 } else { 0_i16 });
    let false_values: [i16; 16] = core::array::from_fn(|i| if i % 3 == 0 { 0_i16 } else { -1_i16 });
    let mask = mask16x16::from_slice(simd, &mask_values);
    let if_true = mask16x16::from_slice(simd, &true_values);
    let if_false = mask16x16::from_slice(simd, &false_values);
    let expected: [i16; 16] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_mask16x16(mask, if_true, if_false);
    assert_eq!(<[i16; 16]>::from(result), expected);
}

#[simd_test]
fn select_mask16x32<S: Simd>(simd: S) {
    let mask_values: [i16; 32] = core::array::from_fn(|i| if i % 2 == 0 { -1_i16 } else { 0_i16 });
    let true_values: [i16; 32] = core::array::from_fn(|i| if i % 3 == 0 { -1_i16 } else { 0_i16 });
    let false_values: [i16; 32] = core::array::from_fn(|i| if i % 3 == 0 { 0_i16 } else { -1_i16 });
    let mask = mask16x32::from_slice(simd, &mask_values);
    let if_true = mask16x32::from_slice(simd, &true_values);
    let if_false = mask16x32::from_slice(simd, &false_values);
    let expected: [i16; 32] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_mask16x32(mask, if_true, if_false);
    assert_eq!(<[i16; 32]>::from(result), expected);
}

#[simd_test]
fn select_mask32x8<S: Simd>(simd: S) {
    let mask_values: [i32; 8] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let true_values: [i32; 8] = core::array::from_fn(|i| if i % 3 == 0 { -1_i32 } else { 0_i32 });
    let false_values: [i32; 8] = core::array::from_fn(|i| if i % 3 == 0 { 0_i32 } else { -1_i32 });
    let mask = mask32x8::from_slice(simd, &mask_values);
    let if_true = mask32x8::from_slice(simd, &true_values);
    let if_false = mask32x8::from_slice(simd, &false_values);
    let expected: [i32; 8] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_mask32x8(mask, if_true, if_false);
    assert_eq!(<[i32; 8]>::from(result), expected);
}

#[simd_test]
fn select_mask32x16<S: Simd>(simd: S) {
    let mask_values: [i32; 16] = core::array::from_fn(|i| if i % 2 == 0 { -1_i32 } else { 0_i32 });
    let true_values: [i32; 16] = core::array::from_fn(|i| if i % 3 == 0 { -1_i32 } else { 0_i32 });
    let false_values: [i32; 16] = core::array::from_fn(|i| if i % 3 == 0 { 0_i32 } else { -1_i32 });
    let mask = mask32x16::from_slice(simd, &mask_values);
    let if_true = mask32x16::from_slice(simd, &true_values);
    let if_false = mask32x16::from_slice(simd, &false_values);
    let expected: [i32; 16] = core::array::from_fn(|i| {
        if mask_values[i] != 0 {
            true_values[i]
        } else {
            false_values[i]
        }
    });
    let result = simd.select_mask32x16(mask, if_true, if_false);
    assert_eq!(<[i32; 16]>::from(result), expected);
}
