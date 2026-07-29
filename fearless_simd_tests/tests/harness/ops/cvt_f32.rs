// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

#[simd_test]
fn cvt_f32_u32x4<S: Simd>(simd: S) {
    let a = u32x4::from_slice(simd, &[0, 42, 1000000, u32::MAX]);
    assert_eq!(
        *a.to_float::<f32x4<_>>(),
        [0.0, 42.0, 1000000.0, u32::MAX as f32]
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
fn cvt_f32_u32x8<S: Simd>(simd: S) {
    let values = [
        0,
        42,
        1_000_000,
        i32::MAX as u32,
        0x8000_0000,
        0xffff_ff00,
        u32::MAX - 1,
        u32::MAX,
    ];
    let a = u32x8::from_slice(simd, &values);
    assert_eq!(*a.to_float::<f32x8<_>>(), values.map(|x| x as f32));
}

#[simd_test]
fn cvt_f32_i32x16<S: Simd>(simd: S) {
    use fearless_simd::SimdCvtFloat;
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
    use fearless_simd::SimdCvtFloat;
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

// Generated gap-fill coverage rows.

#[simd_test]
fn cvt_f32_i32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = core::array::from_fn(|i| (i % 17) as i32 + 1_i32);
    let a = i32x8::from_slice(simd, &values);
    let expected: [f32; 8] = core::array::from_fn(|i| values[i] as f32);
    let result = simd.cvt_f32_i32x8(a);
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[simd_test]
fn cvt_f32_u32x16_high_low_bit_matrix<S: Simd>(simd: S) {
    const HIGH_BITS: [u32; 16] = [
        0x0000_0000,
        0x0001_0000,
        0x007f_0000,
        0x0080_0000,
        0x00ff_0000,
        0x0100_0000,
        0x3fff_0000,
        0x4000_0000,
        0x7fff_0000,
        0x8000_0000,
        0x8001_0000,
        0xbfff_0000,
        0xc000_0000,
        0xff00_0000,
        0xfffe_0000,
        0xffff_0000,
    ];
    const LOW_BITS: [u32; 16] = [
        0x0000, 0x0001, 0x0002, 0x007f, 0x0080, 0x00ff, 0x0100, 0x3fff, 0x4000, 0x7fff, 0x8000,
        0x8001, 0xff00, 0xff7f, 0xfffe, 0xffff,
    ];

    for high_bits in HIGH_BITS {
        let values = LOW_BITS.map(|low_bits| high_bits | low_bits);
        let expected = values.map(|value| value as f32);
        let result = simd.cvt_f32_u32x16(u32x16::from_slice(simd, &values));

        assert_eq!(
            result.as_slice(),
            expected.as_slice(),
            "high bits {high_bits:#010x}",
        );
    }
}
