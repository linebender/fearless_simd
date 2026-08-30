// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// This operation uses a very specific, portable operation order,
// which guarantees the same output regardless of the platform.
// The large values intersperced with 1.0 are there to catch
// deviations from that order, which result in large deviations
// in output result due to rounding a large float
// that can't increment by 1 due to the very large exponent.

#[simd_test]
fn reduce_sum_f32x4<S: Simd>(simd: S) {
    assert_eq!(
        f32x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0])
            .reduce_sum()
            .to_bits(),
        10.0_f32.to_bits()
    );

    let large = 1.0e30_f32;
    assert_eq!(
        f32x4::from_slice(simd, &[large, 1.0, -large, 1.0])
            .reduce_sum()
            .to_bits(),
        0.0_f32.to_bits()
    );
    assert_eq!(
        f32x4::from_slice(simd, &[-0.0; 4]).reduce_sum().to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        f32x4::from_slice(simd, &[0.0; 4]).reduce_sum().to_bits(),
        0.0_f32.to_bits()
    );
    let nan = f32::from_bits(0x7fc0_1234);
    assert!(
        f32x4::from_slice(simd, &[0.0, 0.0, 0.0, nan])
            .reduce_sum()
            .is_nan()
    );
}

#[simd_test]
fn reduce_sum_i8x16<S: Simd>(simd: S) {
    assert_eq!(i8x16::from_slice(simd, &[1; 16]).reduce_sum(), 16);

    let mut overflow = [0; 16];
    overflow[0] = i8::MAX;
    overflow[1] = 1;
    assert_eq!(i8x16::from_slice(simd, &overflow).reduce_sum(), i8::MIN);

    let mut underflow = [0; 16];
    underflow[0] = i8::MIN;
    underflow[1] = -1;
    assert_eq!(i8x16::from_slice(simd, &underflow).reduce_sum(), i8::MAX);
}

#[simd_test]
fn reduce_sum_u8x16<S: Simd>(simd: S) {
    assert_eq!(u8x16::from_slice(simd, &[1; 16]).reduce_sum(), 16);

    let mut overflow = [0; 16];
    overflow[0] = u8::MAX;
    overflow[1] = 1;
    assert_eq!(u8x16::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_i16x8<S: Simd>(simd: S) {
    assert_eq!(i16x8::from_slice(simd, &[1; 8]).reduce_sum(), 8);

    let mut overflow = [0; 8];
    overflow[0] = i16::MAX;
    overflow[1] = 1;
    assert_eq!(i16x8::from_slice(simd, &overflow).reduce_sum(), i16::MIN);

    let mut underflow = [0; 8];
    underflow[0] = i16::MIN;
    underflow[1] = -1;
    assert_eq!(i16x8::from_slice(simd, &underflow).reduce_sum(), i16::MAX);
}

#[simd_test]
fn reduce_sum_u16x8<S: Simd>(simd: S) {
    assert_eq!(u16x8::from_slice(simd, &[1; 8]).reduce_sum(), 8);

    let mut overflow = [0; 8];
    overflow[0] = u16::MAX;
    overflow[1] = 1;
    assert_eq!(u16x8::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_i32x4<S: Simd>(simd: S) {
    assert_eq!(i32x4::from_slice(simd, &[1; 4]).reduce_sum(), 4);

    let mut overflow = [0; 4];
    overflow[0] = i32::MAX;
    overflow[1] = 1;
    assert_eq!(i32x4::from_slice(simd, &overflow).reduce_sum(), i32::MIN);

    let mut underflow = [0; 4];
    underflow[0] = i32::MIN;
    underflow[1] = -1;
    assert_eq!(i32x4::from_slice(simd, &underflow).reduce_sum(), i32::MAX);
}

#[simd_test]
fn reduce_sum_u32x4<S: Simd>(simd: S) {
    assert_eq!(u32x4::from_slice(simd, &[1; 4]).reduce_sum(), 4);

    let mut overflow = [0; 4];
    overflow[0] = u32::MAX;
    overflow[1] = 1;
    assert_eq!(u32x4::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_f64x2<S: Simd>(simd: S) {
    assert_eq!(
        f64x2::from_slice(simd, &[1.0, 2.0]).reduce_sum().to_bits(),
        3.0_f64.to_bits()
    );
    assert_eq!(
        f64x2::from_slice(simd, &[-0.0; 2]).reduce_sum().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        f64x2::from_slice(simd, &[0.0; 2]).reduce_sum().to_bits(),
        0.0_f64.to_bits()
    );
    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    assert!(f64x2::from_slice(simd, &[0.0, nan]).reduce_sum().is_nan());
}

#[simd_test]
fn reduce_sum_i64x2<S: Simd>(simd: S) {
    assert_eq!(i64x2::from_slice(simd, &[1; 2]).reduce_sum(), 2);

    let overflow = [i64::MAX, 1];
    assert_eq!(i64x2::from_slice(simd, &overflow).reduce_sum(), i64::MIN);

    let underflow = [i64::MIN, -1];
    assert_eq!(i64x2::from_slice(simd, &underflow).reduce_sum(), i64::MAX);
}

#[simd_test]
fn reduce_sum_u64x2<S: Simd>(simd: S) {
    assert_eq!(u64x2::from_slice(simd, &[1; 2]).reduce_sum(), 2);

    let overflow = [u64::MAX, 1];
    assert_eq!(u64x2::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_f32x8<S: Simd>(simd: S) {
    assert_eq!(
        f32x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .reduce_sum()
            .to_bits(),
        36.0_f32.to_bits()
    );

    let large = 1.0e30_f32;
    assert_eq!(
        f32x8::from_slice(
            simd,
            &[large, large, large, 1.0, -large, -large, -large, 1.0],
        )
        .reduce_sum()
        .to_bits(),
        2.0_f32.to_bits()
    );
    assert_eq!(
        f32x8::from_slice(simd, &[-0.0; 8]).reduce_sum().to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        f32x8::from_slice(simd, &[0.0; 8]).reduce_sum().to_bits(),
        0.0_f32.to_bits()
    );
    let nan = f32::from_bits(0x7fc0_1234);
    assert!(
        f32x8::from_slice(simd, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan])
            .reduce_sum()
            .is_nan()
    );
}

#[simd_test]
fn reduce_sum_i8x32<S: Simd>(simd: S) {
    assert_eq!(i8x32::from_slice(simd, &[1; 32]).reduce_sum(), 32);

    let mut overflow = [0; 32];
    overflow[0] = i8::MAX;
    overflow[16] = 1;
    assert_eq!(i8x32::from_slice(simd, &overflow).reduce_sum(), i8::MIN);

    let mut underflow = [0; 32];
    underflow[0] = i8::MIN;
    underflow[16] = -1;
    assert_eq!(i8x32::from_slice(simd, &underflow).reduce_sum(), i8::MAX);
}

#[simd_test]
fn reduce_sum_u8x32<S: Simd>(simd: S) {
    assert_eq!(u8x32::from_slice(simd, &[1; 32]).reduce_sum(), 32);

    let mut overflow = [0; 32];
    overflow[0] = u8::MAX;
    overflow[16] = 1;
    assert_eq!(u8x32::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_i16x16<S: Simd>(simd: S) {
    assert_eq!(i16x16::from_slice(simd, &[1; 16]).reduce_sum(), 16);

    let mut overflow = [0; 16];
    overflow[0] = i16::MAX;
    overflow[8] = 1;
    assert_eq!(i16x16::from_slice(simd, &overflow).reduce_sum(), i16::MIN);

    let mut underflow = [0; 16];
    underflow[0] = i16::MIN;
    underflow[8] = -1;
    assert_eq!(i16x16::from_slice(simd, &underflow).reduce_sum(), i16::MAX);
}

#[simd_test]
fn reduce_sum_u16x16<S: Simd>(simd: S) {
    assert_eq!(u16x16::from_slice(simd, &[1; 16]).reduce_sum(), 16);

    let mut overflow = [0; 16];
    overflow[0] = u16::MAX;
    overflow[8] = 1;
    assert_eq!(u16x16::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_i32x8<S: Simd>(simd: S) {
    assert_eq!(i32x8::from_slice(simd, &[1; 8]).reduce_sum(), 8);

    let mut overflow = [0; 8];
    overflow[0] = i32::MAX;
    overflow[4] = 1;
    assert_eq!(i32x8::from_slice(simd, &overflow).reduce_sum(), i32::MIN);

    let mut underflow = [0; 8];
    underflow[0] = i32::MIN;
    underflow[4] = -1;
    assert_eq!(i32x8::from_slice(simd, &underflow).reduce_sum(), i32::MAX);
}

#[simd_test]
fn reduce_sum_u32x8<S: Simd>(simd: S) {
    assert_eq!(u32x8::from_slice(simd, &[1; 8]).reduce_sum(), 8);

    let mut overflow = [0; 8];
    overflow[0] = u32::MAX;
    overflow[4] = 1;
    assert_eq!(u32x8::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_f64x4<S: Simd>(simd: S) {
    assert_eq!(
        f64x4::from_slice(simd, &[1.0, 2.0, 3.0, 4.0])
            .reduce_sum()
            .to_bits(),
        10.0_f64.to_bits()
    );

    let large = 1.0e300_f64;
    assert_eq!(
        f64x4::from_slice(simd, &[large, 1.0, -large, 1.0])
            .reduce_sum()
            .to_bits(),
        2.0_f64.to_bits()
    );
    assert_eq!(
        f64x4::from_slice(simd, &[-0.0; 4]).reduce_sum().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        f64x4::from_slice(simd, &[0.0; 4]).reduce_sum().to_bits(),
        0.0_f64.to_bits()
    );
    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    assert!(
        f64x4::from_slice(simd, &[0.0, 0.0, 0.0, nan])
            .reduce_sum()
            .is_nan()
    );
}

#[simd_test]
fn reduce_sum_i64x4<S: Simd>(simd: S) {
    assert_eq!(i64x4::from_slice(simd, &[1; 4]).reduce_sum(), 4);

    let mut overflow = [0; 4];
    overflow[0] = i64::MAX;
    overflow[2] = 1;
    assert_eq!(i64x4::from_slice(simd, &overflow).reduce_sum(), i64::MIN);

    let mut underflow = [0; 4];
    underflow[0] = i64::MIN;
    underflow[2] = -1;
    assert_eq!(i64x4::from_slice(simd, &underflow).reduce_sum(), i64::MAX);
}

#[simd_test]
fn reduce_sum_u64x4<S: Simd>(simd: S) {
    assert_eq!(u64x4::from_slice(simd, &[1; 4]).reduce_sum(), 4);

    let mut overflow = [0; 4];
    overflow[0] = u64::MAX;
    overflow[2] = 1;
    assert_eq!(u64x4::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_f32x16<S: Simd>(simd: S) {
    assert_eq!(
        f32x16::from_slice(simd, &[1.0; 16]).reduce_sum().to_bits(),
        16.0_f32.to_bits()
    );

    let large = 1.0e30_f32;
    assert_eq!(
        f32x16::from_slice(
            simd,
            &[
                large, large, 1.0, 0.0, -large, -large, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ],
        )
        .reduce_sum()
        .to_bits(),
        1.0_f32.to_bits()
    );
    assert_eq!(
        f32x16::from_slice(simd, &[-0.0; 16]).reduce_sum().to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        f32x16::from_slice(simd, &[0.0; 16]).reduce_sum().to_bits(),
        0.0_f32.to_bits()
    );
    let nan = f32::from_bits(0x7fc0_1234);
    assert!(
        f32x16::from_slice(
            simd,
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan,
            ],
        )
        .reduce_sum()
        .is_nan()
    );
}

#[simd_test]
fn reduce_sum_i8x64<S: Simd>(simd: S) {
    assert_eq!(i8x64::from_slice(simd, &[1; 64]).reduce_sum(), 64);

    let mut overflow = [0; 64];
    overflow[0] = i8::MAX;
    overflow[32] = 1;
    assert_eq!(i8x64::from_slice(simd, &overflow).reduce_sum(), i8::MIN);

    let mut underflow = [0; 64];
    underflow[0] = i8::MIN;
    underflow[32] = -1;
    assert_eq!(i8x64::from_slice(simd, &underflow).reduce_sum(), i8::MAX);
}

#[simd_test]
fn reduce_sum_u8x64<S: Simd>(simd: S) {
    assert_eq!(u8x64::from_slice(simd, &[1; 64]).reduce_sum(), 64);

    let mut overflow = [0; 64];
    overflow[0] = u8::MAX;
    overflow[32] = 1;
    assert_eq!(u8x64::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_i16x32<S: Simd>(simd: S) {
    assert_eq!(i16x32::from_slice(simd, &[1; 32]).reduce_sum(), 32);

    let mut overflow = [0; 32];
    overflow[0] = i16::MAX;
    overflow[16] = 1;
    assert_eq!(i16x32::from_slice(simd, &overflow).reduce_sum(), i16::MIN);

    let mut underflow = [0; 32];
    underflow[0] = i16::MIN;
    underflow[16] = -1;
    assert_eq!(i16x32::from_slice(simd, &underflow).reduce_sum(), i16::MAX);
}

#[simd_test]
fn reduce_sum_u16x32<S: Simd>(simd: S) {
    assert_eq!(u16x32::from_slice(simd, &[1; 32]).reduce_sum(), 32);

    let mut overflow = [0; 32];
    overflow[0] = u16::MAX;
    overflow[16] = 1;
    assert_eq!(u16x32::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_i32x16<S: Simd>(simd: S) {
    assert_eq!(i32x16::from_slice(simd, &[1; 16]).reduce_sum(), 16);

    let mut overflow = [0; 16];
    overflow[0] = i32::MAX;
    overflow[8] = 1;
    assert_eq!(i32x16::from_slice(simd, &overflow).reduce_sum(), i32::MIN);

    let mut underflow = [0; 16];
    underflow[0] = i32::MIN;
    underflow[8] = -1;
    assert_eq!(i32x16::from_slice(simd, &underflow).reduce_sum(), i32::MAX);
}

#[simd_test]
fn reduce_sum_u32x16<S: Simd>(simd: S) {
    assert_eq!(u32x16::from_slice(simd, &[1; 16]).reduce_sum(), 16);

    let mut overflow = [0; 16];
    overflow[0] = u32::MAX;
    overflow[8] = 1;
    assert_eq!(u32x16::from_slice(simd, &overflow).reduce_sum(), 0);
}

#[simd_test]
fn reduce_sum_f64x8<S: Simd>(simd: S) {
    assert_eq!(
        f64x8::from_slice(simd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .reduce_sum()
            .to_bits(),
        36.0_f64.to_bits()
    );

    let large = 1.0e300_f64;
    assert_eq!(
        f64x8::from_slice(simd, &[large, 1.0, -large, 0.0, -1.0, 0.0, 0.0, 0.0])
            .reduce_sum()
            .to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        f64x8::from_slice(simd, &[-0.0; 8]).reduce_sum().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        f64x8::from_slice(simd, &[0.0; 8]).reduce_sum().to_bits(),
        0.0_f64.to_bits()
    );
    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    assert!(
        f64x8::from_slice(simd, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan])
            .reduce_sum()
            .is_nan()
    );
}

#[simd_test]
fn reduce_sum_i64x8<S: Simd>(simd: S) {
    assert_eq!(i64x8::from_slice(simd, &[1; 8]).reduce_sum(), 8);

    let mut overflow = [0; 8];
    overflow[0] = i64::MAX;
    overflow[4] = 1;
    assert_eq!(i64x8::from_slice(simd, &overflow).reduce_sum(), i64::MIN);

    let mut underflow = [0; 8];
    underflow[0] = i64::MIN;
    underflow[4] = -1;
    assert_eq!(i64x8::from_slice(simd, &underflow).reduce_sum(), i64::MAX);
}

#[simd_test]
fn reduce_sum_u64x8<S: Simd>(simd: S) {
    assert_eq!(u64x8::from_slice(simd, &[1; 8]).reduce_sum(), 8);

    let mut overflow = [0; 8];
    overflow[0] = u64::MAX;
    overflow[4] = 1;
    assert_eq!(u64x8::from_slice(simd, &overflow).reduce_sum(), 0);
}
