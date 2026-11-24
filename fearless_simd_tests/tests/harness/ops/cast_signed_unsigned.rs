// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

fn get_data_signed<
    T: From<i8> + Copy + std::ops::Div<Output = T> + std::ops::AddAssign,
    const N: usize,
>(
    min: T,
    max: T,
) -> [T; N] {
    let mut arr = [T::from(0); N];
    let mut div = T::from(1);
    #[allow(clippy::needless_range_loop, reason = "it's easier")]
    for i in 0..N {
        let x = if i.is_multiple_of(2) { min } else { max };
        arr[i] = x / div;
        if !i.is_multiple_of(2) {
            div += T::from(1);
        }
    }
    arr
}
fn get_data_unsigned<
    T: From<u8> + Copy + std::ops::Div<Output = T> + std::ops::AddAssign,
    const N: usize,
>(
    max: T,
) -> [T; N] {
    let mut arr = [T::from(0); N];
    let mut div = T::from(1);
    #[allow(clippy::needless_range_loop, reason = "it's easier")]
    for i in 0..N {
        let x = if i.is_multiple_of(2) { div } else { max / div };
        arr[i] = x;
        if !i.is_multiple_of(2) {
            div += T::from(1);
        }
    }
    arr
}

// One concrete test row per supported vector type.

#[simd_test]
fn cast_unsigned_i8x16<S: Simd>(simd: S) {
    let data = get_data_signed::<i8, 16>(i8::MIN, i8::MAX);
    let x = i8x16::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i8x32<S: Simd>(simd: S) {
    let data = get_data_signed::<i8, 32>(i8::MIN, i8::MAX);
    let x = i8x32::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i8x64<S: Simd>(simd: S) {
    let data = get_data_signed::<i8, 64>(i8::MIN, i8::MAX);
    let x = i8x64::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i16x8<S: Simd>(simd: S) {
    let data = get_data_signed::<i16, 8>(i16::MIN, i16::MAX);
    let x = i16x8::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i16x16<S: Simd>(simd: S) {
    let data = get_data_signed::<i16, 16>(i16::MIN, i16::MAX);
    let x = i16x16::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i16x32<S: Simd>(simd: S) {
    let data = get_data_signed::<i16, 32>(i16::MIN, i16::MAX);
    let x = i16x32::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i32x4<S: Simd>(simd: S) {
    let data = get_data_signed::<i32, 4>(i32::MIN, i32::MAX);
    let x = i32x4::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i32x8<S: Simd>(simd: S) {
    let data = get_data_signed::<i32, 8>(i32::MIN, i32::MAX);
    let x = i32x8::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i32x16<S: Simd>(simd: S) {
    let data = get_data_signed::<i32, 16>(i32::MIN, i32::MAX);
    let x = i32x16::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i64x2<S: Simd>(simd: S) {
    let data = get_data_signed::<i64, 2>(i64::MIN, i64::MAX);
    let x = i64x2::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i64x4<S: Simd>(simd: S) {
    let data = get_data_signed::<i64, 4>(i64::MIN, i64::MAX);
    let x = i64x4::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_unsigned_i64x8<S: Simd>(simd: S) {
    let data = get_data_signed::<i64, 8>(i64::MIN, i64::MAX);
    let x = i64x8::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data.map(|v| v.cast_unsigned()));
}

#[simd_test]
fn cast_signed_u8x16<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u8, 16>(u8::MAX);
    let x = u8x16::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u8x32<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u8, 32>(u8::MAX);
    let x = u8x32::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u8x64<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u8, 64>(u8::MAX);
    let x = u8x64::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u16x8<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u16, 8>(u16::MAX);
    let x = u16x8::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u16x16<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u16, 16>(u16::MAX);
    let x = u16x16::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u16x32<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u16, 32>(u16::MAX);
    let x = u16x32::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u32x4<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u32, 4>(u32::MAX);
    let x = u32x4::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u32x8<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u32, 8>(u32::MAX);
    let x = u32x8::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u32x16<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u32, 16>(u32::MAX);
    let x = u32x16::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u64x2<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u64, 2>(u64::MAX);
    let x = u64x2::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u64x4<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u64, 4>(u64::MAX);
    let x = u64x4::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

#[simd_test]
fn cast_signed_u64x8<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u64, 8>(u64::MAX);
    let x = u64x8::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data.map(|v| v.cast_signed()));
}

// Now the types where casting is a noop

#[simd_test]
fn cast_signed_i8x16<S: Simd>(simd: S) {
    let data = get_data_signed::<i8, 16>(i8::MIN, i8::MAX);
    let x = i8x16::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i8x32<S: Simd>(simd: S) {
    let data = get_data_signed::<i8, 32>(i8::MIN, i8::MAX);
    let x = i8x32::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i8x64<S: Simd>(simd: S) {
    let data = get_data_signed::<i8, 64>(i8::MIN, i8::MAX);
    let x = i8x64::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i16x8<S: Simd>(simd: S) {
    let data = get_data_signed::<i16, 8>(i16::MIN, i16::MAX);
    let x = i16x8::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i16x16<S: Simd>(simd: S) {
    let data = get_data_signed::<i16, 16>(i16::MIN, i16::MAX);
    let x = i16x16::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i16x32<S: Simd>(simd: S) {
    let data = get_data_signed::<i16, 32>(i16::MIN, i16::MAX);
    let x = i16x32::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i32x4<S: Simd>(simd: S) {
    let data = get_data_signed::<i32, 4>(i32::MIN, i32::MAX);
    let x = i32x4::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i32x8<S: Simd>(simd: S) {
    let data = get_data_signed::<i32, 8>(i32::MIN, i32::MAX);
    let x = i32x8::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i32x16<S: Simd>(simd: S) {
    let data = get_data_signed::<i32, 16>(i32::MIN, i32::MAX);
    let x = i32x16::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i64x2<S: Simd>(simd: S) {
    let data = get_data_signed::<i64, 2>(i64::MIN, i64::MAX);
    let x = i64x2::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i64x4<S: Simd>(simd: S) {
    let data = get_data_signed::<i64, 4>(i64::MIN, i64::MAX);
    let x = i64x4::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_signed_i64x8<S: Simd>(simd: S) {
    let data = get_data_signed::<i64, 8>(i64::MIN, i64::MAX);
    let x = i64x8::simd_from(simd, data);
    assert_eq!(*x.cast_signed(), data);
}

#[simd_test]
fn cast_unsigned_u8x16<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u8, 16>(u8::MAX);
    let x = u8x16::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u8x32<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u8, 32>(u8::MAX);
    let x = u8x32::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u8x64<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u8, 64>(u8::MAX);
    let x = u8x64::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u16x8<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u16, 8>(u16::MAX);
    let x = u16x8::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u16x16<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u16, 16>(u16::MAX);
    let x = u16x16::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u16x32<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u16, 32>(u16::MAX);
    let x = u16x32::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u32x4<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u32, 4>(u32::MAX);
    let x = u32x4::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u32x8<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u32, 8>(u32::MAX);
    let x = u32x8::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u32x16<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u32, 16>(u32::MAX);
    let x = u32x16::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u64x2<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u64, 2>(u64::MAX);
    let x = u64x2::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u64x4<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u64, 4>(u64::MAX);
    let x = u64x4::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}

#[simd_test]
fn cast_unsigned_u64x8<S: Simd>(simd: S) {
    let data = get_data_unsigned::<u64, 8>(u64::MAX);
    let x = u64x8::simd_from(simd, data);
    assert_eq!(*x.cast_unsigned(), data);
}
