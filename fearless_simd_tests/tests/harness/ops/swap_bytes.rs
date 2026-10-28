// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

trait SwapBytes {
    fn swap_bytes(self) -> Self;
}
impl SwapBytes for u8 {
    fn swap_bytes(self) -> Self {
        self.swap_bytes()
    }
}
impl SwapBytes for u16 {
    fn swap_bytes(self) -> Self {
        self.swap_bytes()
    }
}
impl SwapBytes for u32 {
    fn swap_bytes(self) -> Self {
        self.swap_bytes()
    }
}
impl SwapBytes for u64 {
    fn swap_bytes(self) -> Self {
        self.swap_bytes()
    }
}
impl SwapBytes for i8 {
    fn swap_bytes(self) -> Self {
        self.swap_bytes()
    }
}
impl SwapBytes for i16 {
    fn swap_bytes(self) -> Self {
        self.swap_bytes()
    }
}
impl SwapBytes for i32 {
    fn swap_bytes(self) -> Self {
        self.swap_bytes()
    }
}
impl SwapBytes for i64 {
    fn swap_bytes(self) -> Self {
        self.swap_bytes()
    }
}

fn get_data<S: Simd, T: SimdInt<S>>(simd: S) -> T {
    let unique_bytes: [u8; 64] = std::array::from_fn(|i| i as u8);
    T::ByteVector::from_slice(simd, &unique_bytes[..T::ByteVector::LEN]).bitcast::<T>()
}

#[simd_test]
fn swap_bytes_u8x16<S: Simd>(simd: S) {
    let data = get_data::<S, u8x16<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u8x32<S: Simd>(simd: S) {
    let data = get_data::<S, u8x32<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u8x64<S: Simd>(simd: S) {
    let data = get_data::<S, u8x64<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i8x16<S: Simd>(simd: S) {
    let data = get_data::<S, i8x16<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i8x32<S: Simd>(simd: S) {
    let data = get_data::<S, i8x32<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i8x64<S: Simd>(simd: S) {
    let data = get_data::<S, i8x64<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u16x8<S: Simd>(simd: S) {
    let data = get_data::<S, u16x8<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u16x16<S: Simd>(simd: S) {
    let data = get_data::<S, u16x16<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u16x32<S: Simd>(simd: S) {
    let data = get_data::<S, u16x32<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i16x8<S: Simd>(simd: S) {
    let data = get_data::<S, i16x8<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i16x16<S: Simd>(simd: S) {
    let data = get_data::<S, i16x16<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i16x32<S: Simd>(simd: S) {
    let data = get_data::<S, i16x32<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u32x4<S: Simd>(simd: S) {
    let data = get_data::<S, u32x4<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u32x8<S: Simd>(simd: S) {
    let data = get_data::<S, u32x8<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u32x16<S: Simd>(simd: S) {
    let data = get_data::<S, u32x16<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i32x4<S: Simd>(simd: S) {
    let data = get_data::<S, i32x4<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i32x8<S: Simd>(simd: S) {
    let data = get_data::<S, i32x8<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i32x16<S: Simd>(simd: S) {
    let data = get_data::<S, i32x16<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u64x2<S: Simd>(simd: S) {
    let data = get_data::<S, u64x2<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u64x4<S: Simd>(simd: S) {
    let data = get_data::<S, u64x4<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_u64x8<S: Simd>(simd: S) {
    let data = get_data::<S, u64x8<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i64x2<S: Simd>(simd: S) {
    let data = get_data::<S, i64x2<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i64x4<S: Simd>(simd: S) {
    let data = get_data::<S, i64x4<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}

#[simd_test]
fn swap_bytes_i64x8<S: Simd>(simd: S) {
    let data = get_data::<S, i64x8<S>>(simd);
    assert_eq!(
        *data.swap_bytes(),
        data.to_array().map(SwapBytes::swap_bytes)
    );
}
