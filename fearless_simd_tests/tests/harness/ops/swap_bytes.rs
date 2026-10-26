// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.

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

macro_rules! test_swap_bytes {
    ($scalar:ident, $len:literal, $vector:ty, $simd:expr) => {
        let data = get_data::<S, $vector>($simd);
        for data in [data, !data] {
            let expected = data.to_array().map(SwapBytes::swap_bytes);
            let actual = data.swap_bytes().to_array();
            assert!(
                actual == expected,
                "Failed:\n     input: {:X?}\n  expected: {:X?}\n    actual: {:X?}",
                data,
                expected,
                actual
            );
        }
    };
}

#[simd_test]
fn combine_u8<S: Simd>(simd: S) {
    test_swap_bytes!(u8, 16, u8x16<S>, simd);
    test_swap_bytes!(u8, 32, u8x32<S>, simd);
    test_swap_bytes!(u8, 64, u8x64<S>, simd);
}

#[simd_test]
fn combine_i8<S: Simd>(simd: S) {
    test_swap_bytes!(i8, 16, i8x16<S>, simd);
    test_swap_bytes!(i8, 32, i8x32<S>, simd);
    test_swap_bytes!(i8, 64, i8x64<S>, simd);
}

#[simd_test]
fn combine_u16<S: Simd>(simd: S) {
    test_swap_bytes!(u16, 8, u16x8<S>, simd);
    test_swap_bytes!(u16, 16, u16x16<S>, simd);
    test_swap_bytes!(u16, 32, u16x32<S>, simd);
}

#[simd_test]
fn combine_i16<S: Simd>(simd: S) {
    test_swap_bytes!(i16, 8, i16x8<S>, simd);
    test_swap_bytes!(i16, 16, i16x16<S>, simd);
    test_swap_bytes!(i16, 32, i16x32<S>, simd);
}

#[simd_test]
fn combine_u32<S: Simd>(simd: S) {
    test_swap_bytes!(u32, 4, u32x4<S>, simd);
    test_swap_bytes!(u32, 8, u32x8<S>, simd);
    test_swap_bytes!(u32, 16, u32x16<S>, simd);
}

#[simd_test]
fn combine_i32<S: Simd>(simd: S) {
    test_swap_bytes!(i32, 4, i32x4<S>, simd);
    test_swap_bytes!(i32, 8, i32x8<S>, simd);
    test_swap_bytes!(i32, 16, i32x16<S>, simd);
}

#[simd_test]
fn combine_u64<S: Simd>(simd: S) {
    test_swap_bytes!(u64, 2, u64x2<S>, simd);
    test_swap_bytes!(u64, 4, u64x4<S>, simd);
    test_swap_bytes!(u64, 8, u64x8<S>, simd);
}

#[simd_test]
fn combine_i64<S: Simd>(simd: S) {
    test_swap_bytes!(i64, 2, i64x2<S>, simd);
    test_swap_bytes!(i64, 4, i64x4<S>, simd);
    test_swap_bytes!(i64, 8, i64x8<S>, simd);
}
