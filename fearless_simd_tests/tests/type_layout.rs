//! Tests enforcing the memory layout promises made for the vector types.

// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{
    Simd, SimdBase, f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, i8x16, i8x32, i8x64, i16x8, i16x16,
    i16x32, i32x4, i32x8, i32x16, i64x2, i64x4, i64x8, u8x16, u8x32, u8x64, u16x8, u16x16, u16x32,
    u32x4, u32x8, u32x16, u64x2, u64x4, u64x8,
};
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn vector_layout<S: Simd>(_: S) {
    fn assert_layout<S: Simd, T: SimdBase<S>>() {
        assert_eq!(
            size_of::<T>(),
            size_of::<T::Array>(),
            "vector type's size does not match documented promise"
        );
        assert_eq!(
            align_of::<T>(),
            size_of::<T::Array>(),
            "vector's alignment does not match documented promise"
        );
    }

    assert_layout::<S, u8x64<S>>();
    assert_layout::<S, u8x32<S>>();
    assert_layout::<S, u8x16<S>>();

    assert_layout::<S, u16x32<S>>();
    assert_layout::<S, u16x16<S>>();
    assert_layout::<S, u16x8<S>>();

    assert_layout::<S, u32x16<S>>();
    assert_layout::<S, u32x8<S>>();
    assert_layout::<S, u32x4<S>>();

    assert_layout::<S, u64x8<S>>();
    assert_layout::<S, u64x4<S>>();
    assert_layout::<S, u64x2<S>>();

    assert_layout::<S, i8x64<S>>();
    assert_layout::<S, i8x32<S>>();
    assert_layout::<S, i8x16<S>>();

    assert_layout::<S, i16x32<S>>();
    assert_layout::<S, i16x16<S>>();
    assert_layout::<S, i16x8<S>>();

    assert_layout::<S, i32x16<S>>();
    assert_layout::<S, i32x8<S>>();
    assert_layout::<S, i32x4<S>>();

    assert_layout::<S, i64x8<S>>();
    assert_layout::<S, i64x4<S>>();
    assert_layout::<S, i64x2<S>>();

    assert_layout::<S, f32x16<S>>();
    assert_layout::<S, f32x8<S>>();
    assert_layout::<S, f32x4<S>>();

    assert_layout::<S, f64x8<S>>();
    assert_layout::<S, f64x4<S>>();
    assert_layout::<S, f64x2<S>>();
}
