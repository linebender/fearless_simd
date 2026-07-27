//! Tests enforcing that generic code still compiles

// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(dead_code, reason = "Compile only tests")]

use fearless_simd::prelude::*;

// Ensure that we can cast between generic native-width vectors
fn generic_cast<S: Simd>(x: S::f32s) -> S::u32s {
    x.to_int()
}

// Ensure that a generic vector's mask can select between vectors of that type
fn generic_select<S: Simd, V: SimdBase<S>>(mask: V::Mask, if_true: V, if_false: V) -> V {
    mask.select(if_true, if_false)
}

// Ensure that elements can be copied out of a generic vector
fn generic_first<S: Simd, V: SimdBase<S>>(vector: V) -> V::Element {
    vector[0]
}

// Ensure that a generic vector's 128-bit block is its own block
fn generic_block_splat<S: Simd, V: SimdBase<S>>(block: V::Block) -> V::Block {
    V::Block::block_splat(block)
}

// Ensure that a mask lane's signed integer encoding type is its own mask lane encoding type
fn generic_mask_lane_encoding_idempotent<E: SimdElement>(
    lane_encoding: E::Mask,
) -> <E::Mask as SimdElement>::Mask {
    lane_encoding
}

// Ensure that combining and then splitting generic vectors returns the original type
fn generic_combine_split<S: Simd, V: SimdCombine<S>>(left: V, right: V) -> (V, V) {
    left.combine(right).split()
}

// Ensure that splitting and then combining a generic vector returns the original type
fn generic_split_combine<S: Simd, V: SimdSplit<S>>(vector: V) -> V {
    let (left, right) = vector.split();
    left.combine(right)
}

// Ensure that a generic vector can round-trip through its associated array type
fn generic_array_roundtrip<S: Simd, V: SimdBase<S>>(vector: V) -> V {
    fn require_debug<T: core::fmt::Debug>(_: &T) {}

    let simd = vector.witness();
    let mut array: V::Array = vector.into();
    let array_copy = array;
    #[expect(clippy::clone_on_copy, reason = "Deliberate test")]
    let array_clone = array.clone();
    require_debug(&array_clone);
    let _: Option<V::Element> = array_clone.into_iter().next();
    let _: &[V::Element] = array.as_ref();
    let _: &mut [V::Element] = array.as_mut();
    V::simd_from(simd, array_copy)
}
