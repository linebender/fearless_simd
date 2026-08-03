//! Tests enforcing that generic code still compiles

// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(dead_code, reason = "Compile only tests")]

use fearless_simd::prelude::*;

// Ensure that we can cast between generic native-width vectors
fn generic_cast<S: Simd>(x: S::f32s) -> S::u32s {
    x.to_int()
}

// Ensure that a generic vector's byte representation is itself a same-token
// byte vector whose byte representation is idempotent.
fn generic_bytes<S: Simd, V: SimdBase<S>>(value: V) -> V {
    let simd = value.witness();
    let bytes = value.to_bytes();
    let bytes =
        <V::Bytes as SimdBase<S>>::from_slice(simd, <V::Bytes as SimdBase<S>>::as_slice(&bytes));
    let _: u8 = bytes[0];
    let bytes: V::Bytes = bytes.to_bytes();
    let bytes: V::Bytes = bytes.bitcast();
    V::from_bytes(bytes)
}

fn generic_bytes_idempotent<T: Bytes>(bytes: T::Bytes) -> T::Bytes {
    bytes.to_bytes()
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

// Ensure widening and narrowing can be expressed using only the bespoke generic traits.
fn generic_widen_narrow<S: Simd, V: SimdWiden<S>>(vector: V) -> V {
    let (low, high) = vector.widen();
    low.narrow(high)
}

fn generic_saturating_widen_narrow<S: Simd, V: SimdWiden<S>>(vector: V) -> V {
    let (low, high) = vector.widen();
    low.saturating_narrow(high)
}

fn generic_narrow<S: Simd, V: SimdNarrow<S>>(low: V, high: V) -> V::Narrowed {
    low.narrow(high)
}

fn generic_saturating_narrow<S: Simd, V: SimdNarrow<S>>(low: V, high: V) -> V::Narrowed {
    low.saturating_narrow(high)
}

// Ensure the native-width associated-type bounds expose every adjacent relationship without
// additional where-clauses.
fn generic_native_width_widen<S: Simd>(value: S::u8s) -> (S::u16s, S::u16s) {
    value.widen()
}

fn generic_native_width_narrow<S: Simd>(low: S::i64s, high: S::i64s) -> S::i32s {
    low.narrow(high)
}

fn generic_native_width_saturating_narrow<S: Simd>(low: S::u32s, high: S::u32s) -> S::u16s {
    low.saturating_narrow(high)
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
