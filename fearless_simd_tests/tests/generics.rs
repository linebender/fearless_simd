//! Tests enforcing that generic code still compiles

// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(dead_code, reason = "Compile only tests")]

use fearless_simd::{f32x4, f64x2, prelude::*};

// Ensure that we can cast between generic native-width vectors
fn generic_cast<S: Simd>(x: S::f32s) -> S::u32s {
    x.to_int()
}

fn generic_f64_to_u64<S: Simd>(x: S::f64s) -> S::u64s {
    x.to_int()
}

fn generic_f64_to_i64<S: Simd>(x: S::f64s) -> S::i64s {
    x.to_int()
}

fn generic_f64_to_u64_precise<S: Simd>(x: S::f64s) -> S::u64s {
    x.to_int_precise()
}

fn generic_f64_to_i64_precise<S: Simd>(x: S::f64s) -> S::i64s {
    x.to_int_precise()
}

fn generic_u64_to_f64<S: Simd>(x: S::u64s) -> S::f64s {
    x.to_float()
}

fn generic_i64_to_f64<S: Simd>(x: S::i64s) -> S::f64s {
    x.to_float()
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

// Ensure that a generic four-way interleaved workload can iterate over a scalar slice.
// This color transform swaps the red and blue channels of complete RGBA pixels.
fn generic_four_interleaved_color_transform<S: Simd, V: SimdInterleaved<S>>(
    simd: S,
    pixels: &mut [V::Element],
) {
    let mut chunks = pixels.chunks_exact_mut(V::N * 4);
    for chunk in &mut chunks {
        let [red, green, blue, alpha] = V::load_four_interleaved(simd, chunk);
        V::store_four_interleaved([blue, green, red, alpha], chunk);
    }

    for pixel in chunks.into_remainder().chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
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

fn generic_fixed_width_float_widen<S: Simd>(value: f32x4<S>) -> (f64x2<S>, f64x2<S>) {
    value.widen()
}

fn generic_fixed_width_float_narrow<S: Simd>(low: f64x2<S>, high: f64x2<S>) -> f32x4<S> {
    low.narrow(high)
}

fn generic_native_width_float_widen<S: Simd>(value: S::f32s) -> (S::f64s, S::f64s) {
    value.widen()
}

fn generic_native_width_float_narrow<S: Simd>(low: S::f64s, high: S::f64s) -> S::f32s {
    low.narrow(high)
}

fn generic_native_width_float_saturating_narrow<S: Simd>(low: S::f64s, high: S::f64s) -> S::f32s {
    low.saturating_narrow(high)
}

// Ensure that a generic vector can round-trip through its associated array type
fn generic_array_roundtrip<S: Simd, V: SimdBase<S>>(vector: V) -> V {
    fn require_debug<T: core::fmt::Debug>(_: &T) {}

    let simd = vector.witness();
    let mut vector = vector;
    let mut array = vector.as_array();
    let array_copy = array;
    #[expect(clippy::clone_on_copy, reason = "Deliberate test")]
    let array_clone = array.clone();
    require_debug(&array_clone);
    let _: Option<V::Element> = array_clone.into_iter().next();
    let _: &[V::Element] = array.as_ref();
    let _: &mut [V::Element] = array.as_mut();
    let _: &V::Array = vector.as_array_ref();
    let _: &mut V::Array = vector.as_array_mut();
    let _ = V::load_array_ref(simd, &array_copy);
    vector.store_array(&mut array);
    V::load_array(simd, array)
}

// Ensure that `SimdInt` exposes scalar arithmetic on its element type without
// requiring a repeated `T::Element: SimdIntElement` bound.
#[expect(clippy::op_ref, reason = "Deliberately test operations by reference")]
fn generic_int_element_arithmetic<S: Simd, T: SimdInt<S>>(
    value: T::Element,
    shift: usize,
) -> T::Element {
    let one = T::Element::from(true);
    let by_value = ((value + one) << shift) - one;
    let by_reference = ((value + &one) << &shift) - &one;
    (by_value & value) | (by_reference ^ &value)
}
