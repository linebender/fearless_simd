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
