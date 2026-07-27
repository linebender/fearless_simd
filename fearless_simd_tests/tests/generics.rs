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
