// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compile-only fixtures for testing `fearless_simd` without the standard library.

#![no_std]

use fearless_simd::Simd;
use fearless_simd_macros::simd;

/// Doubles every value using a function annotated with [`simd`].
///
/// This function intentionally lives in the library target so CI can compile it for a target
/// where the standard library is unavailable.
#[simd]
pub fn double_without_std<S: Simd>(_simd: S, values: &mut [u32]) {
    for value in values {
        *value *= 2;
    }
}
