// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SIMD proof token for AVX2.

/// A token for AVX2 intrinsics on `x86` and `x86_64`.
#[derive(Clone, Copy, Debug)]
pub struct Avx2 {
    _private: (),
}

impl Avx2 {
    /// Create a SIMD token.
    ///
    /// # Safety
    ///
    /// The required CPU features must be available.
    pub const unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}
