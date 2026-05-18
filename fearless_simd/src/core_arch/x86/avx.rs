// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SIMD proof token for AVX.

/// A token for AVX intrinsics on `x86` and `x86_64`.
#[derive(Clone, Copy, Debug)]
pub struct Avx {
    _private: (),
}

impl Avx {
    /// Create a SIMD token.
    ///
    /// # Safety
    ///
    /// The required CPU features must be available.
    #[inline]
    pub const unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}
