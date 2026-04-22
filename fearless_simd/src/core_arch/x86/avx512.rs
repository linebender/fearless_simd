// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to AVX-512 intrinsics (Ice Lake feature set).

/// A token for AVX-512 intrinsics (Ice Lake feature set) on `x86` and `x86_64`.
#[derive(Clone, Copy, Debug)]
pub struct Avx512 {
    _private: (),
}

impl Avx512 {
    /// Create a SIMD token.
    ///
    /// # Safety
    ///
    /// The required CPU features must be available.
    pub const unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}
