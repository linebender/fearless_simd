// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SIMD proof token for SSE4.2.

/// A token for SSE4.2 intrinsics on `x86` and `x86_64`.
#[derive(Clone, Copy, Debug)]
pub struct Sse4_2 {
    _private: (),
}

impl Sse4_2 {
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
