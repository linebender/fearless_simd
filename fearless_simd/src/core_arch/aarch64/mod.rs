// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SIMD proof token for aarch64 Neon.

/// A token for Neon intrinsics on aarch64.
#[derive(Clone, Copy, Debug)]
pub struct Neon {
    _private: (),
}

impl Neon {
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
