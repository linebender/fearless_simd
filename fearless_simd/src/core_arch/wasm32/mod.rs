// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// A token for WASM SIMD128.
#[derive(Clone, Copy, Debug)]
pub struct WasmSimd128 {
    _private: (),
}

// There is intentionally no method delegation here because all the WASM SIMD128 methods are enabled or disabled
// statically--there is no feature detection.
impl WasmSimd128 {
    /// Create a SIMD token.
    #[inline]
    pub const fn new() -> Self {
        Self { _private: () }
    }
}
