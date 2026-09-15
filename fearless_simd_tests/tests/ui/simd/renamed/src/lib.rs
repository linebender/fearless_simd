// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![no_std]
#![forbid(unsafe_code)]

use fearless_simd_macros::simd;
use simd_backend as fearless_simd;

#[simd]
pub fn sum<S: simd_backend::Simd>(_: S, a: u64, b: u64, c: u64) -> u64 {
    a.wrapping_add(b).wrapping_add(c)
}

// Instantiate the macro without force_support_fallback and without std. This
// also checks that it does not need Simd imported to resolve generated calls.
pub fn baseline_sum() -> u64 {
    simd_backend::dispatch!(simd_backend::Level::baseline(), simd => sum(simd, 1, 2, 3))
}

pub mod facade {
    pub use simd_backend;
}

pub mod through_reexport {
    use crate::facade::simd_backend as fearless_simd;
    use fearless_simd_macros::simd;

    #[simd]
    pub fn sum<S: fearless_simd::Simd>(_: S, a: u64, b: u64, c: u64) -> u64 {
        a.wrapping_add(b).wrapping_add(c)
    }

    pub fn baseline_sum() -> u64 {
        fearless_simd::dispatch!(fearless_simd::Level::baseline(), simd => sum(simd, 1, 2, 3))
    }
}
