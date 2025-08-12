// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to architecture-specific intrinsics.

#![expect(
    missing_docs,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod fallback;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;
