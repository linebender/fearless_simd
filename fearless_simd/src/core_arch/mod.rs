// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to architecture-specific intrinsics.

#![cfg_attr(
    not(target_arch = "wasm32"),
    expect(
        clippy::missing_safety_doc,
        reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
    )
)]
#![cfg_attr(
    all(not(feature = "std"), target_arch = "x86_64"),
    expect(
        clippy::too_many_arguments,
        reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
    )
)]

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod fallback;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;
