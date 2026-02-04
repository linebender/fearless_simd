// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Access to architecture-specific intrinsics.

#![expect(
    missing_docs,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]
#![allow(
    clippy::allow_attributes_without_reason,
    reason = "these attributes are copied from stdarch"
)]
#![allow(
    deprecated,
    reason = "some intrinsics are deprecated, and hence their wrappers call deprecated functions"
)]
#![cfg_attr(
    any(target_arch = "x86", target_arch = "x86_64"),
    expect(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "_mm_prefetch is safe to call, but clippy thinks it dereferences the pointer for some reason"
    )
)]

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod fallback;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm32;
