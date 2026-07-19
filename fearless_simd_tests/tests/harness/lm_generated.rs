// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(not(miri))] // too slow
mod mask_roundtrip;
#[cfg(not(miri))] // too slow
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod mask_roundtrip_x86;
mod mod_256;
mod mod_512;
mod swizzle_dyn_precise;
