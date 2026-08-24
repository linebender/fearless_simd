// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{Fallback, Simd};
use fearless_simd_macros::simd;

#[simd]
#[inline(never)]
#[must_use]
fn with_generics<'a, S, T>(simd: S, value: &'a mut T) -> Result<&'a mut T, ()>
where
    S: Simd,
{
    let _ = simd.level();
    Ok(value)
}

#[simd]
#[inline]
fn with_inline<S: Simd>(_simd: S, value: u32) -> u32 {
    value
}

#[simd]
#[inline(always)]
fn with_inline_always<S: Simd>(_simd: S, value: u32) -> u32 {
    value
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[simd]
#[target_feature(enable = "sse2")]
unsafe fn with_target_feature<S: Simd>(_simd: S, value: u32) -> u32 {
    value
}

#[simd]
fn mutable_token<S: Simd>(mut simd: S, value: u32) -> u32 {
    let _ = &mut simd;
    value
}

#[simd]
fn wildcard_token<S: Simd>(_: S, value: u32) -> u32 {
    value
}

#[simd]
fn wildcard_binding_is_hygienic<S: Simd>(_: S, __fearless_simd_token: u32) -> u32 {
    __fearless_simd_token
}

#[simd]
unsafe extern "Rust" fn unsafe_with_abi<S: Simd>(simd: S, ptr: *const u32) -> u32 {
    let _ = simd.level();
    // SAFETY: The caller promises that `ptr` is valid to read.
    unsafe { ptr.read() }
}

fn main() {
    let fallback = Fallback::new();
    let mut value = 1;
    assert_eq!(with_generics(fallback, &mut value), Ok(&mut 1));
    assert_eq!(with_inline(fallback, 2), 2);
    assert_eq!(with_inline_always(fallback, 3), 3);
    assert_eq!(mutable_token(fallback, 2), 2);
    assert_eq!(wildcard_token(fallback, 3), 3);
    assert_eq!(wildcard_binding_is_hygienic(fallback, 4), 4);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: SSE2 is part of the x86-64 baseline used by this UI test.
    assert_eq!(unsafe { with_target_feature(fallback, 5) }, 5);
    // SAFETY: The pointer is derived from a live local value.
    assert_eq!(unsafe { unsafe_with_abi(fallback, &value) }, 1);
}
