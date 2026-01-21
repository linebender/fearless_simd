// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
#[expect(
    unnameable_types,
    reason = "This is used internally, but needs to be `pub` as it's used in a sealed interface"
)]
/// Wrapper for internal native vector types that gives them 128-bit alignment.
pub struct Aligned128<T>(pub T);

#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
#[expect(
    unnameable_types,
    reason = "This is used internally, but needs to be `pub` as it's used in a sealed interface"
)]
/// Wrapper for internal native vector types that gives them 256-bit alignment.
pub struct Aligned256<T>(pub T);

#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
#[expect(
    unnameable_types,
    reason = "This is used internally, but needs to be `pub` as it's used in a sealed interface"
)]
/// Wrapper for internal native vector types that gives them 512-bit alignment.
pub struct Aligned512<T>(pub T);

/// The actual `Debug` implementation for all `SimdBase` types. This only needs to be monomorphized once per element
/// type, rather than once per vector type.
#[inline(never)]
pub(crate) fn simd_debug_impl<Element: core::fmt::Debug>(
    f: &mut core::fmt::Formatter<'_>,
    type_name: &str,
    token: &dyn core::fmt::Debug,
    items: &[Element],
) -> core::fmt::Result {
    f.debug_struct(type_name)
        .field("val", &items)
        .field("simd", token)
        .finish()
}
