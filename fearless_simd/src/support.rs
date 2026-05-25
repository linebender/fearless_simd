// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use core::mem::size_of;

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

/// Like [`core::mem::transmute_copy`], but statically rejects differently-sized types.
///
/// # Safety
///
/// `src` must be valid to copy as `Dst`. This helper only checks the size invariant; the caller
/// is still responsible for the rest of `transmute_copy`'s safety contract.
#[inline(always)]
#[cfg_attr(
    target_arch = "wasm32",
    expect(
        dead_code,
        reason = "native vector conversions are not used by the wasm32 libm Clippy build"
    )
)]
#[allow(
    clippy::disallowed_methods,
    reason = "This is the central checked wrapper around transmute_copy"
)]
pub(crate) unsafe fn checked_transmute_copy<Src: Copy, Dst: Copy>(src: &Src) -> Dst {
    const {
        assert!(
            size_of::<Src>() == size_of::<Dst>(),
            "checked_transmute_copy requires source and destination to have the same size"
        );
    }
    // Safety: The caller upholds `transmute_copy`'s validity requirements, and the
    // const assertion above prevents the "destination larger than source" footgun.
    unsafe { core::mem::transmute_copy(src) }
}

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

/// Selects the input operands to be used for `slignr`/`vext`/etc. when computing a single output block for cross-block
/// "slide" operations. Extracts from [a : b].
#[inline(always)]
#[allow(clippy::allow_attributes, reason = "Only needed in some cfgs.")]
#[allow(dead_code, reason = "Only used in some cfgs.")]
pub(crate) fn cross_block_slide_blocks_at<const N: usize, Block: Copy>(
    a: &[Block; N],
    b: &[Block; N],
    out_idx: usize,
    shift_bytes: usize,
) -> [Block; 2] {
    const BLOCK_BYTES: usize = 16;
    let out_byte_start = out_idx * BLOCK_BYTES + shift_bytes;
    let lo_idx = out_byte_start.div_euclid(BLOCK_BYTES);
    let hi_idx = lo_idx + 1;
    // Concatenation is [a : b], so indices 0..N are from a, indices N..2N are from b
    let lo_block = if lo_idx < N { a[lo_idx] } else { b[lo_idx - N] };
    let hi_block = if hi_idx < N { a[hi_idx] } else { b[hi_idx - N] };
    [lo_block, hi_block]
}
