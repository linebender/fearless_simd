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

const fn compact_control_table(expand: bool) -> [u64; 256] {
    let mut table = [u64::MAX; 256];
    let mut mask = 0_usize;
    while mask < table.len() {
        let mut control = u64::MAX;
        let mut selected = 0_usize;
        let mut lane = 0_usize;
        while lane < 8 {
            if mask & (1 << lane) != 0 {
                let output_lane = if expand { lane } else { selected };
                control &= !(0xff_u64 << (output_lane * 8));
                let index = if expand { selected } else { lane };
                control |= (index as u64) << (output_lane * 8);
                selected += 1;
            }
            lane += 1;
        }
        table[mask] = control;
        mask += 1;
    }
    table
}

const fn compact_count_table() -> [u8; 256] {
    let mut table = [0; 256];
    let mut mask = 0_usize;
    while mask < table.len() {
        let mut bits = mask;
        let mut count = 0;
        while bits != 0 {
            if bits & 1 != 0 {
                count += 1;
            }
            bits >>= 1;
        }
        table[mask] = count;
        mask += 1;
    }
    table
}

const fn compact_splice_control_table() -> [u128; 9] {
    let mut table = [u128::MAX; 9];
    let mut low_count = 0;
    while low_count <= 8 {
        let mut control = u128::MAX;
        let mut lane = 0;
        while lane < low_count {
            control &= !(0xff_u128 << (lane * 8));
            control |= (lane as u128) << (lane * 8);
            lane += 1;
        }
        lane = 0;
        while lane < 8 {
            let output_lane = low_count + lane;
            if output_lane < 16 {
                control &= !(0xff_u128 << (output_lane * 8));
                control |= ((lane + 8) as u128) << (output_lane * 8);
            }
            lane += 1;
        }
        table[low_count] = control;
        low_count += 1;
    }
    table
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "lane is bounded to 0..16 by the table-construction loop"
)]
const fn compact_16_splice_control_table() -> [Aligned256<[u8; 32]>; 17] {
    let mut table = [Aligned256([0x80; 32]); 17];
    let mut low_count = 0;
    while low_count <= 16 {
        let mut lane = 0;
        while lane < 16 {
            let output_lane = low_count + lane;
            if output_lane < 32 {
                table[low_count].0[output_lane] = lane as u8;
            }
            lane += 1;
        }
        low_count += 1;
    }
    table
}

const fn compact_prefix_mask_table() -> [Aligned256<[u8; 32]>; 33] {
    let mut table = [Aligned256([0; 32]); 33];
    let mut count = 0;
    while count <= 32 {
        let mut lane = 0;
        while lane < count {
            table[count].0[lane] = u8::MAX;
            lane += 1;
        }
        count += 1;
    }
    table
}

#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub(crate) struct Compact32StitchControls {
    pub(crate) left_same: [u8; 32],
    pub(crate) left_cross: [u8; 32],
    pub(crate) right_same: [u8; 32],
    pub(crate) right_cross: [u8; 32],
}

/// Controls for concatenating two compacted 32-byte vectors entirely in registers.
///
/// For a first-vector length `count`, `left_*` shifts the second vector left by `count`
/// bytes and `right_*` shifts it right by `32 - count` bytes. Separate same-lane and
/// cross-lane controls account for AVX2 `vpshufb` operating independently in each
/// 128-bit lane.
#[allow(
    clippy::cast_possible_truncation,
    reason = "shuffle indices are reduced modulo 16"
)]
const fn compact_32_stitch_control_table() -> [Compact32StitchControls; 33] {
    const EMPTY: Compact32StitchControls = Compact32StitchControls {
        left_same: [0x80; 32],
        left_cross: [0x80; 32],
        right_same: [0x80; 32],
        right_cross: [0x80; 32],
    };

    let mut table = [EMPTY; 33];
    let mut count = 0;
    while count <= 32 {
        let mut lane = 0;
        while lane < 32 {
            if lane >= count {
                let source = lane - count;
                if lane / 16 == source / 16 {
                    table[count].left_same[lane] = (source % 16) as u8;
                } else {
                    table[count].left_cross[lane] = (source % 16) as u8;
                }
            }

            let source = lane + (32 - count);
            if source < 32 {
                if lane / 16 == source / 16 {
                    table[count].right_same[lane] = (source % 16) as u8;
                } else {
                    table[count].right_cross[lane] = (source % 16) as u8;
                }
            }
            lane += 1;
        }
        count += 1;
    }
    table
}

/// Eight-byte shuffle controls indexed by an eight-bit selection mask.
#[allow(
    dead_code,
    reason = "Used only by SIMD backends with byte-table shuffles"
)]
pub(crate) const COMPRESS_8_CONTROLS: [u64; 256] = compact_control_table(false);

/// Eight-byte inverse-shuffle controls indexed by an eight-bit selection mask.
#[allow(
    dead_code,
    reason = "Used only by SIMD backends with byte-table shuffles"
)]
pub(crate) const EXPAND_8_CONTROLS: [u64; 256] = compact_control_table(true);

/// Population counts indexed by an eight-bit selection mask.
#[allow(
    dead_code,
    reason = "Used only by SIMD backends with byte-table shuffles"
)]
pub(crate) const COMPACT_8_COUNTS: [u8; 256] = compact_count_table();

/// Shuffle controls that splice two independently compacted eight-byte halves.
#[allow(dead_code, reason = "Used only by x86 byte compression")]
pub(crate) const COMPACT_8_SPLICE_CONTROLS: [u128; 9] = compact_splice_control_table();

/// Shuffle controls that append a compacted 16-byte high lane after a compacted low lane.
#[allow(dead_code, reason = "Used only by AVX2 byte compression")]
pub(crate) const COMPACT_16_SPLICE_CONTROLS: [Aligned256<[u8; 32]>; 17] =
    compact_16_splice_control_table();

/// Shuffle controls that concatenate two independently compacted 32-byte vectors.
#[allow(dead_code, reason = "Used only by AVX2 64-byte compression")]
pub(crate) const COMPACT_32_STITCH_CONTROLS: [Compact32StitchControls; 33] =
    compact_32_stitch_control_table();

/// Byte prefix masks indexed by the number of active lanes.
#[allow(dead_code, reason = "Used only by x86 byte compression")]
pub(crate) const COMPACT_PREFIX_MASKS: [Aligned256<[u8; 32]>; 33] = compact_prefix_mask_table();
