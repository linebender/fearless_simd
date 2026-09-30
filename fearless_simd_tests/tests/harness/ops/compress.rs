// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn compact_u8x16<S: Simd>(simd: S) {
    let values: [u8; 16] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as u8
    });
    let merge: [u8; 16] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u8x16::simd_from(simd, values);
    let merge_vec = u8x16::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 16);
    let single_lanes = (0..16).map(|lane| 1_u64 << lane);
    let prefixes = (1..=16).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..16).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask8x16::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 16];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 16];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..16 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u8x16(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u8x16(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u8x16(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u8x16(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u8x32<S: Simd>(simd: S) {
    let values: [u8; 32] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as u8
    });
    let merge: [u8; 32] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u8x32::simd_from(simd, values);
    let merge_vec = u8x32::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 32);
    let single_lanes = (0..32).map(|lane| 1_u64 << lane);
    let prefixes = (1..=32).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..32).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask8x32::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 32];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 32];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..32 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u8x32(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u8x32(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u8x32(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u8x32(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u8x64<S: Simd>(simd: S) {
    let values: [u8; 64] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as u8
    });
    let merge: [u8; 64] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u8x64::simd_from(simd, values);
    let merge_vec = u8x64::simd_from(simd, merge);
    let all_lanes = u64::MAX;
    let single_lanes = (0..64).map(|lane| 1_u64 << lane);
    let prefixes = (1..=64).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..64).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask8x64::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 64];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 64];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..64 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u8x64(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u8x64(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u8x64(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u8x64(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i8x16<S: Simd>(simd: S) {
    let values: [i8; 16] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i8
    });
    let merge: [i8; 16] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i8x16::simd_from(simd, values);
    let merge_vec = i8x16::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 16);
    let single_lanes = (0..16).map(|lane| 1_u64 << lane);
    let prefixes = (1..=16).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..16).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask8x16::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 16];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 16];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..16 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i8x16(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i8x16(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i8x16(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i8x16(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i8x32<S: Simd>(simd: S) {
    let values: [i8; 32] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i8
    });
    let merge: [i8; 32] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i8x32::simd_from(simd, values);
    let merge_vec = i8x32::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 32);
    let single_lanes = (0..32).map(|lane| 1_u64 << lane);
    let prefixes = (1..=32).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..32).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask8x32::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 32];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 32];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..32 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i8x32(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i8x32(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i8x32(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i8x32(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i8x64<S: Simd>(simd: S) {
    let values: [i8; 64] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i8
    });
    let merge: [i8; 64] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i8x64::simd_from(simd, values);
    let merge_vec = i8x64::simd_from(simd, merge);
    let all_lanes = u64::MAX;
    let single_lanes = (0..64).map(|lane| 1_u64 << lane);
    let prefixes = (1..=64).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..64).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask8x64::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 64];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 64];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..64 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i8x64(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i8x64(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i8x64(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i8x64(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u16x8<S: Simd>(simd: S) {
    let values: [u16; 8] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as u16
    });
    let merge: [u16; 8] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u16x8::simd_from(simd, values);
    let merge_vec = u16x8::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 8);
    let single_lanes = (0..8).map(|lane| 1_u64 << lane);
    let prefixes = (1..=8).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..8).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask16x8::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 8];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 8];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..8 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u16x8(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u16x8(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u16x8(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u16x8(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u16x16<S: Simd>(simd: S) {
    let values: [u16; 16] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as u16
    });
    let merge: [u16; 16] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u16x16::simd_from(simd, values);
    let merge_vec = u16x16::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 16);
    let single_lanes = (0..16).map(|lane| 1_u64 << lane);
    let prefixes = (1..=16).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..16).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask16x16::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 16];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 16];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..16 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u16x16(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u16x16(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u16x16(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u16x16(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u16x32<S: Simd>(simd: S) {
    let values: [u16; 32] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as u16
    });
    let merge: [u16; 32] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u16x32::simd_from(simd, values);
    let merge_vec = u16x32::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 32);
    let single_lanes = (0..32).map(|lane| 1_u64 << lane);
    let prefixes = (1..=32).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..32).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask16x32::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 32];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 32];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..32 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u16x32(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u16x32(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u16x32(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u16x32(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i16x8<S: Simd>(simd: S) {
    let values: [i16; 8] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i16
    });
    let merge: [i16; 8] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i16x8::simd_from(simd, values);
    let merge_vec = i16x8::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 8);
    let single_lanes = (0..8).map(|lane| 1_u64 << lane);
    let prefixes = (1..=8).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..8).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask16x8::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 8];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 8];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..8 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i16x8(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i16x8(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i16x8(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i16x8(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i16x16<S: Simd>(simd: S) {
    let values: [i16; 16] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i16
    });
    let merge: [i16; 16] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i16x16::simd_from(simd, values);
    let merge_vec = i16x16::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 16);
    let single_lanes = (0..16).map(|lane| 1_u64 << lane);
    let prefixes = (1..=16).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..16).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask16x16::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 16];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 16];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..16 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i16x16(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i16x16(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i16x16(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i16x16(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i16x32<S: Simd>(simd: S) {
    let values: [i16; 32] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i16
    });
    let merge: [i16; 32] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i16x32::simd_from(simd, values);
    let merge_vec = i16x32::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 32);
    let single_lanes = (0..32).map(|lane| 1_u64 << lane);
    let prefixes = (1..=32).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..32).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask16x32::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 32];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 32];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..32 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i16x32(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i16x32(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i16x32(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i16x32(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u32x4<S: Simd>(simd: S) {
    let values: [u32; 4] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as u32
    });
    let merge: [u32; 4] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u32x4::simd_from(simd, values);
    let merge_vec = u32x4::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 4);
    let single_lanes = (0..4).map(|lane| 1_u64 << lane);
    let prefixes = (1..=4).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..4).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask32x4::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 4];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 4];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..4 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u32x4(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u32x4(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u32x4(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u32x4(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u32x8<S: Simd>(simd: S) {
    let values: [u32; 8] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as u32
    });
    let merge: [u32; 8] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u32x8::simd_from(simd, values);
    let merge_vec = u32x8::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 8);
    let single_lanes = (0..8).map(|lane| 1_u64 << lane);
    let prefixes = (1..=8).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..8).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask32x8::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 8];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 8];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..8 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u32x8(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u32x8(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u32x8(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u32x8(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u32x16<S: Simd>(simd: S) {
    let values: [u32; 16] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as u32
    });
    let merge: [u32; 16] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u32x16::simd_from(simd, values);
    let merge_vec = u32x16::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 16);
    let single_lanes = (0..16).map(|lane| 1_u64 << lane);
    let prefixes = (1..=16).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..16).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask32x16::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 16];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 16];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..16 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u32x16(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u32x16(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u32x16(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u32x16(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i32x4<S: Simd>(simd: S) {
    let values: [i32; 4] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i32
    });
    let merge: [i32; 4] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i32x4::simd_from(simd, values);
    let merge_vec = i32x4::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 4);
    let single_lanes = (0..4).map(|lane| 1_u64 << lane);
    let prefixes = (1..=4).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..4).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask32x4::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 4];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 4];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..4 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i32x4(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i32x4(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i32x4(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i32x4(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i32x8<S: Simd>(simd: S) {
    let values: [i32; 8] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i32
    });
    let merge: [i32; 8] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i32x8::simd_from(simd, values);
    let merge_vec = i32x8::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 8);
    let single_lanes = (0..8).map(|lane| 1_u64 << lane);
    let prefixes = (1..=8).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..8).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask32x8::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 8];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 8];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..8 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i32x8(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i32x8(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i32x8(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i32x8(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i32x16<S: Simd>(simd: S) {
    let values: [i32; 16] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i32
    });
    let merge: [i32; 16] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i32x16::simd_from(simd, values);
    let merge_vec = i32x16::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 16);
    let single_lanes = (0..16).map(|lane| 1_u64 << lane);
    let prefixes = (1..=16).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..16).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask32x16::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 16];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 16];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..16 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i32x16(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i32x16(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i32x16(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i32x16(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u64x2<S: Simd>(simd: S) {
    let values: [u64; 2] =
        core::array::from_fn(|lane| 0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1));
    let merge: [u64; 2] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u64x2::simd_from(simd, values);
    let merge_vec = u64x2::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 2);
    let single_lanes = (0..2).map(|lane| 1_u64 << lane);
    let prefixes = (1..=2).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..2).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask64x2::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 2];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 2];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..2 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u64x2(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u64x2(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u64x2(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u64x2(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u64x4<S: Simd>(simd: S) {
    let values: [u64; 4] =
        core::array::from_fn(|lane| 0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1));
    let merge: [u64; 4] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u64x4::simd_from(simd, values);
    let merge_vec = u64x4::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 4);
    let single_lanes = (0..4).map(|lane| 1_u64 << lane);
    let prefixes = (1..=4).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..4).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask64x4::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 4];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 4];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..4 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u64x4(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u64x4(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u64x4(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u64x4(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_u64x8<S: Simd>(simd: S) {
    let values: [u64; 8] =
        core::array::from_fn(|lane| 0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1));
    let merge: [u64; 8] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = u64x8::simd_from(simd, values);
    let merge_vec = u64x8::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 8);
    let single_lanes = (0..8).map(|lane| 1_u64 << lane);
    let prefixes = (1..=8).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..8).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask64x8::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 8];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 8];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..8 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_u64x8(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_u64x8(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_u64x8(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_u64x8(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i64x2<S: Simd>(simd: S) {
    let values: [i64; 2] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i64
    });
    let merge: [i64; 2] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i64x2::simd_from(simd, values);
    let merge_vec = i64x2::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 2);
    let single_lanes = (0..2).map(|lane| 1_u64 << lane);
    let prefixes = (1..=2).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..2).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask64x2::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 2];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 2];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..2 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i64x2(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i64x2(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i64x2(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i64x2(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i64x4<S: Simd>(simd: S) {
    let values: [i64; 4] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i64
    });
    let merge: [i64; 4] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i64x4::simd_from(simd, values);
    let merge_vec = i64x4::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 4);
    let single_lanes = (0..4).map(|lane| 1_u64 << lane);
    let prefixes = (1..=4).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..4).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask64x4::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 4];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 4];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..4 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i64x4(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i64x4(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i64x4(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i64x4(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_i64x8<S: Simd>(simd: S) {
    let values: [i64; 8] = core::array::from_fn(|lane| {
        (0x91e3_a5c7_d9fb_2d4f_u64.wrapping_mul(lane as u64 + 1)) as i64
    });
    let merge: [i64; 8] = core::array::from_fn(|lane| !values[lane]);
    let values_vec = i64x8::simd_from(simd, values);
    let merge_vec = i64x8::simd_from(simd, merge);
    let all_lanes = u64::MAX >> (64 - 8);
    let single_lanes = (0..8).map(|lane| 1_u64 << lane);
    let prefixes = (1..=8).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..8).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask64x8::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 8];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 8];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..8 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected_compress);
        assert_eq!(*simd.compress_i64x8(values_vec, mask), expected_compress);
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(
            *simd.compress_merge_i64x8(values_vec, mask, merge_vec),
            expected_compress_merge
        );
        assert_eq!(*values_vec.expand(mask), expected_expand);
        assert_eq!(*simd.expand_i64x8(values_vec, mask), expected_expand);
        assert_eq!(
            *values_vec.expand_merge(mask, merge_vec),
            expected_expand_merge
        );
        assert_eq!(
            *simd.expand_merge_i64x8(values_vec, mask, merge_vec),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_f32x4<S: Simd>(simd: S) {
    let patterns: [u32; 8] = [
        0x8000_0000,
        0x7fc1_2345,
        0x7f80_0000,
        0x7f80_0001,
        0,
        0xff80_0000,
        0xffc5_4321,
        0x3f80_0000,
    ];
    let values: [u32; 4] = core::array::from_fn(|lane| patterns[lane % 8]);
    let merge: [u32; 4] = core::array::from_fn(|lane| patterns[(lane + 3) % 8]);
    let values_vec = f32x4::simd_from(simd, values.map(f32::from_bits));
    let merge_vec = f32x4::simd_from(simd, merge.map(f32::from_bits));
    let all_lanes = u64::MAX >> (64 - 4);
    let single_lanes = (0..4).map(|lane| 1_u64 << lane);
    let prefixes = (1..=4).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..4).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask32x4::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 4];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 4];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..4 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(
            values_vec.compress(mask).to_array().map(f32::to_bits),
            expected_compress
        );
        assert_eq!(
            simd.compress_f32x4(values_vec, mask)
                .to_array()
                .map(f32::to_bits),
            expected_compress
        );
        assert_eq!(
            values_vec
                .compress_merge(mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            simd.compress_merge_f32x4(values_vec, mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            values_vec.expand(mask).to_array().map(f32::to_bits),
            expected_expand
        );
        assert_eq!(
            simd.expand_f32x4(values_vec, mask)
                .to_array()
                .map(f32::to_bits),
            expected_expand
        );
        assert_eq!(
            values_vec
                .expand_merge(mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_expand_merge
        );
        assert_eq!(
            simd.expand_merge_f32x4(values_vec, mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_f32x8<S: Simd>(simd: S) {
    let patterns: [u32; 8] = [
        0x8000_0000,
        0x7fc1_2345,
        0x7f80_0000,
        0x7f80_0001,
        0,
        0xff80_0000,
        0xffc5_4321,
        0x3f80_0000,
    ];
    let values: [u32; 8] = core::array::from_fn(|lane| patterns[lane % 8]);
    let merge: [u32; 8] = core::array::from_fn(|lane| patterns[(lane + 3) % 8]);
    let values_vec = f32x8::simd_from(simd, values.map(f32::from_bits));
    let merge_vec = f32x8::simd_from(simd, merge.map(f32::from_bits));
    let all_lanes = u64::MAX >> (64 - 8);
    let single_lanes = (0..8).map(|lane| 1_u64 << lane);
    let prefixes = (1..=8).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..8).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask32x8::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 8];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 8];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..8 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(
            values_vec.compress(mask).to_array().map(f32::to_bits),
            expected_compress
        );
        assert_eq!(
            simd.compress_f32x8(values_vec, mask)
                .to_array()
                .map(f32::to_bits),
            expected_compress
        );
        assert_eq!(
            values_vec
                .compress_merge(mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            simd.compress_merge_f32x8(values_vec, mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            values_vec.expand(mask).to_array().map(f32::to_bits),
            expected_expand
        );
        assert_eq!(
            simd.expand_f32x8(values_vec, mask)
                .to_array()
                .map(f32::to_bits),
            expected_expand
        );
        assert_eq!(
            values_vec
                .expand_merge(mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_expand_merge
        );
        assert_eq!(
            simd.expand_merge_f32x8(values_vec, mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_f32x16<S: Simd>(simd: S) {
    let patterns: [u32; 8] = [
        0x8000_0000,
        0x7fc1_2345,
        0x7f80_0000,
        0x7f80_0001,
        0,
        0xff80_0000,
        0xffc5_4321,
        0x3f80_0000,
    ];
    let values: [u32; 16] = core::array::from_fn(|lane| patterns[lane % 8]);
    let merge: [u32; 16] = core::array::from_fn(|lane| patterns[(lane + 3) % 8]);
    let values_vec = f32x16::simd_from(simd, values.map(f32::from_bits));
    let merge_vec = f32x16::simd_from(simd, merge.map(f32::from_bits));
    let all_lanes = u64::MAX >> (64 - 16);
    let single_lanes = (0..16).map(|lane| 1_u64 << lane);
    let prefixes = (1..=16).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..16).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask32x16::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 16];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 16];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..16 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(
            values_vec.compress(mask).to_array().map(f32::to_bits),
            expected_compress
        );
        assert_eq!(
            simd.compress_f32x16(values_vec, mask)
                .to_array()
                .map(f32::to_bits),
            expected_compress
        );
        assert_eq!(
            values_vec
                .compress_merge(mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            simd.compress_merge_f32x16(values_vec, mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            values_vec.expand(mask).to_array().map(f32::to_bits),
            expected_expand
        );
        assert_eq!(
            simd.expand_f32x16(values_vec, mask)
                .to_array()
                .map(f32::to_bits),
            expected_expand
        );
        assert_eq!(
            values_vec
                .expand_merge(mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_expand_merge
        );
        assert_eq!(
            simd.expand_merge_f32x16(values_vec, mask, merge_vec)
                .to_array()
                .map(f32::to_bits),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_f64x2<S: Simd>(simd: S) {
    let patterns: [u64; 8] = [
        0x8000_0000_0000_0000,
        0x7ff8_1234_5678_9abc,
        0x7ff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0,
        0xfff0_0000_0000_0000,
        0xfff8_abcd_1234_5678,
        0x3ff0_0000_0000_0000,
    ];
    let values: [u64; 2] = core::array::from_fn(|lane| patterns[lane % 8]);
    let merge: [u64; 2] = core::array::from_fn(|lane| patterns[(lane + 3) % 8]);
    let values_vec = f64x2::simd_from(simd, values.map(f64::from_bits));
    let merge_vec = f64x2::simd_from(simd, merge.map(f64::from_bits));
    let all_lanes = u64::MAX >> (64 - 2);
    let single_lanes = (0..2).map(|lane| 1_u64 << lane);
    let prefixes = (1..=2).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..2).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask64x2::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 2];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 2];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..2 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(
            values_vec.compress(mask).to_array().map(f64::to_bits),
            expected_compress
        );
        assert_eq!(
            simd.compress_f64x2(values_vec, mask)
                .to_array()
                .map(f64::to_bits),
            expected_compress
        );
        assert_eq!(
            values_vec
                .compress_merge(mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            simd.compress_merge_f64x2(values_vec, mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            values_vec.expand(mask).to_array().map(f64::to_bits),
            expected_expand
        );
        assert_eq!(
            simd.expand_f64x2(values_vec, mask)
                .to_array()
                .map(f64::to_bits),
            expected_expand
        );
        assert_eq!(
            values_vec
                .expand_merge(mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_expand_merge
        );
        assert_eq!(
            simd.expand_merge_f64x2(values_vec, mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_f64x4<S: Simd>(simd: S) {
    let patterns: [u64; 8] = [
        0x8000_0000_0000_0000,
        0x7ff8_1234_5678_9abc,
        0x7ff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0,
        0xfff0_0000_0000_0000,
        0xfff8_abcd_1234_5678,
        0x3ff0_0000_0000_0000,
    ];
    let values: [u64; 4] = core::array::from_fn(|lane| patterns[lane % 8]);
    let merge: [u64; 4] = core::array::from_fn(|lane| patterns[(lane + 3) % 8]);
    let values_vec = f64x4::simd_from(simd, values.map(f64::from_bits));
    let merge_vec = f64x4::simd_from(simd, merge.map(f64::from_bits));
    let all_lanes = u64::MAX >> (64 - 4);
    let single_lanes = (0..4).map(|lane| 1_u64 << lane);
    let prefixes = (1..=4).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..4).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    .chain(0..=all_lanes)
    {
        let mask = mask64x4::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 4];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 4];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..4 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(
            values_vec.compress(mask).to_array().map(f64::to_bits),
            expected_compress
        );
        assert_eq!(
            simd.compress_f64x4(values_vec, mask)
                .to_array()
                .map(f64::to_bits),
            expected_compress
        );
        assert_eq!(
            values_vec
                .compress_merge(mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            simd.compress_merge_f64x4(values_vec, mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            values_vec.expand(mask).to_array().map(f64::to_bits),
            expected_expand
        );
        assert_eq!(
            simd.expand_f64x4(values_vec, mask)
                .to_array()
                .map(f64::to_bits),
            expected_expand
        );
        assert_eq!(
            values_vec
                .expand_merge(mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_expand_merge
        );
        assert_eq!(
            simd.expand_merge_f64x4(values_vec, mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_f64x8<S: Simd>(simd: S) {
    let patterns: [u64; 8] = [
        0x8000_0000_0000_0000,
        0x7ff8_1234_5678_9abc,
        0x7ff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0,
        0xfff0_0000_0000_0000,
        0xfff8_abcd_1234_5678,
        0x3ff0_0000_0000_0000,
    ];
    let values: [u64; 8] = core::array::from_fn(|lane| patterns[lane % 8]);
    let merge: [u64; 8] = core::array::from_fn(|lane| patterns[(lane + 3) % 8]);
    let values_vec = f64x8::simd_from(simd, values.map(f64::from_bits));
    let merge_vec = f64x8::simd_from(simd, merge.map(f64::from_bits));
    let all_lanes = u64::MAX >> (64 - 8);
    let single_lanes = (0..8).map(|lane| 1_u64 << lane);
    let prefixes = (1..=8).map(|count| u64::MAX >> (64 - count));
    let suffixes = (0..8).map(|lane| all_lanes << lane);
    for mask_bits in [
        0,
        all_lanes,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x936c_a5f0_817e_42bd,
    ]
    .into_iter()
    .chain(single_lanes)
    .chain(prefixes)
    .chain(suffixes)
    {
        let mask = mask64x8::from_bitmask(simd, mask_bits);
        let mut expected_compress = [0; 8];
        let mut expected_compress_merge = merge;
        let mut expected_expand = [0; 8];
        let mut expected_expand_merge = merge;
        let mut selected = 0;
        for lane in 0..8 {
            if mask_bits & (1_u64 << lane) != 0 {
                expected_compress[selected] = values[lane];
                expected_compress_merge[selected] = values[lane];
                expected_expand[lane] = values[selected];
                expected_expand_merge[lane] = values[selected];
                selected += 1;
            }
        }
        assert_eq!(
            values_vec.compress(mask).to_array().map(f64::to_bits),
            expected_compress
        );
        assert_eq!(
            simd.compress_f64x8(values_vec, mask)
                .to_array()
                .map(f64::to_bits),
            expected_compress
        );
        assert_eq!(
            values_vec
                .compress_merge(mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            simd.compress_merge_f64x8(values_vec, mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_compress_merge
        );
        assert_eq!(
            values_vec.expand(mask).to_array().map(f64::to_bits),
            expected_expand
        );
        assert_eq!(
            simd.expand_f64x8(values_vec, mask)
                .to_array()
                .map(f64::to_bits),
            expected_expand
        );
        assert_eq!(
            values_vec
                .expand_merge(mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_expand_merge
        );
        assert_eq!(
            simd.expand_merge_f64x8(values_vec, mask, merge_vec)
                .to_array()
                .map(f64::to_bits),
            expected_expand_merge
        );
    }
}

#[simd_test]
fn compact_generic_base<S: Simd>(simd: S) {
    fn compact<S: Simd, V: SimdBase<S>>(value: V, mask: V::Mask, merge: V::Element) -> [V; 4] {
        [
            value.compress(mask),
            value.compress_merge(mask, merge),
            value.expand(mask),
            value.expand_merge(mask, merge),
        ]
    }
    let values = u32x4::simd_from(simd, [10, 20, 30, 40]);
    let mask = mask32x4::from_bitmask(simd, 0b1010);
    let results = compact(values, mask, 99);
    assert_eq!(*results[0], [20, 40, 0, 0]);
    assert_eq!(*results[1], [20, 40, 99, 99]);
    assert_eq!(*results[2], [0, 10, 0, 20]);
    assert_eq!(*results[3], [99, 10, 99, 20]);
}

#[simd_test]
fn compress_all_widths<S: Simd>(simd: S) {
    macro_rules! check_width {
        ($bytes:ident, $mask:ident, $lanes:literal, $compress:ident, $compress_merge:ident) => {{
            let values: [u8; $lanes] = core::array::from_fn(|lane| (lane * 3 + 1) as u8);
            let merge: [u8; $lanes] = core::array::from_fn(|lane| 0xe0_u8.wrapping_add(lane as u8));
            let patterned_mask: u64 = (0..$lanes)
                .filter(|lane| lane % 3 == 0 || lane % 7 == 2)
                .fold(0, |mask, lane| mask | (1_u64 << lane));
            let values_vec = $bytes::simd_from(simd, values);
            let merge_vec = $bytes::simd_from(simd, merge);
            let all_lanes = u64::MAX >> (64 - $lanes);

            let repeated_byte_masks = (0_u64..=255).map(|byte_mask| {
                (0..($lanes / 8)).fold(0, |mask, block| mask | (byte_mask << (block * 8)))
            });
            let stitch_masks = ($lanes == 64).then(|| {
                (0..=32).map(|low_count| {
                    let low = if low_count == 32 {
                        u64::from(u32::MAX)
                    } else {
                        (1_u64 << low_count) - 1
                    };
                    low | (0xa5a5_a5a5_u64 << 32)
                })
            });
            for mask_bits in [0, 1, patterned_mask, all_lanes]
                .into_iter()
                .chain(repeated_byte_masks)
                .chain(stitch_masks.into_iter().flatten())
            {
                let mask = $mask::from_bitmask(simd, mask_bits);
                let mut expected = [0; $lanes];
                let mut expected_merge = merge;
                let mut output_lane = 0;
                for (input_lane, value) in values.into_iter().enumerate() {
                    if mask_bits & (1_u64 << input_lane) != 0 {
                        expected[output_lane] = value;
                        expected_merge[output_lane] = value;
                        output_lane += 1;
                    }
                }

                assert_eq!(*simd.$compress(values_vec, mask), expected);
                assert_eq!(
                    *simd.$compress_merge(values_vec, mask, merge_vec),
                    expected_merge
                );
            }
        }};
    }

    check_width!(u8x16, mask8x16, 16, compress_u8x16, compress_merge_u8x16);
    check_width!(u8x32, mask8x32, 32, compress_u8x32, compress_merge_u8x32);
    check_width!(u8x64, mask8x64, 64, compress_u8x64, compress_merge_u8x64);
}

#[simd_test]
fn compress_u8x16_exhaustive_masks<S: Simd>(simd: S) {
    let values = [
        0, 255, 128, 1, 127, 254, 2, 253, 3, 252, 4, 251, 5, 250, 6, 249,
    ];
    let merge = [
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    ];
    let values_vec = u8x16::simd_from(simd, values);
    let merge_vec = u8x16::simd_from(simd, merge);
    for bits in 0..=u16::MAX {
        let mask = mask8x16::from_bitmask(simd, u64::from(bits));
        let mut expected = [0; 16];
        let mut expected_merge = merge;
        let mut selected = 0;
        for (lane, value) in values.into_iter().enumerate() {
            if bits & (1 << lane) != 0 {
                expected[selected] = value;
                expected_merge[selected] = value;
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected, "mask {bits:#06x}");
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_merge,
            "mask {bits:#06x}"
        );
    }
}

#[simd_test]
fn compress_u32x8_exhaustive_masks<S: Simd>(simd: S) {
    let values = [
        0,
        u32::MAX,
        0x8000_0000,
        1,
        0x7fff_ffff,
        0x1234_5678,
        0xfedc_ba98,
        0x0102_0304,
    ];
    let merge = [
        0xa0b0_c001,
        0xa0b0_c002,
        0xa0b0_c003,
        0xa0b0_c004,
        0xa0b0_c005,
        0xa0b0_c006,
        0xa0b0_c007,
        0xa0b0_c008,
    ];
    let values_vec = u32x8::simd_from(simd, values);
    let merge_vec = u32x8::simd_from(simd, merge);
    for bits in 0..=u8::MAX {
        let mask = mask32x8::from_bitmask(simd, u64::from(bits));
        let mut expected = [0; 8];
        let mut expected_merge = merge;
        let mut selected = 0;
        for (lane, value) in values.into_iter().enumerate() {
            if bits & (1 << lane) != 0 {
                expected[selected] = value;
                expected_merge[selected] = value;
                selected += 1;
            }
        }
        assert_eq!(*values_vec.compress(mask), expected, "mask {bits:#04x}");
        assert_eq!(
            *values_vec.compress_merge(mask, merge_vec),
            expected_merge,
            "mask {bits:#04x}"
        );
    }
}
