// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// One concrete test row per supported vector type.
// Randomized widening and narrowing coverage lives in `narrow.rs`.

#[simd_test]
fn widen_u8x16<S: Simd>(simd: S) {
    let input = u8x16::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(*low, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(*high, [248, 249, 250, 251, 252, 253, 254, 255]);
}

#[simd_test]
fn widen_u8x32<S: Simd>(simd: S) {
    let input = u8x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 240, 241, 242, 243, 244, 245,
            246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(*low, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(
        *high,
        [
            240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
        ]
    );
}

#[simd_test]
fn widen_u8x64<S: Simd>(simd: S) {
    let input = u8x64::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
            235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
            252, 253, 254, 255,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(
        *low,
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
        ]
    );
    assert_eq!(
        *high,
        [
            224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
            241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
        ]
    );
}

#[simd_test]
fn widen_i8x16<S: Simd>(simd: S) {
    let input = i8x16::from_slice(
        simd,
        &[
            -128, -127, -126, -125, -124, -123, -122, -121, 120, 121, 122, 123, 124, 125, 126, 127,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(*low, [-128, -127, -126, -125, -124, -123, -122, -121]);
    assert_eq!(*high, [120, 121, 122, 123, 124, 125, 126, 127]);
}

#[simd_test]
fn widen_i8x32<S: Simd>(simd: S) {
    let input = i8x32::from_slice(
        simd,
        &[
            -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115,
            -114, -113, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
            127,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(
        *low,
        [
            -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115,
            -114, -113
        ]
    );
    assert_eq!(
        *high,
        [
            112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
        ]
    );
}

#[simd_test]
fn widen_i8x64<S: Simd>(simd: S) {
    let input = i8x64::from_slice(
        simd,
        &[
            -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115,
            -114, -113, -112, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102, -101,
            -100, -99, -98, -97, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
            127,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(
        *low,
        [
            -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115,
            -114, -113, -112, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102, -101,
            -100, -99, -98, -97
        ]
    );
    assert_eq!(
        *high,
        [
            96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
            114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
        ]
    );
}

#[simd_test]
fn widen_u16x8<S: Simd>(simd: S) {
    let input = u16x8::from_slice(simd, &[0, 1, 2, 3, 65_532, 65_533, 65_534, 65_535]);
    let (low, high) = input.widen();

    assert_eq!(*low, [0, 1, 2, 3]);
    assert_eq!(*high, [65_532, 65_533, 65_534, 65_535]);
}

#[simd_test]
fn widen_u16x16<S: Simd>(simd: S) {
    let input = u16x16::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 65_528, 65_529, 65_530, 65_531, 65_532, 65_533, 65_534, 65_535,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(*low, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(
        *high,
        [
            65_528, 65_529, 65_530, 65_531, 65_532, 65_533, 65_534, 65_535
        ]
    );
}

#[simd_test]
fn widen_u16x32<S: Simd>(simd: S) {
    let input = u16x32::from_slice(
        simd,
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 65_520, 65_521, 65_522, 65_523,
            65_524, 65_525, 65_526, 65_527, 65_528, 65_529, 65_530, 65_531, 65_532, 65_533, 65_534,
            65_535,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(*low, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(
        *high,
        [
            65_520, 65_521, 65_522, 65_523, 65_524, 65_525, 65_526, 65_527, 65_528, 65_529, 65_530,
            65_531, 65_532, 65_533, 65_534, 65_535
        ]
    );
}

#[simd_test]
fn widen_i16x8<S: Simd>(simd: S) {
    let input = i16x8::from_slice(
        simd,
        &[
            -32_768, -32_767, -32_766, -32_765, 32_764, 32_765, 32_766, 32_767,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(*low, [-32_768, -32_767, -32_766, -32_765]);
    assert_eq!(*high, [32_764, 32_765, 32_766, 32_767]);
}

#[simd_test]
fn widen_i16x16<S: Simd>(simd: S) {
    let input = i16x16::from_slice(
        simd,
        &[
            -32_768, -32_767, -32_766, -32_765, -32_764, -32_763, -32_762, -32_761, 32_760, 32_761,
            32_762, 32_763, 32_764, 32_765, 32_766, 32_767,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(
        *low,
        [
            -32_768, -32_767, -32_766, -32_765, -32_764, -32_763, -32_762, -32_761
        ]
    );
    assert_eq!(
        *high,
        [
            32_760, 32_761, 32_762, 32_763, 32_764, 32_765, 32_766, 32_767
        ]
    );
}

#[simd_test]
fn widen_i16x32<S: Simd>(simd: S) {
    let input = i16x32::from_slice(
        simd,
        &[
            -32_768, -32_767, -32_766, -32_765, -32_764, -32_763, -32_762, -32_761, -32_760,
            -32_759, -32_758, -32_757, -32_756, -32_755, -32_754, -32_753, 32_752, 32_753, 32_754,
            32_755, 32_756, 32_757, 32_758, 32_759, 32_760, 32_761, 32_762, 32_763, 32_764, 32_765,
            32_766, 32_767,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(
        *low,
        [
            -32_768, -32_767, -32_766, -32_765, -32_764, -32_763, -32_762, -32_761, -32_760,
            -32_759, -32_758, -32_757, -32_756, -32_755, -32_754, -32_753
        ]
    );
    assert_eq!(
        *high,
        [
            32_752, 32_753, 32_754, 32_755, 32_756, 32_757, 32_758, 32_759, 32_760, 32_761, 32_762,
            32_763, 32_764, 32_765, 32_766, 32_767
        ]
    );
}

#[simd_test]
fn widen_u32x4<S: Simd>(simd: S) {
    let input = u32x4::from_slice(simd, &[0, 2_147_483_647, 2_147_483_648, 4_294_967_295]);
    let (low, high) = input.widen();

    assert_eq!(*low, [0, 2_147_483_647]);
    assert_eq!(*high, [2_147_483_648, 4_294_967_295]);
}

#[simd_test]
fn widen_u32x8<S: Simd>(simd: S) {
    let input = u32x8::from_slice(
        simd,
        &[
            0,
            1,
            2,
            3,
            4_294_967_292,
            4_294_967_293,
            4_294_967_294,
            4_294_967_295,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(*low, [0, 1, 2, 3]);
    assert_eq!(
        *high,
        [4_294_967_292, 4_294_967_293, 4_294_967_294, 4_294_967_295]
    );
}

#[simd_test]
fn widen_u32x16<S: Simd>(simd: S) {
    let input = u32x16::from_slice(
        simd,
        &[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            4_294_967_288,
            4_294_967_289,
            4_294_967_290,
            4_294_967_291,
            4_294_967_292,
            4_294_967_293,
            4_294_967_294,
            4_294_967_295,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(*low, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(
        *high,
        [
            4_294_967_288,
            4_294_967_289,
            4_294_967_290,
            4_294_967_291,
            4_294_967_292,
            4_294_967_293,
            4_294_967_294,
            4_294_967_295
        ]
    );
}

#[simd_test]
fn widen_i32x4<S: Simd>(simd: S) {
    let input = i32x4::from_slice(simd, &[-2_147_483_648, -1, 0, 2_147_483_647]);
    let (low, high) = input.widen();

    assert_eq!(*low, [-2_147_483_648, -1]);
    assert_eq!(*high, [0, 2_147_483_647]);
}

#[simd_test]
fn widen_i32x8<S: Simd>(simd: S) {
    let input = i32x8::from_slice(
        simd,
        &[
            -2_147_483_648,
            -2_147_483_647,
            -2_147_483_646,
            -2_147_483_645,
            2_147_483_644,
            2_147_483_645,
            2_147_483_646,
            2_147_483_647,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(
        *low,
        [
            -2_147_483_648,
            -2_147_483_647,
            -2_147_483_646,
            -2_147_483_645
        ]
    );
    assert_eq!(
        *high,
        [2_147_483_644, 2_147_483_645, 2_147_483_646, 2_147_483_647]
    );
}

#[simd_test]
fn widen_i32x16<S: Simd>(simd: S) {
    let input = i32x16::from_slice(
        simd,
        &[
            -2_147_483_648,
            -2_147_483_647,
            -2_147_483_646,
            -2_147_483_645,
            -2_147_483_644,
            -2_147_483_643,
            -2_147_483_642,
            -2_147_483_641,
            2_147_483_640,
            2_147_483_641,
            2_147_483_642,
            2_147_483_643,
            2_147_483_644,
            2_147_483_645,
            2_147_483_646,
            2_147_483_647,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(
        *low,
        [
            -2_147_483_648,
            -2_147_483_647,
            -2_147_483_646,
            -2_147_483_645,
            -2_147_483_644,
            -2_147_483_643,
            -2_147_483_642,
            -2_147_483_641
        ]
    );
    assert_eq!(
        *high,
        [
            2_147_483_640,
            2_147_483_641,
            2_147_483_642,
            2_147_483_643,
            2_147_483_644,
            2_147_483_645,
            2_147_483_646,
            2_147_483_647
        ]
    );
}

#[simd_test]
fn widen_f32x4<S: Simd>(simd: S) {
    let input = f32x4::from_slice(simd, &[-2.25, -0.5, 0.5, 1.5]);
    let (low, high) = input.widen();

    assert_eq!(*low, [-2.25, -0.5]);
    assert_eq!(*high, [0.5, 1.5]);
}

#[simd_test]
fn widen_f32x8<S: Simd>(simd: S) {
    let input = f32x8::from_slice(
        simd,
        &[-1_000.0, -42.5, -2.25, -0.5, 0.5, 1.5, 42.5, 1_000.0],
    );
    let (low, high) = input.widen();

    assert_eq!(*low, [-1_000.0, -42.5, -2.25, -0.5]);
    assert_eq!(*high, [0.5, 1.5, 42.5, 1_000.0]);
}

#[simd_test]
fn widen_f32x16<S: Simd>(simd: S) {
    let input = f32x16::from_slice(
        simd,
        &[
            -65_536.0, -1_000.0, -42.5, -2.25, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.25,
            42.5, 1_000.0, 65_536.0,
        ],
    );
    let (low, high) = input.widen();

    assert_eq!(
        *low,
        [-65_536.0, -1_000.0, -42.5, -2.25, -1.5, -1.0, -0.5, -0.25]
    );
    assert_eq!(*high, [0.25, 0.5, 1.0, 1.5, 2.25, 42.5, 1_000.0, 65_536.0]);
}

#[simd_test]
fn widen_mask8x16<S: Simd>(simd: S) {
    for bits in [0, u64::MAX, 0xd36a_59c2_8e17_b4a5, 0x8181] {
        let input = mask8x16::from_bitmask(simd, bits);
        let (low, high): (mask16x8<S>, mask16x8<S>) = input.widen();
        let low_bits = bits & 0xff;
        let high_bits = (bits >> 8) & 0xff;
        assert_eq!(low.to_bitmask(), low_bits);
        assert_eq!(high.to_bitmask(), high_bits);
        assert_eq!(
            <[i16; 8]>::from(low),
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        assert_eq!(
            <[i16; 8]>::from(high),
            core::array::from_fn(|i| if high_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        let selected = low.select(i16x8::splat(simd, 7), i16x8::splat(simd, 3));
        assert_eq!(
            *selected,
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { 7 } else { 3 }),
        );
        assert_eq!(low.narrow(high).to_bitmask(), bits & 0xffff);
    }

    for lane in 0..16 {
        let input = mask8x16::from_bitmask(simd, 1_u64 << lane);
        let (low, high) = input.widen();
        assert_eq!(low.to_bitmask(), (1_u64 << lane) & 0xff);
        assert_eq!(high.to_bitmask(), (1_u64 << lane) >> 8);
    }
}

#[simd_test]
fn widen_mask16x8<S: Simd>(simd: S) {
    for bits in [0, u64::MAX, 0xd36a_59c2_8e17_b4a5, 0x99] {
        let input = mask16x8::from_bitmask(simd, bits);
        let (low, high): (mask32x4<S>, mask32x4<S>) = input.widen();
        let low_bits = bits & 0xf;
        let high_bits = (bits >> 4) & 0xf;
        assert_eq!(low.to_bitmask(), low_bits);
        assert_eq!(high.to_bitmask(), high_bits);
        assert_eq!(
            <[i32; 4]>::from(low),
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        assert_eq!(
            <[i32; 4]>::from(high),
            core::array::from_fn(|i| if high_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        let selected = low.select(i32x4::splat(simd, 7), i32x4::splat(simd, 3));
        assert_eq!(
            *selected,
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { 7 } else { 3 }),
        );
        assert_eq!(low.narrow(high).to_bitmask(), bits & 0xff);
    }

    for lane in 0..8 {
        let input = mask16x8::from_bitmask(simd, 1_u64 << lane);
        let (low, high) = input.widen();
        assert_eq!(low.to_bitmask(), (1_u64 << lane) & 0xf);
        assert_eq!(high.to_bitmask(), (1_u64 << lane) >> 4);
    }
}

#[simd_test]
fn widen_mask32x4<S: Simd>(simd: S) {
    for bits in [0, u64::MAX, 0xd36a_59c2_8e17_b4a5, 0xf] {
        let input = mask32x4::from_bitmask(simd, bits);
        let (low, high): (mask64x2<S>, mask64x2<S>) = input.widen();
        let low_bits = bits & 0x3;
        let high_bits = (bits >> 2) & 0x3;
        assert_eq!(low.to_bitmask(), low_bits);
        assert_eq!(high.to_bitmask(), high_bits);
        assert_eq!(
            <[i64; 2]>::from(low),
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        assert_eq!(
            <[i64; 2]>::from(high),
            core::array::from_fn(|i| if high_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        let selected = low.select(i64x2::splat(simd, 7), i64x2::splat(simd, 3));
        assert_eq!(
            *selected,
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { 7 } else { 3 }),
        );
        assert_eq!(low.narrow(high).to_bitmask(), bits & 0xf);
    }

    for lane in 0..4 {
        let input = mask32x4::from_bitmask(simd, 1_u64 << lane);
        let (low, high) = input.widen();
        assert_eq!(low.to_bitmask(), (1_u64 << lane) & 0x3);
        assert_eq!(high.to_bitmask(), (1_u64 << lane) >> 2);
    }
}

#[simd_test]
fn widen_mask8x32<S: Simd>(simd: S) {
    for bits in [0, u64::MAX, 0xd36a_59c2_8e17_b4a5, 0x80018001] {
        let input = mask8x32::from_bitmask(simd, bits);
        let (low, high): (mask16x16<S>, mask16x16<S>) = input.widen();
        let low_bits = bits & 0xffff;
        let high_bits = (bits >> 16) & 0xffff;
        assert_eq!(low.to_bitmask(), low_bits);
        assert_eq!(high.to_bitmask(), high_bits);
        assert_eq!(
            <[i16; 16]>::from(low),
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        assert_eq!(
            <[i16; 16]>::from(high),
            core::array::from_fn(|i| if high_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        let selected = low.select(i16x16::splat(simd, 7), i16x16::splat(simd, 3));
        assert_eq!(
            *selected,
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { 7 } else { 3 }),
        );
        assert_eq!(low.narrow(high).to_bitmask(), bits & 0xffffffff);
    }

    for lane in 0..32 {
        let input = mask8x32::from_bitmask(simd, 1_u64 << lane);
        let (low, high) = input.widen();
        assert_eq!(low.to_bitmask(), (1_u64 << lane) & 0xffff);
        assert_eq!(high.to_bitmask(), (1_u64 << lane) >> 16);
    }
}

#[simd_test]
fn widen_mask16x16<S: Simd>(simd: S) {
    for bits in [0, u64::MAX, 0xd36a_59c2_8e17_b4a5, 0x8181] {
        let input = mask16x16::from_bitmask(simd, bits);
        let (low, high): (mask32x8<S>, mask32x8<S>) = input.widen();
        let low_bits = bits & 0xff;
        let high_bits = (bits >> 8) & 0xff;
        assert_eq!(low.to_bitmask(), low_bits);
        assert_eq!(high.to_bitmask(), high_bits);
        assert_eq!(
            <[i32; 8]>::from(low),
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        assert_eq!(
            <[i32; 8]>::from(high),
            core::array::from_fn(|i| if high_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        let selected = low.select(i32x8::splat(simd, 7), i32x8::splat(simd, 3));
        assert_eq!(
            *selected,
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { 7 } else { 3 }),
        );
        assert_eq!(low.narrow(high).to_bitmask(), bits & 0xffff);
    }

    for lane in 0..16 {
        let input = mask16x16::from_bitmask(simd, 1_u64 << lane);
        let (low, high) = input.widen();
        assert_eq!(low.to_bitmask(), (1_u64 << lane) & 0xff);
        assert_eq!(high.to_bitmask(), (1_u64 << lane) >> 8);
    }
}

#[simd_test]
fn widen_mask32x8<S: Simd>(simd: S) {
    for bits in [0, u64::MAX, 0xd36a_59c2_8e17_b4a5, 0x99] {
        let input = mask32x8::from_bitmask(simd, bits);
        let (low, high): (mask64x4<S>, mask64x4<S>) = input.widen();
        let low_bits = bits & 0xf;
        let high_bits = (bits >> 4) & 0xf;
        assert_eq!(low.to_bitmask(), low_bits);
        assert_eq!(high.to_bitmask(), high_bits);
        assert_eq!(
            <[i64; 4]>::from(low),
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        assert_eq!(
            <[i64; 4]>::from(high),
            core::array::from_fn(|i| if high_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        let selected = low.select(i64x4::splat(simd, 7), i64x4::splat(simd, 3));
        assert_eq!(
            *selected,
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { 7 } else { 3 }),
        );
        assert_eq!(low.narrow(high).to_bitmask(), bits & 0xff);
    }

    for lane in 0..8 {
        let input = mask32x8::from_bitmask(simd, 1_u64 << lane);
        let (low, high) = input.widen();
        assert_eq!(low.to_bitmask(), (1_u64 << lane) & 0xf);
        assert_eq!(high.to_bitmask(), (1_u64 << lane) >> 4);
    }
}

#[simd_test]
fn widen_mask8x64<S: Simd>(simd: S) {
    for bits in [0, u64::MAX, 0xd36a_59c2_8e17_b4a5, 0x8000000180000001] {
        let input = mask8x64::from_bitmask(simd, bits);
        let (low, high): (mask16x32<S>, mask16x32<S>) = input.widen();
        let low_bits = bits & 0xffffffff;
        let high_bits = (bits >> 32) & 0xffffffff;
        assert_eq!(low.to_bitmask(), low_bits);
        assert_eq!(high.to_bitmask(), high_bits);
        assert_eq!(
            <[i16; 32]>::from(low),
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        assert_eq!(
            <[i16; 32]>::from(high),
            core::array::from_fn(|i| if high_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        let selected = low.select(i16x32::splat(simd, 7), i16x32::splat(simd, 3));
        assert_eq!(
            *selected,
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { 7 } else { 3 }),
        );
        assert_eq!(low.narrow(high).to_bitmask(), bits & 0xffffffffffffffff);
    }

    for lane in 0..64 {
        let input = mask8x64::from_bitmask(simd, 1_u64 << lane);
        let (low, high) = input.widen();
        assert_eq!(low.to_bitmask(), (1_u64 << lane) & 0xffffffff);
        assert_eq!(high.to_bitmask(), (1_u64 << lane) >> 32);
    }
}

#[simd_test]
fn widen_mask16x32<S: Simd>(simd: S) {
    for bits in [0, u64::MAX, 0xd36a_59c2_8e17_b4a5, 0x80018001] {
        let input = mask16x32::from_bitmask(simd, bits);
        let (low, high): (mask32x16<S>, mask32x16<S>) = input.widen();
        let low_bits = bits & 0xffff;
        let high_bits = (bits >> 16) & 0xffff;
        assert_eq!(low.to_bitmask(), low_bits);
        assert_eq!(high.to_bitmask(), high_bits);
        assert_eq!(
            <[i32; 16]>::from(low),
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        assert_eq!(
            <[i32; 16]>::from(high),
            core::array::from_fn(|i| if high_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        let selected = low.select(i32x16::splat(simd, 7), i32x16::splat(simd, 3));
        assert_eq!(
            *selected,
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { 7 } else { 3 }),
        );
        assert_eq!(low.narrow(high).to_bitmask(), bits & 0xffffffff);
    }

    for lane in 0..32 {
        let input = mask16x32::from_bitmask(simd, 1_u64 << lane);
        let (low, high) = input.widen();
        assert_eq!(low.to_bitmask(), (1_u64 << lane) & 0xffff);
        assert_eq!(high.to_bitmask(), (1_u64 << lane) >> 16);
    }
}

#[simd_test]
fn widen_mask32x16<S: Simd>(simd: S) {
    for bits in [0, u64::MAX, 0xd36a_59c2_8e17_b4a5, 0x8181] {
        let input = mask32x16::from_bitmask(simd, bits);
        let (low, high): (mask64x8<S>, mask64x8<S>) = input.widen();
        let low_bits = bits & 0xff;
        let high_bits = (bits >> 8) & 0xff;
        assert_eq!(low.to_bitmask(), low_bits);
        assert_eq!(high.to_bitmask(), high_bits);
        assert_eq!(
            <[i64; 8]>::from(low),
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        assert_eq!(
            <[i64; 8]>::from(high),
            core::array::from_fn(|i| if high_bits & (1 << i) != 0 { -1 } else { 0 }),
        );
        let selected = low.select(i64x8::splat(simd, 7), i64x8::splat(simd, 3));
        assert_eq!(
            *selected,
            core::array::from_fn(|i| if low_bits & (1 << i) != 0 { 7 } else { 3 }),
        );
        assert_eq!(low.narrow(high).to_bitmask(), bits & 0xffff);
    }

    for lane in 0..16 {
        let input = mask32x16::from_bitmask(simd, 1_u64 << lane);
        let (low, high) = input.widen();
        assert_eq!(low.to_bitmask(), (1_u64 << lane) & 0xff);
        assert_eq!(high.to_bitmask(), (1_u64 << lane) >> 8);
    }
}

#[simd_test]
fn widen_native_mask8s<S: Simd>(simd: S) {
    let input = S::mask8s::from_bitmask(simd, 0xd36a_59c2_8e17_b4a5);
    let (low, high): (S::mask16s, S::mask16s) = input.widen();
    let half_mask = (1_u64 << S::mask16s::LEN) - 1;
    assert_eq!(low.to_bitmask(), input.to_bitmask() & half_mask);
    assert_eq!(high.to_bitmask(), input.to_bitmask() >> S::mask16s::LEN);
    let roundtrip: S::mask8s = low.narrow(high);
    assert_eq!(roundtrip.to_bitmask(), input.to_bitmask());
}

#[simd_test]
fn widen_native_mask16s<S: Simd>(simd: S) {
    let input = S::mask16s::from_bitmask(simd, 0xd36a_59c2_8e17_b4a5);
    let (low, high): (S::mask32s, S::mask32s) = input.widen();
    let half_mask = (1_u64 << S::mask32s::LEN) - 1;
    assert_eq!(low.to_bitmask(), input.to_bitmask() & half_mask);
    assert_eq!(high.to_bitmask(), input.to_bitmask() >> S::mask32s::LEN);
    let roundtrip: S::mask16s = low.narrow(high);
    assert_eq!(roundtrip.to_bitmask(), input.to_bitmask());
}

#[simd_test]
fn widen_native_mask32s<S: Simd>(simd: S) {
    let input = S::mask32s::from_bitmask(simd, 0xd36a_59c2_8e17_b4a5);
    let (low, high): (S::mask64s, S::mask64s) = input.widen();
    let half_mask = (1_u64 << S::mask64s::LEN) - 1;
    assert_eq!(low.to_bitmask(), input.to_bitmask() & half_mask);
    assert_eq!(high.to_bitmask(), input.to_bitmask() >> S::mask64s::LEN);
    let roundtrip: S::mask32s = low.narrow(high);
    assert_eq!(roundtrip.to_bitmask(), input.to_bitmask());
}
