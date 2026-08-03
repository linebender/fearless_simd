// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn widen_narrow_u8x16<S: Simd>(simd: S) {
    let input = u8x16::from_fn(simd, |i| (i * 17) as u8);
    let (low, high) = input.widen();
    for i in 0..u16x8::<S>::N {
        assert_eq!(low[i], input[i] as u16);
        assert_eq!(high[i], input[i + u16x8::<S>::N] as u16);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let a = u16x8::from_fn(simd, |i| match i % 4 {
        0 => 0,
        1 => u8::MAX as u16,
        2 => u8::MAX as u16 + 1,
        _ => u16::MAX,
    });
    let b = u16x8::from_fn(simd, |i| match (i + 1) % 4 {
        0 => 0,
        1 => u8::MAX as u16,
        2 => u8::MAX as u16 + 1,
        _ => u16::MAX,
    });
    let truncated = a.narrow(b);
    let saturated = a.saturating_narrow(b);
    for i in 0..u16x8::<S>::N {
        assert_eq!(truncated[i], a[i] as u8);
        assert_eq!(truncated[i + u16x8::<S>::N], b[i] as u8);
        assert_eq!(saturated[i], a[i].min(u8::MAX as u16) as u8);
        assert_eq!(saturated[i + u16x8::<S>::N], b[i].min(u8::MAX as u16) as u8);
    }
}

#[simd_test]
fn widen_narrow_i8x16<S: Simd>(simd: S) {
    let input = i8x16::from_fn(simd, |i| (i as i16 * 17 - 120) as i8);
    let (low, high) = input.widen();
    for i in 0..i16x8::<S>::N {
        assert_eq!(low[i], input[i] as i16);
        assert_eq!(high[i], input[i + i16x8::<S>::N] as i16);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let a = i16x8::from_fn(simd, |i| match i % 4 {
        0 => i8::MIN as i16 - 1,
        1 => i8::MIN as i16,
        2 => i8::MAX as i16,
        _ => i8::MAX as i16 + 1,
    });
    let b = i16x8::from_fn(simd, |i| match (i + 1) % 4 {
        0 => i8::MIN as i16 - 1,
        1 => i8::MIN as i16,
        2 => i8::MAX as i16,
        _ => i8::MAX as i16 + 1,
    });
    let truncated = a.narrow(b);
    let saturated = a.saturating_narrow(b);
    for i in 0..i16x8::<S>::N {
        assert_eq!(truncated[i], a[i] as i8);
        assert_eq!(truncated[i + i16x8::<S>::N], b[i] as i8);
        assert_eq!(
            saturated[i],
            a[i].clamp(i8::MIN as i16, i8::MAX as i16) as i8
        );
        assert_eq!(
            saturated[i + i16x8::<S>::N],
            b[i].clamp(i8::MIN as i16, i8::MAX as i16) as i8
        );
    }
}

#[simd_test]
fn widen_narrow_u16x8<S: Simd>(simd: S) {
    let input = u16x8::from_fn(simd, |i| (i * 9001) as u16);
    let (low, high) = input.widen();
    for i in 0..u32x4::<S>::N {
        assert_eq!(low[i], input[i] as u32);
        assert_eq!(high[i], input[i + u32x4::<S>::N] as u32);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let a = u32x4::from_fn(simd, |i| match i % 4 {
        0 => 0,
        1 => u16::MAX as u32,
        2 => u16::MAX as u32 + 1,
        _ => u32::MAX,
    });
    let b = u32x4::from_fn(simd, |i| match (i + 1) % 4 {
        0 => 0,
        1 => u16::MAX as u32,
        2 => u16::MAX as u32 + 1,
        _ => u32::MAX,
    });
    let truncated = a.narrow(b);
    let saturated = a.saturating_narrow(b);
    for i in 0..u32x4::<S>::N {
        assert_eq!(truncated[i], a[i] as u16);
        assert_eq!(truncated[i + u32x4::<S>::N], b[i] as u16);
        assert_eq!(saturated[i], a[i].min(u16::MAX as u32) as u16);
        assert_eq!(
            saturated[i + u32x4::<S>::N],
            b[i].min(u16::MAX as u32) as u16
        );
    }
}

#[simd_test]
fn widen_narrow_i16x8<S: Simd>(simd: S) {
    let input = i16x8::from_fn(simd, |i| (i as i32 * 9001 - 30000) as i16);
    let (low, high) = input.widen();
    for i in 0..i32x4::<S>::N {
        assert_eq!(low[i], input[i] as i32);
        assert_eq!(high[i], input[i + i32x4::<S>::N] as i32);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let a = i32x4::from_fn(simd, |i| match i % 4 {
        0 => i16::MIN as i32 - 1,
        1 => i16::MIN as i32,
        2 => i16::MAX as i32,
        _ => i16::MAX as i32 + 1,
    });
    let b = i32x4::from_fn(simd, |i| match (i + 1) % 4 {
        0 => i16::MIN as i32 - 1,
        1 => i16::MIN as i32,
        2 => i16::MAX as i32,
        _ => i16::MAX as i32 + 1,
    });
    let truncated = a.narrow(b);
    let saturated = a.saturating_narrow(b);
    for i in 0..i32x4::<S>::N {
        assert_eq!(truncated[i], a[i] as i16);
        assert_eq!(truncated[i + i32x4::<S>::N], b[i] as i16);
        assert_eq!(
            saturated[i],
            a[i].clamp(i16::MIN as i32, i16::MAX as i32) as i16
        );
        assert_eq!(
            saturated[i + i32x4::<S>::N],
            b[i].clamp(i16::MIN as i32, i16::MAX as i32) as i16
        );
    }
}

#[simd_test]
fn widen_narrow_u32x4<S: Simd>(simd: S) {
    let input = u32x4::from_fn(simd, |i| i as u32 * 1_000_000_007);
    let (low, high) = input.widen();
    for i in 0..u64x2::<S>::N {
        assert_eq!(low[i], input[i] as u64);
        assert_eq!(high[i], input[i + u64x2::<S>::N] as u64);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let a = u64x2::from_fn(simd, |i| {
        if i == 0 {
            u32::MAX as u64
        } else {
            u32::MAX as u64 + 1
        }
    });
    let b = u64x2::from_fn(simd, |i| if i == 0 { u64::MAX } else { 0 });
    let truncated = a.narrow(b);
    let saturated = a.saturating_narrow(b);
    for i in 0..u64x2::<S>::N {
        assert_eq!(truncated[i], a[i] as u32);
        assert_eq!(truncated[i + u64x2::<S>::N], b[i] as u32);
        assert_eq!(saturated[i], a[i].min(u32::MAX as u64) as u32);
        assert_eq!(
            saturated[i + u64x2::<S>::N],
            b[i].min(u32::MAX as u64) as u32
        );
    }
}

#[simd_test]
fn widen_narrow_i32x4<S: Simd>(simd: S) {
    let input = i32x4::from_fn(simd, |i| (i as i64 * 1_000_000_007 - 1_500_000_000) as i32);
    let (low, high) = input.widen();
    for i in 0..i64x2::<S>::N {
        assert_eq!(low[i], input[i] as i64);
        assert_eq!(high[i], input[i + i64x2::<S>::N] as i64);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let a = i64x2::from_fn(simd, |i| {
        if i == 0 {
            i32::MIN as i64 - 1
        } else {
            i32::MIN as i64
        }
    });
    let b = i64x2::from_fn(simd, |i| {
        if i == 0 {
            i32::MAX as i64
        } else {
            i32::MAX as i64 + 1
        }
    });
    let truncated = a.narrow(b);
    let saturated = a.saturating_narrow(b);
    for i in 0..i64x2::<S>::N {
        assert_eq!(truncated[i], a[i] as i32);
        assert_eq!(truncated[i + i64x2::<S>::N], b[i] as i32);
        assert_eq!(
            saturated[i],
            a[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32
        );
        assert_eq!(
            saturated[i + i64x2::<S>::N],
            b[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32
        );
    }
}

#[simd_test]
fn widen_narrow_u8_wide<S: Simd>(simd: S) {
    let input = u8x32::from_fn(simd, |i| (i * 17) as u8);
    let (low, high) = input.widen();
    for i in 0..u16x16::<S>::N {
        assert_eq!(low[i], input[i] as u16);
        assert_eq!(high[i], input[i + u16x16::<S>::N] as u16);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let input = u8x64::from_fn(simd, |i| (i * 17) as u8);
    let (low, high) = input.widen();
    for i in 0..u16x32::<S>::N {
        assert_eq!(low[i], input[i] as u16);
        assert_eq!(high[i], input[i + u16x32::<S>::N] as u16);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);
}

#[simd_test]
fn widen_narrow_i8_wide<S: Simd>(simd: S) {
    let input = i8x32::from_fn(simd, |i| (i as i16 * 17 - 120) as i8);
    let (low, high) = input.widen();
    for i in 0..i16x16::<S>::N {
        assert_eq!(low[i], input[i] as i16);
        assert_eq!(high[i], input[i + i16x16::<S>::N] as i16);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let input = i8x64::from_fn(simd, |i| (i as i16 * 17 - 120) as i8);
    let (low, high) = input.widen();
    for i in 0..i16x32::<S>::N {
        assert_eq!(low[i], input[i] as i16);
        assert_eq!(high[i], input[i + i16x32::<S>::N] as i16);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);
}

#[simd_test]
fn widen_narrow_u16_wide<S: Simd>(simd: S) {
    let input = u16x16::from_fn(simd, |i| (i * 9001) as u16);
    let (low, high) = input.widen();
    for i in 0..u32x8::<S>::N {
        assert_eq!(low[i], input[i] as u32);
        assert_eq!(high[i], input[i + u32x8::<S>::N] as u32);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let input = u16x32::from_fn(simd, |i| (i * 9001) as u16);
    let (low, high) = input.widen();
    for i in 0..u32x16::<S>::N {
        assert_eq!(low[i], input[i] as u32);
        assert_eq!(high[i], input[i + u32x16::<S>::N] as u32);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);
}

#[simd_test]
fn widen_narrow_i16_wide<S: Simd>(simd: S) {
    let input = i16x16::from_fn(simd, |i| (i as i32 * 9001 - 30000) as i16);
    let (low, high) = input.widen();
    for i in 0..i32x8::<S>::N {
        assert_eq!(low[i], input[i] as i32);
        assert_eq!(high[i], input[i + i32x8::<S>::N] as i32);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let input = i16x32::from_fn(simd, |i| (i as i32 * 9001 - 30000) as i16);
    let (low, high) = input.widen();
    for i in 0..i32x16::<S>::N {
        assert_eq!(low[i], input[i] as i32);
        assert_eq!(high[i], input[i + i32x16::<S>::N] as i32);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);
}

#[simd_test]
fn widen_narrow_u32_wide<S: Simd>(simd: S) {
    let input = u32x8::from_fn(simd, |i| (i as u64 * 1_000_000_007) as u32);
    let (low, high) = input.widen();
    for i in 0..u64x4::<S>::N {
        assert_eq!(low[i], input[i] as u64);
        assert_eq!(high[i], input[i + u64x4::<S>::N] as u64);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let input = u32x16::from_fn(simd, |i| (i as u64 * 1_000_000_007) as u32);
    let (low, high) = input.widen();
    for i in 0..u64x8::<S>::N {
        assert_eq!(low[i], input[i] as u64);
        assert_eq!(high[i], input[i + u64x8::<S>::N] as u64);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);
}

#[simd_test]
fn widen_narrow_i32_wide<S: Simd>(simd: S) {
    let input = i32x8::from_fn(simd, |i| (i as i64 * 1_000_000_007 - 1_500_000_000) as i32);
    let (low, high) = input.widen();
    for i in 0..i64x4::<S>::N {
        assert_eq!(low[i], input[i] as i64);
        assert_eq!(high[i], input[i + i64x4::<S>::N] as i64);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);

    let input = i32x16::from_fn(simd, |i| (i as i64 * 1_000_000_007 - 1_500_000_000) as i32);
    let (low, high) = input.widen();
    for i in 0..i64x8::<S>::N {
        assert_eq!(low[i], input[i] as i64);
        assert_eq!(high[i], input[i + i64x8::<S>::N] as i64);
    }
    assert_eq!(*low.narrow(high), *input);
    assert_eq!(*low.saturating_narrow(high), *input);
}
