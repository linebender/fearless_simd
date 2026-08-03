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

#[simd_test]
fn widen_narrow_f32x4<S: Simd>(simd: S) {
    let input_values = [0.0_f32, -0.0, 1.5, -2.25];
    let input = f32x4::from_slice(simd, &input_values);
    let (low, high) = input.widen();
    for i in 0..f64x2::<S>::N {
        assert_eq!(low[i].to_bits(), (input_values[i] as f64).to_bits());
        assert_eq!(
            high[i].to_bits(),
            (input_values[i + f64x2::<S>::N] as f64).to_bits()
        );
    }
    let roundtrip = low.narrow(high);
    for i in 0..f32x4::<S>::N {
        assert_eq!(roundtrip[i].to_bits(), input_values[i].to_bits());
    }

    let low_values = [
        1.0_f64 + 2.0_f64.powi(-24),
        1.0 + 3.0_f64 * 2.0_f64.powi(-24),
    ];
    let high_values = [-0.0_f64, 1.0 / 3.0];
    let low = f64x2::from_slice(simd, &low_values);
    let high = f64x2::from_slice(simd, &high_values);
    let narrowed = low.narrow(high);
    let saturated = low.saturating_narrow(high);
    assert_eq!(narrowed[0].to_bits(), 1.0_f32.to_bits());
    assert_eq!(narrowed[1].to_bits(), (1.0_f32.to_bits() + 2));
    for i in 0..f32x4::<S>::N {
        let source = if i < 2 {
            low_values[i]
        } else {
            high_values[i - 2]
        };
        let expected = source as f32;
        assert_eq!(narrowed[i].to_bits(), expected.to_bits());
        assert_eq!(saturated[i].to_bits(), narrowed[i].to_bits());
    }
}

#[simd_test]
fn widen_narrow_f32x8<S: Simd>(simd: S) {
    let input_values = [
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        42.5,
    ];
    let input = f32x8::from_slice(simd, &input_values);
    let (low, high) = input.widen();
    for i in 0..f64x4::<S>::N {
        assert_eq!(low[i].to_bits(), (input_values[i] as f64).to_bits());
        assert_eq!(
            high[i].to_bits(),
            (input_values[i + f64x4::<S>::N] as f64).to_bits()
        );
    }
    let roundtrip = low.saturating_narrow(high);
    for i in 0..f32x8::<S>::N {
        assert_eq!(roundtrip[i].to_bits(), input_values[i].to_bits());
    }

    let half_min_subnormal = 2.0_f64.powi(-150);
    let low_values = [f32::MAX as f64, f64::MAX, -f64::MAX, f64::INFINITY];
    let high_values = [
        f64::NEG_INFINITY,
        half_min_subnormal,
        3.0 * half_min_subnormal,
        f64::NAN,
    ];
    let low = f64x4::from_slice(simd, &low_values);
    let high = f64x4::from_slice(simd, &high_values);
    let narrowed = low.narrow(high);
    let saturated = low.saturating_narrow(high);
    assert_eq!(narrowed[1], f32::INFINITY);
    assert_eq!(narrowed[2], f32::NEG_INFINITY);
    assert_eq!(narrowed[5].to_bits(), 0.0_f32.to_bits());
    assert_eq!(narrowed[6].to_bits(), f32::from_bits(2).to_bits());
    for i in 0..f32x8::<S>::N {
        let source = if i < 4 {
            low_values[i]
        } else {
            high_values[i - 4]
        };
        let expected = source as f32;
        if expected.is_nan() {
            assert!(narrowed[i].is_nan());
            assert!(saturated[i].is_nan());
        } else {
            assert_eq!(narrowed[i].to_bits(), expected.to_bits());
            assert_eq!(saturated[i].to_bits(), narrowed[i].to_bits());
        }
    }
}

#[simd_test]
fn widen_narrow_f32x16<S: Simd>(simd: S) {
    let input_values = [
        0.0_f32,
        -0.0,
        1.0,
        -1.0,
        1.0 / 3.0,
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_1234),
        16_777_216.0,
        -16_777_216.0,
    ];
    let input = f32x16::from_slice(simd, &input_values);
    let (low, high) = input.widen();
    for i in 0..f64x8::<S>::N {
        assert_eq!(low[i].to_bits(), (input_values[i] as f64).to_bits());
        let source = input_values[i + f64x8::<S>::N];
        if source.is_nan() {
            assert!(high[i].is_nan());
        } else {
            assert_eq!(high[i].to_bits(), (source as f64).to_bits());
        }
    }
    let roundtrip = low.narrow(high);
    let saturated_roundtrip = low.saturating_narrow(high);
    for i in 0..f32x16::<S>::N {
        if input_values[i].is_nan() {
            assert!(roundtrip[i].is_nan());
            assert!(saturated_roundtrip[i].is_nan());
        } else {
            assert_eq!(roundtrip[i].to_bits(), input_values[i].to_bits());
            assert_eq!(saturated_roundtrip[i].to_bits(), input_values[i].to_bits());
        }
    }

    let low_values = [
        0.0_f64,
        -0.0,
        1.5,
        -2.25,
        1.0 + 2.0_f64.powi(-24),
        1.0 + 3.0_f64 * 2.0_f64.powi(-24),
        f32::MAX as f64,
        f64::MAX,
    ];
    let high_values = [
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2.0_f64.powi(-150),
        3.0_f64 * 2.0_f64.powi(-150),
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::NAN,
    ];
    let low = f64x8::from_slice(simd, &low_values);
    let high = f64x8::from_slice(simd, &high_values);
    let narrowed = low.narrow(high);
    let saturated = low.saturating_narrow(high);
    for i in 0..f32x16::<S>::N {
        let source = if i < 8 {
            low_values[i]
        } else {
            high_values[i - 8]
        };
        let expected = source as f32;
        if expected.is_nan() {
            assert!(narrowed[i].is_nan());
            assert!(saturated[i].is_nan());
        } else {
            assert_eq!(narrowed[i].to_bits(), expected.to_bits());
            assert_eq!(saturated[i].to_bits(), narrowed[i].to_bits());
        }
    }
}

#[simd_test]
#[ignore] // slow: run with `cargo test --release widen_narrow_random -- --ignored`.
fn widen_narrow_random<S: Simd>(simd: S) {
    let mut rng = fastrand::Rng::with_seed(0x243f_6a88_85a3_08d3);

    for iteration in 0..100_000 {
        let input_u8: [u8; 32] = core::array::from_fn(|_| rng.u8(..));
        let input_u8 = u8x32::from_slice(simd, &input_u8);
        let (low, high) = input_u8.widen();
        let expected_low: [u16; 16] = core::array::from_fn(|i| input_u8[i] as u16);
        let expected_high: [u16; 16] = core::array::from_fn(|i| input_u8[i + 16] as u16);
        assert_eq!(*low, expected_low, "u8 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "u8 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *input_u8,
            "u8 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *input_u8,
            "u8 saturating roundtrip iteration {iteration}",
        );

        let input_i8: [i8; 32] = core::array::from_fn(|_| rng.i8(..));
        let input_i8 = i8x32::from_slice(simd, &input_i8);
        let (low, high) = input_i8.widen();
        let expected_low: [i16; 16] = core::array::from_fn(|i| input_i8[i] as i16);
        let expected_high: [i16; 16] = core::array::from_fn(|i| input_i8[i + 16] as i16);
        assert_eq!(*low, expected_low, "i8 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "i8 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *input_i8,
            "i8 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *input_i8,
            "i8 saturating roundtrip iteration {iteration}",
        );

        let a_u16: [u16; 16] = core::array::from_fn(|_| rng.u16(..));
        let b_u16: [u16; 16] = core::array::from_fn(|_| rng.u16(..));
        let a_u16 = u16x16::from_slice(simd, &a_u16);
        let b_u16 = u16x16::from_slice(simd, &b_u16);
        let (low, high) = a_u16.widen();
        let expected_low: [u32; 8] = core::array::from_fn(|i| a_u16[i] as u32);
        let expected_high: [u32; 8] = core::array::from_fn(|i| a_u16[i + 8] as u32);
        assert_eq!(*low, expected_low, "u16 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "u16 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *a_u16,
            "u16 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *a_u16,
            "u16 saturating roundtrip iteration {iteration}",
        );
        let truncated = a_u16.narrow(b_u16);
        let saturated = a_u16.saturating_narrow(b_u16);
        let expected_truncated: [u8; 32] = core::array::from_fn(|i| {
            let value = if i < 16 { a_u16[i] } else { b_u16[i - 16] };
            value as u8
        });
        let expected_saturated: [u8; 32] = core::array::from_fn(|i| {
            let value = if i < 16 { a_u16[i] } else { b_u16[i - 16] };
            value.min(u8::MAX as u16) as u8
        });
        assert_eq!(
            *truncated, expected_truncated,
            "u16 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "u16 saturation iteration {iteration}",
        );

        let a_i16: [i16; 16] = core::array::from_fn(|_| rng.i16(..));
        let b_i16: [i16; 16] = core::array::from_fn(|_| rng.i16(..));
        let a_i16 = i16x16::from_slice(simd, &a_i16);
        let b_i16 = i16x16::from_slice(simd, &b_i16);
        let (low, high) = a_i16.widen();
        let expected_low: [i32; 8] = core::array::from_fn(|i| a_i16[i] as i32);
        let expected_high: [i32; 8] = core::array::from_fn(|i| a_i16[i + 8] as i32);
        assert_eq!(*low, expected_low, "i16 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "i16 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *a_i16,
            "i16 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *a_i16,
            "i16 saturating roundtrip iteration {iteration}",
        );
        let truncated = a_i16.narrow(b_i16);
        let saturated = a_i16.saturating_narrow(b_i16);
        let expected_truncated: [i8; 32] = core::array::from_fn(|i| {
            let value = if i < 16 { a_i16[i] } else { b_i16[i - 16] };
            value as i8
        });
        let expected_saturated: [i8; 32] = core::array::from_fn(|i| {
            let value = if i < 16 { a_i16[i] } else { b_i16[i - 16] };
            value.clamp(i8::MIN as i16, i8::MAX as i16) as i8
        });
        assert_eq!(
            *truncated, expected_truncated,
            "i16 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "i16 saturation iteration {iteration}",
        );

        let a_u32: [u32; 8] = core::array::from_fn(|_| rng.u32(..));
        let b_u32: [u32; 8] = core::array::from_fn(|_| rng.u32(..));
        let a_u32 = u32x8::from_slice(simd, &a_u32);
        let b_u32 = u32x8::from_slice(simd, &b_u32);
        let (low, high) = a_u32.widen();
        let expected_low: [u64; 4] = core::array::from_fn(|i| a_u32[i] as u64);
        let expected_high: [u64; 4] = core::array::from_fn(|i| a_u32[i + 4] as u64);
        assert_eq!(*low, expected_low, "u32 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "u32 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *a_u32,
            "u32 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *a_u32,
            "u32 saturating roundtrip iteration {iteration}",
        );
        let truncated = a_u32.narrow(b_u32);
        let saturated = a_u32.saturating_narrow(b_u32);
        let expected_truncated: [u16; 16] = core::array::from_fn(|i| {
            let value = if i < 8 { a_u32[i] } else { b_u32[i - 8] };
            value as u16
        });
        let expected_saturated: [u16; 16] = core::array::from_fn(|i| {
            let value = if i < 8 { a_u32[i] } else { b_u32[i - 8] };
            value.min(u16::MAX as u32) as u16
        });
        assert_eq!(
            *truncated, expected_truncated,
            "u32 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "u32 saturation iteration {iteration}",
        );

        let a_i32: [i32; 8] = core::array::from_fn(|_| rng.i32(..));
        let b_i32: [i32; 8] = core::array::from_fn(|_| rng.i32(..));
        let a_i32 = i32x8::from_slice(simd, &a_i32);
        let b_i32 = i32x8::from_slice(simd, &b_i32);
        let (low, high) = a_i32.widen();
        let expected_low: [i64; 4] = core::array::from_fn(|i| a_i32[i] as i64);
        let expected_high: [i64; 4] = core::array::from_fn(|i| a_i32[i + 4] as i64);
        assert_eq!(*low, expected_low, "i32 widening iteration {iteration}");
        assert_eq!(*high, expected_high, "i32 widening iteration {iteration}");
        assert_eq!(
            *low.narrow(high),
            *a_i32,
            "i32 roundtrip iteration {iteration}",
        );
        assert_eq!(
            *low.saturating_narrow(high),
            *a_i32,
            "i32 saturating roundtrip iteration {iteration}",
        );
        let truncated = a_i32.narrow(b_i32);
        let saturated = a_i32.saturating_narrow(b_i32);
        let expected_truncated: [i16; 16] = core::array::from_fn(|i| {
            let value = if i < 8 { a_i32[i] } else { b_i32[i - 8] };
            value as i16
        });
        let expected_saturated: [i16; 16] = core::array::from_fn(|i| {
            let value = if i < 8 { a_i32[i] } else { b_i32[i - 8] };
            value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
        });
        assert_eq!(
            *truncated, expected_truncated,
            "i32 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "i32 saturation iteration {iteration}",
        );

        let a_u64: [u64; 4] = core::array::from_fn(|_| rng.u64(..));
        let b_u64: [u64; 4] = core::array::from_fn(|_| rng.u64(..));
        let a_u64 = u64x4::from_slice(simd, &a_u64);
        let b_u64 = u64x4::from_slice(simd, &b_u64);
        let truncated = a_u64.narrow(b_u64);
        let saturated = a_u64.saturating_narrow(b_u64);
        let expected_truncated: [u32; 8] = core::array::from_fn(|i| {
            let value = if i < 4 { a_u64[i] } else { b_u64[i - 4] };
            value as u32
        });
        let expected_saturated: [u32; 8] = core::array::from_fn(|i| {
            let value = if i < 4 { a_u64[i] } else { b_u64[i - 4] };
            value.min(u32::MAX as u64) as u32
        });
        assert_eq!(
            *truncated, expected_truncated,
            "u64 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "u64 saturation iteration {iteration}",
        );

        let a_i64: [i64; 4] = core::array::from_fn(|_| rng.i64(..));
        let b_i64: [i64; 4] = core::array::from_fn(|_| rng.i64(..));
        let a_i64 = i64x4::from_slice(simd, &a_i64);
        let b_i64 = i64x4::from_slice(simd, &b_i64);
        let truncated = a_i64.narrow(b_i64);
        let saturated = a_i64.saturating_narrow(b_i64);
        let expected_truncated: [i32; 8] = core::array::from_fn(|i| {
            let value = if i < 4 { a_i64[i] } else { b_i64[i - 4] };
            value as i32
        });
        let expected_saturated: [i32; 8] = core::array::from_fn(|i| {
            let value = if i < 4 { a_i64[i] } else { b_i64[i - 4] };
            value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
        });
        assert_eq!(
            *truncated, expected_truncated,
            "i64 truncation iteration {iteration}",
        );
        assert_eq!(
            *saturated, expected_saturated,
            "i64 saturation iteration {iteration}",
        );

        let input_f32: [f32; 8] = core::array::from_fn(|_| f32::from_bits(rng.u32(..)));
        let input_f32 = f32x8::from_slice(simd, &input_f32);
        let (low, high) = input_f32.widen();
        for i in 0..f64x4::<S>::N {
            let expected_low = input_f32[i] as f64;
            let expected_high = input_f32[i + 4] as f64;
            if expected_low.is_nan() {
                assert!(low[i].is_nan(), "f32 low widening iteration {iteration}");
            } else {
                assert_eq!(
                    low[i].to_bits(),
                    expected_low.to_bits(),
                    "f32 low widening iteration {iteration}",
                );
            }
            if expected_high.is_nan() {
                assert!(high[i].is_nan(), "f32 high widening iteration {iteration}");
            } else {
                assert_eq!(
                    high[i].to_bits(),
                    expected_high.to_bits(),
                    "f32 high widening iteration {iteration}",
                );
            }
        }
        let roundtrip = low.narrow(high);
        let saturated_roundtrip = low.saturating_narrow(high);
        for i in 0..f32x8::<S>::N {
            if input_f32[i].is_nan() {
                assert!(roundtrip[i].is_nan(), "f32 roundtrip iteration {iteration}");
                assert!(
                    saturated_roundtrip[i].is_nan(),
                    "f32 saturating roundtrip iteration {iteration}",
                );
            } else {
                assert_eq!(
                    roundtrip[i].to_bits(),
                    input_f32[i].to_bits(),
                    "f32 roundtrip iteration {iteration}",
                );
                assert_eq!(
                    saturated_roundtrip[i].to_bits(),
                    input_f32[i].to_bits(),
                    "f32 saturating roundtrip iteration {iteration}",
                );
            }
        }

        let a_f64: [f64; 4] = core::array::from_fn(|_| f64::from_bits(rng.u64(..)));
        let b_f64: [f64; 4] = core::array::from_fn(|_| f64::from_bits(rng.u64(..)));
        let a_f64 = f64x4::from_slice(simd, &a_f64);
        let b_f64 = f64x4::from_slice(simd, &b_f64);
        let narrowed = a_f64.narrow(b_f64);
        let saturated = a_f64.saturating_narrow(b_f64);
        for i in 0..f32x8::<S>::N {
            let source = if i < 4 { a_f64[i] } else { b_f64[i - 4] };
            let expected = source as f32;
            if expected.is_nan() {
                assert!(narrowed[i].is_nan(), "f64 narrowing iteration {iteration}");
                assert!(
                    saturated[i].is_nan(),
                    "f64 saturating narrowing iteration {iteration}",
                );
            } else {
                assert_eq!(
                    narrowed[i].to_bits(),
                    expected.to_bits(),
                    "f64 narrowing iteration {iteration}",
                );
                assert_eq!(
                    saturated[i].to_bits(),
                    expected.to_bits(),
                    "f64 saturating narrowing iteration {iteration}",
                );
            }
        }
    }
}
