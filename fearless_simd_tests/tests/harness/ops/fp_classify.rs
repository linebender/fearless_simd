// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{f32x4, f32x8, f32x16, f64x2, f64x4, f64x8, mask32x4, mask64x4, prelude::*};
use fearless_simd_dev_macros::simd_test;

#[simd_test]
fn fp_classify_f32<S: Simd>(simd: S) {
    const VALUES: &[f32] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0xFFFF_FFFF), // -NaN
        f32::from_bits(0x7FFF_FFFF), // +NaN
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE.next_down(),
        f32::MIN_POSITIVE.next_up(),
        0.0_f32.next_up(),
        0.0_f32.next_down(),
        0.0_f32.next_down().next_down(),
    ];

    let classify = move |f: f32, func: fn(f32x4<S>) -> mask32x4<S>| -> bool {
        let v = f32x4::splat(simd, f);
        func(v).any_true()
    };

    for &f in VALUES {
        assert_eq!(f.is_nan(), classify(f, f32x4::is_nan), "f = {f}");
        assert_eq!(f.is_infinite(), classify(f, f32x4::is_infinite), "f = {f}");
        assert_eq!(f.is_finite(), classify(f, f32x4::is_finite), "f = {f}");
        assert_eq!(
            f.is_subnormal(),
            classify(f, f32x4::is_subnormal),
            "f = {f}"
        );
        assert_eq!(f.is_normal(), classify(f, f32x4::is_normal), "f = {f}");
        assert_eq!(
            f.is_sign_negative(),
            classify(f, f32x4::is_sign_negative),
            "f = {f}"
        );
        assert_eq!(
            f.is_sign_positive(),
            classify(f, f32x4::is_sign_positive),
            "f = {f}"
        );
    }
}

#[simd_test]
fn fp_classify_f32x4_mixed<S: Simd>(simd: S) {
    let cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7F80_0001), // Positive signaling NaN.
        f32::from_bits(0xFF80_0001), // Negative signaling NaN.
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE.next_down(),
        -f32::MIN_POSITIVE.next_down(),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MIN_POSITIVE.next_up(),
        -f32::MIN_POSITIVE.next_up(),
        f32::MAX,
        f32::MIN,
    ];

    // Rotate the cases through every lane, including across native-vector boundaries.
    for offset in 0..cases.len() {
        let values: [f32; 4] = core::array::from_fn(|lane| cases[(offset + lane) % cases.len()]);
        let a = f32x4::from_slice(simd, &values);
        assert_eq!(
            <[i32; 4]>::from(a.is_nan()),
            values.map(|f| -(f.is_nan() as i32)),
            "f32x4::is_nan: {values:?}"
        );
        assert_eq!(
            <[i32; 4]>::from(a.is_infinite()),
            values.map(|f| -(f.is_infinite() as i32)),
            "f32x4::is_infinite: {values:?}"
        );
        assert_eq!(
            <[i32; 4]>::from(a.is_finite()),
            values.map(|f| -(f.is_finite() as i32)),
            "f32x4::is_finite: {values:?}"
        );
        assert_eq!(
            <[i32; 4]>::from(a.is_subnormal()),
            values.map(|f| -(f.is_subnormal() as i32)),
            "f32x4::is_subnormal: {values:?}"
        );
        assert_eq!(
            <[i32; 4]>::from(a.is_normal()),
            values.map(|f| -(f.is_normal() as i32)),
            "f32x4::is_normal: {values:?}"
        );
    }
}

#[simd_test]
fn fp_classify_f32x8_mixed<S: Simd>(simd: S) {
    let cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7F80_0001), // Positive signaling NaN.
        f32::from_bits(0xFF80_0001), // Negative signaling NaN.
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE.next_down(),
        -f32::MIN_POSITIVE.next_down(),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MIN_POSITIVE.next_up(),
        -f32::MIN_POSITIVE.next_up(),
        f32::MAX,
        f32::MIN,
    ];

    // Rotate the cases through every lane, including across native-vector boundaries.
    for offset in 0..cases.len() {
        let values: [f32; 8] = core::array::from_fn(|lane| cases[(offset + lane) % cases.len()]);
        let a = f32x8::from_slice(simd, &values);
        assert_eq!(
            <[i32; 8]>::from(a.is_nan()),
            values.map(|f| -(f.is_nan() as i32)),
            "f32x8::is_nan: {values:?}"
        );
        assert_eq!(
            <[i32; 8]>::from(a.is_infinite()),
            values.map(|f| -(f.is_infinite() as i32)),
            "f32x8::is_infinite: {values:?}"
        );
        assert_eq!(
            <[i32; 8]>::from(a.is_finite()),
            values.map(|f| -(f.is_finite() as i32)),
            "f32x8::is_finite: {values:?}"
        );
        assert_eq!(
            <[i32; 8]>::from(a.is_subnormal()),
            values.map(|f| -(f.is_subnormal() as i32)),
            "f32x8::is_subnormal: {values:?}"
        );
        assert_eq!(
            <[i32; 8]>::from(a.is_normal()),
            values.map(|f| -(f.is_normal() as i32)),
            "f32x8::is_normal: {values:?}"
        );
    }
}

#[simd_test]
fn fp_classify_f32x16_mixed<S: Simd>(simd: S) {
    let cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7F80_0001), // Positive signaling NaN.
        f32::from_bits(0xFF80_0001), // Negative signaling NaN.
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE.next_down(),
        -f32::MIN_POSITIVE.next_down(),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MIN_POSITIVE.next_up(),
        -f32::MIN_POSITIVE.next_up(),
        f32::MAX,
        f32::MIN,
    ];

    // Rotate the cases through every lane, including across native-vector boundaries.
    for offset in 0..cases.len() {
        let values: [f32; 16] = core::array::from_fn(|lane| cases[(offset + lane) % cases.len()]);
        let a = f32x16::from_slice(simd, &values);
        assert_eq!(
            <[i32; 16]>::from(a.is_nan()),
            values.map(|f| -(f.is_nan() as i32)),
            "f32x16::is_nan: {values:?}"
        );
        assert_eq!(
            <[i32; 16]>::from(a.is_infinite()),
            values.map(|f| -(f.is_infinite() as i32)),
            "f32x16::is_infinite: {values:?}"
        );
        assert_eq!(
            <[i32; 16]>::from(a.is_finite()),
            values.map(|f| -(f.is_finite() as i32)),
            "f32x16::is_finite: {values:?}"
        );
        assert_eq!(
            <[i32; 16]>::from(a.is_subnormal()),
            values.map(|f| -(f.is_subnormal() as i32)),
            "f32x16::is_subnormal: {values:?}"
        );
        assert_eq!(
            <[i32; 16]>::from(a.is_normal()),
            values.map(|f| -(f.is_normal() as i32)),
            "f32x16::is_normal: {values:?}"
        );
    }
}

#[simd_test]
fn fp_classify_f64x2_mixed<S: Simd>(simd: S) {
    let cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7FF0_0000_0000_0001), // Positive signaling NaN.
        f64::from_bits(0xFFF0_0000_0000_0001), // Negative signaling NaN.
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE.next_down(),
        -f64::MIN_POSITIVE.next_down(),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MIN_POSITIVE.next_up(),
        -f64::MIN_POSITIVE.next_up(),
        f64::MAX,
        f64::MIN,
    ];

    // Rotate the cases through every lane, including across native-vector boundaries.
    for offset in 0..cases.len() {
        let values: [f64; 2] = core::array::from_fn(|lane| cases[(offset + lane) % cases.len()]);
        let a = f64x2::from_slice(simd, &values);
        assert_eq!(
            <[i64; 2]>::from(a.is_nan()),
            values.map(|f| -(f.is_nan() as i64)),
            "f64x2::is_nan: {values:?}"
        );
        assert_eq!(
            <[i64; 2]>::from(a.is_infinite()),
            values.map(|f| -(f.is_infinite() as i64)),
            "f64x2::is_infinite: {values:?}"
        );
        assert_eq!(
            <[i64; 2]>::from(a.is_finite()),
            values.map(|f| -(f.is_finite() as i64)),
            "f64x2::is_finite: {values:?}"
        );
        assert_eq!(
            <[i64; 2]>::from(a.is_subnormal()),
            values.map(|f| -(f.is_subnormal() as i64)),
            "f64x2::is_subnormal: {values:?}"
        );
        assert_eq!(
            <[i64; 2]>::from(a.is_normal()),
            values.map(|f| -(f.is_normal() as i64)),
            "f64x2::is_normal: {values:?}"
        );
    }
}

#[simd_test]
fn fp_classify_f64x4_mixed<S: Simd>(simd: S) {
    let cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7FF0_0000_0000_0001), // Positive signaling NaN.
        f64::from_bits(0xFFF0_0000_0000_0001), // Negative signaling NaN.
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE.next_down(),
        -f64::MIN_POSITIVE.next_down(),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MIN_POSITIVE.next_up(),
        -f64::MIN_POSITIVE.next_up(),
        f64::MAX,
        f64::MIN,
    ];

    // Rotate the cases through every lane, including across native-vector boundaries.
    for offset in 0..cases.len() {
        let values: [f64; 4] = core::array::from_fn(|lane| cases[(offset + lane) % cases.len()]);
        let a = f64x4::from_slice(simd, &values);
        assert_eq!(
            <[i64; 4]>::from(a.is_nan()),
            values.map(|f| -(f.is_nan() as i64)),
            "f64x4::is_nan: {values:?}"
        );
        assert_eq!(
            <[i64; 4]>::from(a.is_infinite()),
            values.map(|f| -(f.is_infinite() as i64)),
            "f64x4::is_infinite: {values:?}"
        );
        assert_eq!(
            <[i64; 4]>::from(a.is_finite()),
            values.map(|f| -(f.is_finite() as i64)),
            "f64x4::is_finite: {values:?}"
        );
        assert_eq!(
            <[i64; 4]>::from(a.is_subnormal()),
            values.map(|f| -(f.is_subnormal() as i64)),
            "f64x4::is_subnormal: {values:?}"
        );
        assert_eq!(
            <[i64; 4]>::from(a.is_normal()),
            values.map(|f| -(f.is_normal() as i64)),
            "f64x4::is_normal: {values:?}"
        );
    }
}

#[simd_test]
fn fp_classify_f64x8_mixed<S: Simd>(simd: S) {
    let cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7FF0_0000_0000_0001), // Positive signaling NaN.
        f64::from_bits(0xFFF0_0000_0000_0001), // Negative signaling NaN.
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE.next_down(),
        -f64::MIN_POSITIVE.next_down(),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MIN_POSITIVE.next_up(),
        -f64::MIN_POSITIVE.next_up(),
        f64::MAX,
        f64::MIN,
    ];

    // Rotate the cases through every lane, including across native-vector boundaries.
    for offset in 0..cases.len() {
        let values: [f64; 8] = core::array::from_fn(|lane| cases[(offset + lane) % cases.len()]);
        let a = f64x8::from_slice(simd, &values);
        assert_eq!(
            <[i64; 8]>::from(a.is_nan()),
            values.map(|f| -(f.is_nan() as i64)),
            "f64x8::is_nan: {values:?}"
        );
        assert_eq!(
            <[i64; 8]>::from(a.is_infinite()),
            values.map(|f| -(f.is_infinite() as i64)),
            "f64x8::is_infinite: {values:?}"
        );
        assert_eq!(
            <[i64; 8]>::from(a.is_finite()),
            values.map(|f| -(f.is_finite() as i64)),
            "f64x8::is_finite: {values:?}"
        );
        assert_eq!(
            <[i64; 8]>::from(a.is_subnormal()),
            values.map(|f| -(f.is_subnormal() as i64)),
            "f64x8::is_subnormal: {values:?}"
        );
        assert_eq!(
            <[i64; 8]>::from(a.is_normal()),
            values.map(|f| -(f.is_normal() as i64)),
            "f64x8::is_normal: {values:?}"
        );
    }
}
#[simd_test]
fn fp_classify_f64<S: Simd>(simd: S) {
    const VALUES: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0xFFFF_FFFF_FFFF_FFFF), // -NaN
        f64::from_bits(0x7FFF_FFFF_FFFF_FFFF), // +NaN
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE.next_down(),
        f64::MIN_POSITIVE.next_up(),
        0.0_f64.next_up(),
        0.0_f64.next_down(),
        0.0_f64.next_down().next_down(),
    ];

    let classify = move |f: f64, func: fn(f64x4<S>) -> mask64x4<S>| -> bool {
        let v = f64x4::splat(simd, f);
        func(v).any_true()
    };

    for &f in VALUES {
        assert_eq!(f.is_nan(), classify(f, f64x4::is_nan), "f = {f}");
        assert_eq!(f.is_infinite(), classify(f, f64x4::is_infinite), "f = {f}");
        assert_eq!(f.is_finite(), classify(f, f64x4::is_finite), "f = {f}");
        assert_eq!(
            f.is_subnormal(),
            classify(f, f64x4::is_subnormal),
            "f = {f}"
        );
        assert_eq!(f.is_normal(), classify(f, f64x4::is_normal), "f = {f}");
        assert_eq!(
            f.is_sign_negative(),
            classify(f, f64x4::is_sign_negative),
            "f = {f}"
        );
        assert_eq!(
            f.is_sign_positive(),
            classify(f, f64x4::is_sign_positive),
            "f = {f}"
        );
    }
}
