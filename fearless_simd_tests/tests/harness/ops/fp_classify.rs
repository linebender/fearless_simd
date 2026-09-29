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

// Keep the non-default floating-point environment entirely inside an opaque
// assembly call, so LLVM cannot move classification across MXCSR changes.
#[cfg(all(target_arch = "x86_64", target_os = "linux", not(miri)))]
mod subnormal_flush_modes {
    use super::*;
    use core::arch::{asm, global_asm};
    use fearless_simd::{Avx512, Level};

    global_asm!(
        ".text",
        ".globl fearless_classify_with_mxcsr",
        ".type fearless_classify_with_mxcsr,@function",
        "fearless_classify_with_mxcsr:",
        "sub rsp, 24",
        "stmxcsr [rsp]",
        "mov [rsp + 4], edx",
        "mov rax, rdi",
        "mov rdi, rsi",
        "ldmxcsr [rsp + 4]",
        "call rax",
        "ldmxcsr [rsp]",
        "add rsp, 24",
        "ret",
        ".size fearless_classify_with_mxcsr,.-fearless_classify_with_mxcsr",
    );

    unsafe extern "C" {
        // The callback must not unwind, and the caller must check AVX-512 support.
        fn fearless_classify_with_mxcsr(
            classify: unsafe extern "C" fn(*const u8) -> u64,
            input: *const u8,
            mxcsr: u32,
        ) -> u64;
    }

    #[test]
    fn is_subnormal_f32_flush_modes() {
        if Level::new().as_avx512().is_none() {
            return;
        }

        unsafe extern "C" fn classify_4(input: *const u8) -> u64 {
            // SAFETY: the test checks AVX-512 support and supplies at least 4 elements.
            let simd = unsafe { Avx512::assume_supported() };
            let values = unsafe { core::slice::from_raw_parts(input.cast::<f32>(), 4) };
            f32x4::from_slice(simd, values).is_subnormal().to_bitmask()
        }

        unsafe extern "C" fn classify_8(input: *const u8) -> u64 {
            // SAFETY: the test checks AVX-512 support and supplies at least 8 elements.
            let simd = unsafe { Avx512::assume_supported() };
            let values = unsafe { core::slice::from_raw_parts(input.cast::<f32>(), 8) };
            f32x8::from_slice(simd, values).is_subnormal().to_bitmask()
        }

        unsafe extern "C" fn classify_16(input: *const u8) -> u64 {
            // SAFETY: the test checks AVX-512 support and supplies at least 16 elements.
            let simd = unsafe { Avx512::assume_supported() };
            let values = unsafe { core::slice::from_raw_parts(input.cast::<f32>(), 16) };
            f32x16::from_slice(simd, values).is_subnormal().to_bitmask()
        }

        let cases: [u32; 20] = [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x007f_ffff,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0000,
            0x7f80_0001,
            0xff80_0001,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x0040_0000,
            0x8040_0000,
        ];
        let mut original_mxcsr = 0_u32;
        // SAFETY: the output is valid storage for STMXCSR; this does not change MXCSR.
        unsafe {
            asm!("stmxcsr [{}]", in(reg) &mut original_mxcsr, options(nostack));
        }
        // Test DAZ (input flushing) and FTZ (output flushing) independently and together.
        for flags in [0, 0x0040, 0x8000, 0x8040] {
            let mxcsr = (original_mxcsr & !0x8040) | flags;
            for offset in 0..cases.len() {
                let bits: [u32; 16] = core::array::from_fn(|i| cases[(offset + i) % cases.len()]);
                let values = bits.map(f32::from_bits);
                let classifiers: [(usize, unsafe extern "C" fn(*const u8) -> u64); 3] =
                    [(4, classify_4), (8, classify_8), (16, classify_16)];
                for (lanes, classify) in classifiers {
                    let mut expected = 0;
                    for (i, &value) in bits[..lanes].iter().enumerate() {
                        if value & 0x7f80_0000 == 0 && value & 0x007f_ffff != 0 {
                            expected |= 1 << i;
                        }
                    }
                    // SAFETY: support was checked, values has enough initialized lanes,
                    // and only the supported DAZ/FTZ bits of MXCSR are changed. The shim
                    // restores MXCSR before returning to Rust (including assertions).
                    let actual = unsafe {
                        fearless_classify_with_mxcsr(classify, values.as_ptr().cast(), mxcsr)
                    };
                    assert_eq!(
                        actual, expected,
                        "lanes={lanes}, flags={flags:#x}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn is_subnormal_f64_flush_modes() {
        if Level::new().as_avx512().is_none() {
            return;
        }

        unsafe extern "C" fn classify_2(input: *const u8) -> u64 {
            // SAFETY: the test checks AVX-512 support and supplies at least 2 elements.
            let simd = unsafe { Avx512::assume_supported() };
            let values = unsafe { core::slice::from_raw_parts(input.cast::<f64>(), 2) };
            f64x2::from_slice(simd, values).is_subnormal().to_bitmask()
        }

        unsafe extern "C" fn classify_4(input: *const u8) -> u64 {
            // SAFETY: the test checks AVX-512 support and supplies at least 4 elements.
            let simd = unsafe { Avx512::assume_supported() };
            let values = unsafe { core::slice::from_raw_parts(input.cast::<f64>(), 4) };
            f64x4::from_slice(simd, values).is_subnormal().to_bitmask()
        }

        unsafe extern "C" fn classify_8(input: *const u8) -> u64 {
            // SAFETY: the test checks AVX-512 support and supplies at least 8 elements.
            let simd = unsafe { Avx512::assume_supported() };
            let values = unsafe { core::slice::from_raw_parts(input.cast::<f64>(), 8) };
            f64x8::from_slice(simd, values).is_subnormal().to_bitmask()
        }

        let cases: [u64; 20] = [
            0x0000_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x8000_0000_0000_0001,
            0x000f_ffff_ffff_ffff,
            0x800f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x8010_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0xbff0_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0xfff8_0000_0000_0000,
            0x7ff0_0000_0000_0001,
            0xfff0_0000_0000_0001,
            0x7fef_ffff_ffff_ffff,
            0xffef_ffff_ffff_ffff,
            0x0008_0000_0000_0000,
            0x8008_0000_0000_0000,
        ];
        let mut original_mxcsr = 0_u32;
        // SAFETY: the output is valid storage for STMXCSR; this does not change MXCSR.
        unsafe {
            asm!("stmxcsr [{}]", in(reg) &mut original_mxcsr, options(nostack));
        }
        // Test DAZ (input flushing) and FTZ (output flushing) independently and together.
        for flags in [0, 0x0040, 0x8000, 0x8040] {
            let mxcsr = (original_mxcsr & !0x8040) | flags;
            for offset in 0..cases.len() {
                let bits: [u64; 8] = core::array::from_fn(|i| cases[(offset + i) % cases.len()]);
                let values = bits.map(f64::from_bits);
                let classifiers: [(usize, unsafe extern "C" fn(*const u8) -> u64); 3] =
                    [(2, classify_2), (4, classify_4), (8, classify_8)];
                for (lanes, classify) in classifiers {
                    let mut expected = 0;
                    for (i, &value) in bits[..lanes].iter().enumerate() {
                        if value & 0x7ff0_0000_0000_0000 == 0 && value & 0x000f_ffff_ffff_ffff != 0
                        {
                            expected |= 1 << i;
                        }
                    }
                    // SAFETY: support was checked, values has enough initialized lanes,
                    // and only the supported DAZ/FTZ bits of MXCSR are changed. The shim
                    // restores MXCSR before returning to Rust (including assertions).
                    let actual = unsafe {
                        fearless_classify_with_mxcsr(classify, values.as_ptr().cast(), mxcsr)
                    };
                    assert_eq!(
                        actual, expected,
                        "lanes={lanes}, flags={flags:#x}, offset={offset}"
                    );
                }
            }
        }
    }
}
