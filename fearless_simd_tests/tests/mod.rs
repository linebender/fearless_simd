// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    missing_docs,
    clippy::missing_assert_message,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

mod harness;

#[simd_test]
fn saturate_float_to_int<S: Simd>(simd: S) {
    assert_eq!(
        <[u32; 4]>::from(simd.cvt_u32_f32x4(simd.splat_f32x4(f32::INFINITY))),
        [u32::MAX; 4]
    );
    assert_eq!(
        <[u32; 4]>::from(simd.cvt_u32_f32x4(simd.splat_f32x4(-f32::INFINITY))),
        [0; 4]
    );
    assert_eq!(
        <[i32; 4]>::from(simd.cvt_i32_f32x4(simd.splat_f32x4(f32::INFINITY))),
        [i32::MAX; 4]
    );
    assert_eq!(
        <[i32; 4]>::from(simd.cvt_i32_f32x4(simd.splat_f32x4(-f32::INFINITY))),
        [i32::MIN; 4]
    );
}

// Ensure that we can cast between generic native-width vectors
#[expect(dead_code, reason = "Compile only test")]
fn generic_cast<S: Simd>(x: S::f32s) -> S::u32s {
    x.to_int()
}

#[test]
fn supports_highest_level() {
    // When running tests locally, ensure that every SIMD level to be tested is actually supported. The tests themselves
    // will return early and pass if run with an unsupported SIMD level.
    //
    // We skip this on CI because some runners may not support all SIMD levels--in particular, the macOS x86_64 runner
    // doesn't support AVX2.
    if std::env::var_os("CI").is_none() {
        let level = Level::new();

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert!(
            level.as_avx2().is_some(),
            "This machine supports AVX2 and below"
        );

        #[cfg(target_arch = "aarch64")]
        assert!(level.as_neon().is_some(), "This machine supports NEON");

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        assert!(
            level.as_wasm_simd128().is_some(),
            "This environment supports WASM SIMD128"
        );
    }
}
