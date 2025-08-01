// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

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
#[allow(dead_code)]
fn generic_cast<S: Simd>(x: S::f32s) -> S::u32s {
    x.to_int()
}
