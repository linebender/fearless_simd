// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(panic = "unwind")]
use std::panic::{AssertUnwindSafe, catch_unwind};

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

#[track_caller]
#[cfg(panic = "unwind")]
fn assert_panics(label: &str, f: impl FnOnce()) {
    assert!(
        catch_unwind(AssertUnwindSafe(f)).is_err(),
        "{label} should panic"
    );
}

#[cfg(not(panic = "unwind"))]
fn assert_panics(_label: &str, _f: impl FnOnce()) {
    // These tests need panic unwinding to observe rejected operations. Some targets, such as
    // wasm32-wasip1, abort on panic instead.
}

macro_rules! for_each_simd_type {
    ($test:ident, $simd:expr) => {
        $test!($simd, f32x4, 4);
        $test!($simd, i8x16, 16);
        $test!($simd, u8x16, 16);
        $test!($simd, mask8x16, 16);
        $test!($simd, i16x8, 8);
        $test!($simd, u16x8, 8);
        $test!($simd, mask16x8, 8);
        $test!($simd, i32x4, 4);
        $test!($simd, u32x4, 4);
        $test!($simd, mask32x4, 4);
        $test!($simd, f64x2, 2);
        $test!($simd, mask64x2, 2);
        $test!($simd, f32x8, 8);
        $test!($simd, i8x32, 32);
        $test!($simd, u8x32, 32);
        $test!($simd, mask8x32, 32);
        $test!($simd, i16x16, 16);
        $test!($simd, u16x16, 16);
        $test!($simd, mask16x16, 16);
        $test!($simd, i32x8, 8);
        $test!($simd, u32x8, 8);
        $test!($simd, mask32x8, 8);
        $test!($simd, f64x4, 4);
        $test!($simd, mask64x4, 4);
        $test!($simd, f32x16, 16);
        $test!($simd, i8x64, 64);
        $test!($simd, u8x64, 64);
        $test!($simd, mask8x64, 64);
        $test!($simd, i16x32, 32);
        $test!($simd, u16x32, 32);
        $test!($simd, mask16x32, 32);
        $test!($simd, i32x16, 16);
        $test!($simd, u32x16, 16);
        $test!($simd, mask32x16, 16);
        $test!($simd, f64x8, 8);
        $test!($simd, mask64x8, 8);
    };
}

macro_rules! for_each_mask_type {
    ($test:ident, $simd:expr) => {
        $test!($simd, mask8x16, 16);
        $test!($simd, mask16x8, 8);
        $test!($simd, mask32x4, 4);
        $test!($simd, mask64x2, 2);
        $test!($simd, mask8x32, 32);
        $test!($simd, mask16x16, 16);
        $test!($simd, mask32x8, 8);
        $test!($simd, mask64x4, 4);
        $test!($simd, mask8x64, 64);
        $test!($simd, mask16x32, 32);
        $test!($simd, mask32x16, 16);
        $test!($simd, mask64x8, 8);
    };
}

macro_rules! check_from_slice_short {
    ($simd:expr, $vec:ident, $len:expr) => {
        assert_panics(stringify!($vec::from_slice), || {
            let _ = $vec::from_slice($simd, &[Default::default(); $len - 1]);
        });
    };
}

macro_rules! check_store_slice_short {
    ($simd:expr, $vec:ident, $len:expr) => {{
        let vec = $vec::from_slice($simd, &[Default::default(); $len]);
        let mut short = [Default::default(); $len - 1];

        assert_panics(stringify!($vec::store_slice), || {
            vec.store_slice(&mut short)
        });
    }};
}

macro_rules! check_mask_test_oob {
    ($simd:expr, $mask:ident, $len:expr) => {{
        let mask = $mask::splat($simd, false);

        assert_panics(stringify!($mask::test), || {
            let _ = mask.test($len);
        });
    }};
}

macro_rules! check_mask_set_oob {
    ($simd:expr, $mask:ident, $len:expr) => {{
        let mut mask = $mask::splat($simd, false);

        assert_panics(stringify!($mask::set), || {
            mask.set($len, true);
        });
    }};
}

#[simd_test]
fn from_slice_rejects_short_slice<S: Simd>(simd: S) {
    for_each_simd_type!(check_from_slice_short, simd);
}

#[simd_test]
fn store_slice_rejects_short_slice<S: Simd>(simd: S) {
    for_each_simd_type!(check_store_slice_short, simd);
}

#[simd_test]
fn mask_test_rejects_out_of_bounds<S: Simd>(simd: S) {
    for_each_mask_type!(check_mask_test_oob, simd);
}

#[simd_test]
fn mask_set_rejects_out_of_bounds<S: Simd>(simd: S) {
    for_each_mask_type!(check_mask_set_oob, simd);
}
