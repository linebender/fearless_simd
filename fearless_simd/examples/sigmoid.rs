// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This example demonstrates the typical usage Fearless SIMD.
//!
//! The vector size matches the native vector size of the hardware:
//!
//! - SSE and NEON get 128 bit chunks
//! - AVX2 gets 256 bit ones
//! - AVX-512 gets 512-bit ones
//!
//! All from a single function.

use fearless_simd::{Level, dispatch, prelude::*};

/// Applies the sigmoid function to the input and writes to the output
#[inline(always)]
fn sigmoid<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) {
    let n = S::f32s::LEN; // CPU's native vector size
    for (x, y) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let a = S::f32s::from_slice(simd, x);
        let b = a / (a * a + 1.0).sqrt();
        y.copy_from_slice(b.as_slice());
    }
}

fn main() {
    let level = Level::new();
    let inp = [
        0.1, -0.2, 0.001, 0.4, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
    ];
    let mut out = [0.; 16];
    // dispatch! selects the best implementation for the CPU we're running on
    dispatch!(level, simd => sigmoid(simd, &inp, &mut out));

    println!("{out:?}");
}
