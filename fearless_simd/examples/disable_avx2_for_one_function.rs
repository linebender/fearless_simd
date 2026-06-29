// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This example demonstrates a SIMD function that's never compiled for AVX2.
//!
//! This can be useful if benchmarks show a specific instruction set regressing performance.

use fearless_simd::{Level, dispatch, prelude::*};

#[inline(always)]
fn disable_avx2<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) {
    let level = simd.level();
    match level {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(_) => {
            // downgrade AVX2 to SSE4.2
            let simd_sse4_2 = level.as_sse4_2().unwrap();
            sigmoid(simd_sse4_2, x, out)
        }
        _ => sigmoid(simd, x, out),
    }
}

#[inline(always)]
fn sigmoid<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) {
    // verify that this function is never called with AVX2
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    debug_assert!(!matches!(simd.level(), Level::Avx2(_)));

    // If this function calls anything else with its `simd`,
    // the callee will also never get an AVX2 implementation.
    // The downgrade only needs to happen once anywhere in the call chain.

    let n = S::f32s::N;
    for (x, y) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let a = S::f32s::from_slice(simd, x);
        let b = a / (a * a + 1.0).sqrt();
        y.copy_from_slice(b.as_slice());
    }
}

fn main() {
    let level = Level::new();
    let inp = [0.1, -0.2, 0.001, 0.4, 1., 2., 3., 4.];
    let mut out = [0.; 8];
    dispatch!(level, simd => disable_avx2(simd, &inp, &mut out));

    println!("{out:?}");
}
