// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(dead_code)]

static mut CALLBACK: fn(u32) -> u32 = |value| value;

// Identifiers supplied directly to the helper must be evaluated outside its
// unsafe block, so they cannot hide an unsafe read of a mutable static.
fn forward(proof: fearless_simd::Sse2, value: u32) -> u32 {
    fearless_simd::__fearless_simd_dispatch_entry! {
        //~^ E0133
        Sse2, proof, CALLBACK;
        A => value
    }
}

fn main() {}
