// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//@run

use fearless_simd::{Fallback, Simd};
use fearless_simd_macros::simd;

#[simd]
fn ref_token<S: Simd>(ref simd: S) -> S {
    let token: &S = simd;
    *token
}

#[simd]
fn ref_mut_token<S: Simd>(ref mut simd: S, replacement: S) -> S {
    let token: &mut S = simd;
    *token = replacement;
    *token
}

fn main() {
    let fallback = Fallback::new();
    assert!(ref_token(fallback).level().is_fallback());
    assert!(ref_mut_token(fallback, fallback).level().is_fallback());
}
