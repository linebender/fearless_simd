// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::Simd;
use fearless_simd_macros::simd;

#[simd]
//~^ ERROR: `i32` does not carry a SIMD token
fn invalid<S: Simd>(value: i32, simd: S) -> i32 {
    let _ = simd;
    value
}

#[simd]
//~^ ERROR: `i32` does not carry a SIMD token
fn wildcard(_: i32) {}

fn main() {}
