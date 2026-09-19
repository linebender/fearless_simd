// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{Fallback, Simd};
use fearless_simd_macros::simd;

struct Unit;

#[simd]
fn bare_unit<S: Simd>(simd: S, Unit: Unit) {
    //~^ E0530
    // Reusing a constructor as an argument name must fail instead of creating
    // another value. An explicit path pattern is supported instead.
    let _ = simd;
}

fn main() {
    bare_unit(Fallback::new(), Unit);
}
