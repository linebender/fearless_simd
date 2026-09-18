// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::{Fallback, Simd};
use fearless_simd_macros::simd;

const UNIT: () = ();

#[simd]
fn bare_constant<S: Simd>(simd: S, UNIT: ()) {
    //~^ E0530
    // Reusing a constant as an argument name must fail instead of forwarding
    // the constant's value. An explicit path pattern is supported instead.
    let _ = simd;
}

fn main() {
    bare_constant(Fallback::new(), ());
}
