// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//@run

use fearless_simd::{Fallback, Simd};
use fearless_simd_macros::simd;

#[simd]
fn with_const_generics<S: Simd, const N: usize>(simd: S, array: [f32; N]) -> [u8; N] {
    let _ = simd.level();
    let mut output = [0; N];
    for i in 0..N {
        output[i] = array[i] as u8;
    }
    output
}

fn main() {
    let fallback = Fallback::new();
    assert_eq!(with_const_generics::<_, 0>(fallback, []), []);
    assert_eq!(with_const_generics(fallback, [42.0]), [42]);
    assert_eq!(
        with_const_generics(fallback, [0.0, 1.5, 42.0, 255.0]),
        [0, 1, 42, 255]
    );
}
