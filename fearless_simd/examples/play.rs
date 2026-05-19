// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    missing_docs,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

use fearless_simd::{Level, dispatch, prelude::*};

// The WithSimd idea is adapted from pulp but is clunky; we
// will probably prefer the `dispatch!` macro.
struct Foo;

impl WithSimd for Foo {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let a = simd.splat_f32x4(42.0);
        let b = a + a;
        b[0]
    }
}

#[inline(always)]
fn foo<S: Simd>(simd: S, x: f32) -> f32 {
    let n = S::f32s::N;
    println!("n = {n}");
    simd.splat_f32x4(x).sqrt()[0]
}

fn main() {
    let level = Level::new();
    let x = level.dispatch(Foo);
    let y = dispatch!(level, simd => foo(simd, 42.0));

    println!("level = {level:?}, x = {x}, y = {y}");
}
