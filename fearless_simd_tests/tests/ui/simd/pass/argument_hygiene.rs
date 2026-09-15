// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//@run

#![deny(warnings)]
#![forbid(unsafe_code)]

use fearless_simd::{Fallback, Simd};
use fearless_simd_macros::simd;

struct __FearlessDispatch(u32);

fn entry(value: u32) -> u32 {
    value + 1
}

#[simd]
fn unused_token<S: Simd>(simd: S, value: u32) -> u32 {
    value
}

#[simd]
fn names<S: Simd>(
    _: S,
    __fearless_argument_0: u32,
    __fearless_argument_2: u32,
    r#type: __FearlessDispatch,
    #[expect(unused_variables)] unused: u32,
) -> u32 {
    entry(__fearless_argument_0 + __fearless_argument_2 + r#type.0)
}

macro_rules! pair {
    ($a:ident, $b:ident) => { ($a, $b) };
}

#[simd]
fn pattern<S: Simd>(_: S, pair!(a, b): (u32, u32)) -> u32 {
    a + b
}

impl __FearlessDispatch {
    #[simd]
    fn method<S: Simd>(&self, _: S, value: u32) -> u32 {
        macro_rules! receiver {
            () => { self.0 };
        }
        receiver!() + value
    }
}

fn main() {
    let simd = Fallback::new();
    assert_eq!(unused_token(simd, 42), 42);
    assert_eq!(names(simd, 1, 2, __FearlessDispatch(3), 0), 7);
    assert_eq!(pattern(simd, (1, 2)), 3);
    assert_eq!(__FearlessDispatch(3).method(simd, 4), 7);
}
