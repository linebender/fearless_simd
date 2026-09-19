// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//@run

#![deny(warnings)]
#![forbid(unsafe_code)]

// Deliberately omit ExtractToken: supertrait bounds and macro-generated calls
// must work without that trait being imported.
use fearless_simd::{Fallback, Simd, SimdBase, SimdMask, u32x4};
use fearless_simd_macros::simd;

#[simd]
fn vector<S: Simd, V: SimdBase<S>>(value: V) -> S {
    value.token()
}

#[simd]
fn mask<S: Simd, M: SimdMask<S>>(value: M) -> S {
    value.token()
}

#[simd]
fn unused<S: Simd>(value: u32x4<S>) {}

#[simd]
fn unit<S: Simd>(_: &mut u32x4<S>) -> () {}

struct Container;

impl Container {
    #[simd]
    fn method<S: Simd>(&self, value: u32x4<S>) -> u32x4<S> {
        value + value
    }
}

fn main() {
    let simd = Fallback::new();
    let mut value = u32x4::splat(simd, 21);
    assert!(vector(value).level().is_fallback());
    assert!(mask(value.simd_eq(value)).level().is_fallback());
    unused(value);
    unit(&mut value);
    assert_eq!(Container.method(value).as_slice(), &[42; 4]);
}
