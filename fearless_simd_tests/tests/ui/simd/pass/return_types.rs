// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//@run

use std::fmt::Display;

use fearless_simd::{Fallback, Simd};
use fearless_simd_macros::simd;

#[simd]
fn select_slice<S: Simd>(_: S, first: bool) -> &'static [u32] {
    if first { &[1, 2] } else { &[3] }
}

#[simd]
fn select_box<S: Simd>(_: S, first: bool) -> Box<dyn Display> {
    if first {
        return Box::new(1u32);
    }
    Box::new(2u64)
}

#[simd]
fn nested_opaque<S: Simd>(_: S, first: bool) -> (impl Display, Box<dyn Display>) {
    if first {
        (1, Box::new(2u32))
    } else {
        (3, Box::new(4u64))
    }
}

#[simd]
fn opaque_result<S: Simd>(_: S, fail: bool) -> Result<impl Iterator<Item = u32>, &'static str> {
    if fail {
        return Err("failed");
    }
    Ok([1, 2].into_iter())
}

#[simd]
fn opaque_iterator<S: Simd>(_: S) -> impl Iterator<Item = u32> {
    [1, 2].into_iter()
}

#[simd]
fn borrowed<S: Simd>(_: S, values: &mut [u32; 2]) -> &mut [u32] {
    values
}

#[simd]
fn explicit_unit<S: Simd>(_: S, value: &mut u32) -> () {
    *value += 1;
}

#[simd]
fn diverging<S: Simd>(_: S) -> ! {
    panic!("never returns")
}

fn main() {
    let fallback = Fallback::new();
    assert_eq!(select_slice(fallback, true), &[1, 2]);
    assert_eq!(select_slice(fallback, false), &[3]);
    assert_eq!(select_box(fallback, true).to_string(), "1");
    assert_eq!(select_box(fallback, false).to_string(), "2");
    let (opaque, boxed) = nested_opaque(fallback, true);
    assert_eq!(opaque.to_string(), "1");
    assert_eq!(boxed.to_string(), "2");
    let (opaque, boxed) = nested_opaque(fallback, false);
    assert_eq!(opaque.to_string(), "3");
    assert_eq!(boxed.to_string(), "4");
    assert_eq!(
        opaque_result(fallback, false).unwrap().collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(opaque_result(fallback, true).err(), Some("failed"));
    assert_eq!(opaque_iterator(fallback).collect::<Vec<_>>(), [1, 2]);
    let mut values = [1, 2];
    borrowed(fallback, &mut values)[0] = 3;
    assert_eq!(values, [3, 2]);
    explicit_unit(fallback, &mut values[0]);
    assert_eq!(values, [4, 2]);
    let _: fn(Fallback) -> ! = diverging;
}
