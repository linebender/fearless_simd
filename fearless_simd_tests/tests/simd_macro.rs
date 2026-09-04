// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::prelude::*;
use fearless_simd::{Simd, u32x8};
use fearless_simd_dev_macros::simd_test;
use fearless_simd_macros::simd;

#[simd]
fn double_values<S: Simd>(_simd: S, values: &mut [u32]) {
    for value in values {
        *value *= 2;
    }
}

#[simd]
fn add_vectors<S: Simd>(simd: S, left: [u32; 8], right: [u32; 8]) -> [u32; 8] {
    let left = u32x8::from_slice(simd, &left);
    let right = u32x8::from_slice(simd, &right);
    (left + right).into()
}

#[simd]
fn increment_first<S: Simd>(_simd: S, values: &mut [u32]) -> Result<(), &'static str> {
    let first = values.first_mut().ok_or("slice must not be empty")?;
    *first += 1;
    Ok(())
}

#[simd]
fn wildcard_token<S: Simd>(_: S, value: u32) -> u32 {
    value + 1
}

struct MethodFixture(u32);

impl MethodFixture {
    #[simd]
    fn add<S: Simd>(&self, _simd: S, value: u32) -> u32 {
        self.0 + value
    }
}

trait DefaultMethodFixture {
    #[simd]
    fn subtract<S: Simd>(&self, _simd: S, value: u32) -> u32 {
        value - 1
    }
}

impl DefaultMethodFixture for MethodFixture {}

#[simd_test]
fn simd_attribute_runs_on_every_backend<S: Simd>(simd: S) {
    let mut values = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    double_values(simd, &mut values);
    assert_eq!(
        values,
        [2, 4, 6, 8, 10, 12, 14, 16, 18],
        "the autovectorized body should execute inside vectorize"
    );

    let sum = add_vectors(simd, [1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]);
    assert_eq!(
        sum, [9; 8],
        "explicit portable vector operations should return their result"
    );

    assert_eq!(
        increment_first(simd, &mut values),
        Ok(()),
        "the success path should propagate through vectorize"
    );
    assert_eq!(values[0], 3, "the success path should mutate its input");
    assert_eq!(
        increment_first(simd, &mut []),
        Err("slice must not be empty"),
        "the question-mark operator should return from the wrapped body"
    );

    assert_eq!(
        wildcard_token(simd, 41),
        42,
        "a wildcard SIMD parameter should receive a hidden binding"
    );

    let fixture = MethodFixture(10);
    assert_eq!(
        fixture.add(simd, 5),
        15,
        "the first typed parameter after an inherent receiver should be selected"
    );
    assert_eq!(
        fixture.subtract(simd, 5),
        4,
        "the macro should support default trait methods"
    );
}
