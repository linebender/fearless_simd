// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//@run

use fearless_simd::{Fallback, Simd};
use fearless_simd_macros::simd;

struct Fixture;

impl Fixture {
    #[simd]
    fn inherent<S: Simd>(&self, _simd: S, value: u32) -> u32 {
        value + 1
    }
}

trait Operation {
    #[simd]
    fn default_method<S: Simd>(&self, _simd: S, value: u32) -> u32 {
        value + 2
    }
}

impl Operation for Fixture {}

trait ImplementedOperation {
    fn implemented<S: Simd>(&self, simd: S, value: u32) -> u32;
}

impl ImplementedOperation for Fixture {
    #[simd]
    fn implemented<S: Simd>(&self, _simd: S, value: u32) -> u32 {
        value + 3
    }
}

trait AssociatedTypeOperation {
    type A;

    fn with_associated_type<S: Simd>(&self, simd: S, value: Self::A) -> Self::A;
}

impl AssociatedTypeOperation for Fixture {
    type A = u32;

    #[simd]
    fn with_associated_type<S: Simd>(&self, _: S, value: Self::A) -> Self::A {
        value + 1
    }
}

fn main() {
    let fixture = Fixture;
    let fallback = Fallback::new();
    assert_eq!(fixture.inherent(fallback, 1), 2);
    assert_eq!(fixture.default_method(fallback, 1), 3);
    assert_eq!(fixture.implemented(fallback, 1), 4);
    assert_eq!(fixture.with_associated_type(fallback, 41), 42);
}
