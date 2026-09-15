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
        "the autovectorized body should execute inside the SIMD context"
    );

    let sum = add_vectors(simd, [1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]);
    assert_eq!(
        sum, [9; 8],
        "explicit portable vector operations should return their result"
    );

    assert_eq!(
        increment_first(simd, &mut values),
        Ok(()),
        "the success path should propagate through the SIMD context"
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

#[simd_test]
fn simd_attribute_preserves_argument_patterns_and_borrows<S: Simd>(simd: S) {
    #[simd]
    fn update<'a, S: Simd>(
        _: S,
        (ref label, mut value): (String, u32),
        output: &'a mut [u32],
        tag: impl core::fmt::Display,
    ) -> (&'a mut [u32], impl core::fmt::Display) {
        assert_eq!(label, "value");
        value += 1;
        output[0] = value;
        (output, tag)
    }

    let mut output = [0, 0];
    {
        let (borrowed, tag) = update(simd, ("value".into(), 41), &mut output, "tag");
        assert_eq!(
            borrowed,
            &[42, 0],
            "destructured arguments should update the borrowed output"
        );
        borrowed[1] = 7;
        assert_eq!(
            tag.to_string(),
            "tag",
            "the opaque argument should be returned intact"
        );
    }
    assert_eq!(
        output,
        [42, 7],
        "the returned borrow should still refer to the original output"
    );
}

#[simd_test]
fn simd_attribute_preserves_conditional_arguments<S: Simd>(simd: S) {
    #[simd]
    fn join<S: Simd>(
        _: S,
        #[cfg(test)] prefix: String,
        #[cfg(any())] absent: UndefinedType,
        #[cfg_attr(
            test,
            allow(
                unused_variables,
                reason = "exercise a conditional parameter attribute"
            )
        )]
        unused: u32,
        suffix: String,
    ) -> String {
        prefix + &suffix
    }

    assert_eq!(
        join(simd, "first".into(), 0, "last".into()),
        "firstlast",
        "conditional captures and explicit arguments should both be available"
    );
}

#[simd_test]
fn simd_attribute_preserves_receiver_and_associated_types<S: Simd>(simd: S) {
    struct Container<T>(T);

    impl<T> Container<T> {
        #[simd]
        fn get<S: Simd>(
            &mut self,
            _: S,
            tag: impl core::fmt::Display,
        ) -> (&mut T, impl FnOnce() -> String) {
            let _: Option<Self> = None;
            (&mut self.0, move || tag.to_string())
        }
    }

    trait Identity {
        type Item;

        #[simd]
        fn identity<S: Simd>(&self, _: S, value: Self::Item) -> Self::Item {
            value
        }
    }

    impl Identity for Container<u32> {
        type Item = String;
    }

    let mut container = Container(1);
    let (value, tag) = container.get(simd, "tag");
    *value = 2;
    assert_eq!(tag(), "tag", "the returned closure should own its argument");
    assert_eq!(
        container.0, 2,
        "the returned reference should borrow the receiver"
    );
    assert_eq!(
        container.identity(simd, "owned".into()),
        "owned",
        "Self::Item should resolve in the original trait scope"
    );
}

#[simd_test]
fn simd_attribute_preserves_opaque_return_coercions<S: Simd>(simd: S) {
    #[simd]
    fn select<S: Simd>(
        _: S,
        first: bool,
    ) -> (impl core::fmt::Display, Box<dyn core::fmt::Display>) {
        if first {
            return (1, Box::new(2_u32));
        }
        (3, Box::new(4_u64))
    }

    let (value, boxed) = select(simd, true);
    assert_eq!(
        value.to_string(),
        "1",
        "the early-return opaque value should be preserved"
    );
    assert_eq!(
        boxed.to_string(),
        "2",
        "the early return should coerce to a trait object"
    );
    let (value, boxed) = select(simd, false);
    assert_eq!(
        value.to_string(),
        "3",
        "the tail opaque value should be preserved"
    );
    assert_eq!(
        boxed.to_string(),
        "4",
        "the tail return should coerce to the same trait object type"
    );
}
