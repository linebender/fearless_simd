// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::prelude::*;
use fearless_simd::{Simd, u32x8};
use fearless_simd_dev_macros::simd_test;
use fearless_simd_macros::simd;

#[simd_test]
fn simd_attribute_extracts_vector_and_mask_tokens<S: Simd>(simd: S) {
    #[simd]
    fn fixed<S: Simd>(a: u32x8<S>) -> u32x8<S> {
        a + a
    }

    #[simd]
    fn native<S: Simd>(a: S::u32s) -> S::u32s {
        a + a
    }

    #[simd]
    fn fixed_mask<S: Simd>(a: fearless_simd::mask32x8<S>) -> bool {
        a.all_true()
    }

    #[simd]
    fn native_mask<S: Simd>(a: S::mask32s) -> bool {
        a.all_true()
    }

    let fixed_value = u32x8::splat(simd, 21);
    assert_eq!(
        fixed(fixed_value).as_slice(),
        &[42; 8],
        "a fixed-width vector should supply its token"
    );
    assert!(
        fixed_mask(fixed_value.simd_eq(fixed_value)),
        "a fixed-width mask should supply its token"
    );
    let native_value = S::u32s::splat(simd, 21);
    assert!(
        native::<S>(native_value)
            .as_slice()
            .iter()
            .all(|&x| x == 42),
        "a native-width vector should supply its token"
    );
    assert!(
        native_mask::<S>(native_value.simd_eq(native_value)),
        "a native-width mask should supply its token"
    );
}

#[simd_test]
fn simd_attribute_borrows_carriers<S: Simd>(simd: S) {
    #[simd]
    fn shared<S: Simd>(a: &u32x8<S>) -> u32x8<S> {
        *a + *a
    }

    #[simd]
    fn mutable<S: Simd>(a: &mut u32x8<S>) -> &mut u32x8<S> {
        *a += 1;
        a
    }

    #[simd]
    fn token_ref<S: Simd>(simd: &S) -> S {
        *simd
    }

    #[simd]
    fn token_mut<S: Simd>(simd: &mut S) -> S {
        *simd
    }

    #[simd]
    fn nested_ref<S: Simd>(a: &&mut u32x8<S>) -> u32x8<S> {
        **a
    }

    #[simd]
    fn unsized_carrier<T: ExtractToken + ?Sized>(a: &T) -> T::S {
        a.witness()
    }

    let mut value = u32x8::splat(simd, 21);
    assert_eq!(
        shared(&value).as_slice(),
        &[42; 8],
        "shared carriers should remain borrowed"
    );
    mutable(&mut value)[0] = 100;
    assert_eq!(
        value.as_slice(),
        &[100, 22, 22, 22, 22, 22, 22, 22],
        "the returned borrow should refer to the original vector"
    );
    assert_eq!(
        nested_ref(&&mut value).as_slice(),
        value.as_slice(),
        "reference forwarding should work recursively"
    );
    assert_eq!(
        core::mem::discriminant(&token_ref(&simd).level()),
        core::mem::discriminant(&simd.level()),
        "a shared token reference should preserve the backend"
    );
    let mut token = simd.witness();
    assert_eq!(
        core::mem::discriminant(&token_mut(&mut token).level()),
        core::mem::discriminant(&simd.level()),
        "a mutable token reference should preserve the backend"
    );
    let erased: &dyn ExtractToken<S = S> = &value;
    assert_eq!(
        core::mem::discriminant(&unsized_carrier(erased).level()),
        core::mem::discriminant(&simd.level()),
        "an unsized carrier should preserve the backend"
    );
}

#[simd_test]
fn simd_attribute_preserves_carrier_patterns<S: Simd>(simd: S) {
    #[simd]
    fn wildcard<S: Simd>(_: u32x8<S>) -> u32 {
        42
    }

    #[simd]
    fn ref_pattern<S: Simd>(ref a: u32x8<S>) -> u32x8<S> {
        *a + *a
    }

    #[simd]
    fn ref_mut_pattern<S: Simd>(ref mut a: u32x8<S>) -> u32x8<S> {
        *a += 1;
        *a
    }

    let value = u32x8::splat(simd, 21);
    assert_eq!(
        wildcard(value),
        42,
        "a wildcard carrier should receive a hidden binding"
    );
    assert_eq!(
        ref_pattern(value).as_slice(),
        &[42; 8],
        "ref patterns should borrow the forwarded vector"
    );
    assert_eq!(
        ref_mut_pattern(value).as_slice(),
        &[22; 8],
        "ref mut patterns should borrow the forwarded vector mutably"
    );
}

#[simd_test]
fn simd_attribute_extracts_owned_wrapper_once<S: Simd>(simd: S) {
    use core::cell::Cell;

    struct Wrapper<'a, S: Simd> {
        vector: u32x8<S>,
        extractions: &'a Cell<usize>,
        drops: &'a Cell<usize>,
    }

    impl<S: Simd> ExtractToken for Wrapper<'_, S> {
        type S = S;

        #[inline]
        fn witness(&self) -> S {
            self.extractions.set(self.extractions.get() + 1);
            self.vector.witness()
        }
    }

    impl<S: Simd> Drop for Wrapper<'_, S> {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    impl<S: Simd> Wrapper<'_, S> {
        #[inline]
        fn witness(&self) -> u32 {
            123
        }
    }

    #[simd]
    fn update<S: Simd>(mut value: Wrapper<'_, S>) -> Wrapper<'_, S> {
        assert_eq!(
            value.extractions.get(),
            1,
            "extraction should happen before the body"
        );
        assert_eq!(
            value.drops.get(),
            0,
            "the body should own the original carrier"
        );
        value.vector += 1;
        value
    }

    let extractions = Cell::new(0);
    let drops = Cell::new(0);
    let value = Wrapper {
        vector: u32x8::splat(simd, 41),
        extractions: &extractions,
        drops: &drops,
    };
    assert_eq!(
        value.witness(),
        123,
        "the inherent method has different semantics"
    );
    let result = update(value);
    assert_eq!(
        result.vector.as_slice(),
        &[42; 8],
        "the wrapper should be returned with the body's changes"
    );
    assert_eq!(
        extractions.get(),
        1,
        "dispatch should use the trait exactly once"
    );
    drop(result);
    assert_eq!(drops.get(), 1, "the carrier should be dropped exactly once");
}

#[simd_test]
fn simd_attribute_returns_borrow_from_wrapper<S: Simd>(simd: S) {
    struct Wrapper<S: Simd> {
        vector: u32x8<S>,
        label: String,
    }

    impl<S: Simd> ExtractToken for Wrapper<S> {
        type S = S;

        #[inline]
        fn witness(&self) -> S {
            self.vector.witness()
        }
    }

    #[simd]
    fn label<S: Simd>(value: &mut Wrapper<S>) -> &mut String {
        &mut value.label
    }

    let mut value = Wrapper {
        vector: u32x8::splat(simd, 0),
        label: "hello".into(),
    };
    label(&mut value).push('!');
    assert_eq!(
        value.label, "hello!",
        "the returned borrow should refer to the original wrapper's field"
    );
}

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
