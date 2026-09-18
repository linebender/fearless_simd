// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//@run

#![deny(warnings)]

use fearless_simd::{Fallback, Simd};
use fearless_simd_macros::simd;

#[simd]
#[allow(unused_parens, reason = "exercise a parenthesized parameter binding")]
fn named_patterns<S: Simd>(
    simd: S,
    mut owned: String,
    ref borrowed: String,
    ref mut modified: String,
    whole @ (left, right): (u32, u32),
    (parenthesized): String,
) -> (String, usize, String, u32, String) {
    let _: &String = borrowed;
    let _: &mut String = modified;
    owned.push('!');
    modified.push('!');
    (
        owned,
        borrowed.len(),
        std::mem::take(modified),
        whole.0 + whole.1 + left + right,
        parenthesized,
    )
}

mod values {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub static DROPS: AtomicUsize = AtomicUsize::new(0);
    pub const UNIT: () = ();
    pub struct Unit;

    impl Drop for Unit {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[simd]
fn qualified_patterns<S: Simd>(simd: S, values::Unit: values::Unit, values::UNIT: ()) {
    let _ = simd;
}

fn main() {
    let fallback = Fallback::new();
    assert_eq!(
        named_patterns(
            fallback,
            "owned".into(),
            "borrowed".into(),
            "modified".into(),
            (1, 2),
            "parenthesized".into(),
        ),
        (
            "owned!".into(),
            8,
            "modified!".into(),
            6,
            "parenthesized".into()
        )
    );
    qualified_patterns(fallback, values::Unit, ());
    assert_eq!(values::DROPS.load(std::sync::atomic::Ordering::Relaxed), 1);
}
