// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn index_consistency<S: Simd>(simd: S) {
    // We'll call index methods by name to avoid confusing clippy.
    use std::ops::{Index, IndexMut};

    let mut v = u32x4::from_slice(simd, &[0, 1, 2, 3]);
    for i in 0..4 {
        assert_eq!(i, *v.index(i) as usize);
        assert_eq!(i, *v.index_mut(i) as usize);
    }
}
