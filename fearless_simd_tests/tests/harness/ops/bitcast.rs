// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd::*;
use fearless_simd_dev_macros::simd_test;

// These are direct moves of the pre-existing op tests. Coverage gaps are intentionally left unchanged.

#[simd_test]
fn bitcast_native<S: Simd>(simd: S) {
    let a_i32 = S::i32s::from_slice(simd, &vec![-1; S::i32s::N]);
    assert_eq!(
        a_i32.bitcast::<S::u32s>().as_slice(),
        &vec![u32::MAX; S::i32s::N]
    );
}
