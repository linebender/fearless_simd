// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd_macros::simd;

#[simd]
fn at_token_is_not_supported<S: fearless_simd::Simd>(simd @ _: S) {
    let _ = simd;
}

fn main() {}
