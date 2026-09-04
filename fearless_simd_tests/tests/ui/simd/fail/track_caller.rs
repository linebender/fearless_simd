// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd_macros::simd;

#[simd]
#[track_caller]
fn track_caller_is_not_supported<S: fearless_simd::Simd>(simd: S) {
    let _ = simd;
}

fn main() {}
