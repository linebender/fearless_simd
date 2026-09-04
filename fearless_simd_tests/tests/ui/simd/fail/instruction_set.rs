// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd_macros::simd;

#[simd]
#[instruction_set(arm::a32)]
fn instruction_set_is_not_supported<S: fearless_simd::Simd>(simd: S) {
    let _ = simd;
}

fn main() {}
