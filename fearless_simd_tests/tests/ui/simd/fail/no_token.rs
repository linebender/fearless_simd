// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd_macros::simd;

#[simd]
//~^ ERROR: `#[simd]` requires a SIMD token parameter after any receiver
fn token_is_required() {}

fn main() {}
