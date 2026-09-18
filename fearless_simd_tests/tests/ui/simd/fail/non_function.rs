// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd_macros::simd;

#[simd]
struct NotAFunction;
//~^ ERROR: `#[simd]` can only be used on function and method definitions: expected `fn`

#[simd]
enum NotAFunctionEither {}
//~^ ERROR: `#[simd]` can only be used on function and method definitions: expected `fn`

fn main() {}
