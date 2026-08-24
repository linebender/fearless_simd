// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd_macros::simd;

trait Operation {
    #[simd]
    fn bodyless<S: fearless_simd::Simd>(simd: S);
}

fn main() {}
