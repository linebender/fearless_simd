// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use fearless_simd_macros::simd;

struct Fixture;

trait Operation {
    fn operation<S: fearless_simd::Simd>(simd: S) {
        let _ = simd;
    }
}

impl Operation for Fixture {
    #[simd]
    default fn operation<S: fearless_simd::Simd>(simd: S) {
        let _ = simd;
    }
}

fn main() {}
