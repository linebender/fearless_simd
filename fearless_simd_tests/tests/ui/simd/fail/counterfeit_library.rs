// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![forbid(unsafe_code)]

use fearless_simd_macros::simd;

// Re-exporting the genuine helper must not let a counterfeit namespace supply
// its own proof that a target feature is available.
#[allow(dead_code)]
mod fearless_simd {
    pub use ::fearless_simd::__fearless_simd_dispatch;

    #[derive(Clone, Copy)]
    pub struct CounterfeitToken;

    pub enum Level {
        Avx512(CounterfeitToken),
    }

    pub trait Simd {
        fn level(self) -> Level;

        fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R
        where
            Self: Sized,
        {
            f()
        }
    }

    impl Simd for CounterfeitToken {
        fn level(self) -> Level {
            Level::Avx512(self)
        }
    }
}

#[simd]
//~^ E0277
fn counterfeit<S: fearless_simd::Simd>(simd: S, value: u32) -> u32 {
    let _ = simd.level();
    value
}

fn main() {
    let _ = counterfeit(fearless_simd::CounterfeitToken, 1);
}
