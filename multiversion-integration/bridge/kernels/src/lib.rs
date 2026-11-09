#![no_std]

use fearless_simd::{dispatch, Level};

#[cfg(feature = "std")]
pub fn mix(x: u32) -> u32 {
    dispatch!(Level::new(), simd => mix::__simd(simd, x))
}

// Hand-expanded form of a generated companion namespace. It uses only
// documented public APIs, and names no architecture or target features.
pub mod mix {
    use fearless_simd::Simd;

    #[inline(always)]
    pub fn __simd<S: Simd>(simd: S, x: u32) -> u32 {
        simd.vectorize(#[inline(always)] move || x.wrapping_mul(3).wrapping_add(7))
    }
}

pub fn double_at_level(level: Level, xs: &mut [u32]) {
    dispatch!(level, simd => double::__simd(simd, xs))
}

pub mod double {
    use fearless_simd::prelude::*;

    #[inline(always)]
    pub fn __simd<S: Simd>(simd: S, xs: &mut [u32]) {
        simd.vectorize(#[inline(always)] move || {
            let mut chunks = xs.chunks_exact_mut(S::u32s::LEN);
            for chunk in &mut chunks {
                (S::u32s::from_slice(simd, chunk) * 2).store_slice(chunk);
            }
            for x in chunks.into_remainder() {
                *x = x.wrapping_mul(2);
            }
        });
    }
}
