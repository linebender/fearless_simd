// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Assembly inspection fixtures for the exact-width SVE2 prototype with bounded memory accesses.
//! See SVE2_PROTOTYPE.md for build and execution instructions.

#[cfg(all(target_arch = "aarch64", feature = "experimental_sve2"))]
mod probes {
    use fearless_simd::*;

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_math_128(simd: Sve2_128, a: f32x4<Sve2_128>, b: f32x4<Sve2_128>, c: f32x4<Sve2_128>) -> f32x4<Sve2_128> {
            let _ = simd;
            ((a + b) * c - b) / c
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_math_256(simd: Sve2_256, a: f32x8<Sve2_256>, b: f32x8<Sve2_256>, c: f32x8<Sve2_256>) -> f32x8<Sve2_256> {
            let _ = simd;
            ((a + b) * c - b) / c
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_math_512(simd: Sve2_512, a: f32x16<Sve2_512>, b: f32x16<Sve2_512>, c: f32x16<Sve2_512>) -> f32x16<Sve2_512> {
            let _ = simd;
            ((a + b) * c - b) / c
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_swizzle_128(simd: Sve2_128, a: u8x16<Sve2_128>, indices: u8x16<Sve2_128>) -> u8x16<Sve2_128> {
            let _ = simd;
            a.swizzle_dyn_precise(indices)
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_swizzle_256(simd: Sve2_256, a: u8x32<Sve2_256>, indices: u8x32<Sve2_256>) -> u8x32<Sve2_256> {
            let _ = simd;
            a.swizzle_dyn_precise(indices)
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_swizzle_512(simd: Sve2_512, a: u8x64<Sve2_512>, indices: u8x64<Sve2_512>) -> u8x64<Sve2_512> {
            let _ = simd;
            a.swizzle_dyn_precise(indices)
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_concat_256(simd: Sve2_256, a: u8x32<Sve2_256>, b: u8x32<Sve2_256>, indices: u8x32<Sve2_256>) -> u8x32<Sve2_256> {
            let _ = simd;
            a.concat_swizzle_dyn_precise(b, indices)
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_block_swizzle_512(simd: Sve2_512, a: u8x64<Sve2_512>, indices: u8x64<Sve2_512>) -> u8x64<Sve2_512> {
            let _ = simd;
            a.swizzle_dyn_within_blocks(indices)
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_rearrange_256(simd: Sve2_256, a: f32x8<Sve2_256>, b: f32x8<Sve2_256>) -> f32x8<Sve2_256> {
            let _ = simd;
            a.reverse().zip_high(b).unzip_low(a)
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_interleave_512(simd: Sve2_512, a: u32x16<Sve2_512>, b: u32x16<Sve2_512>) -> (u32x16<Sve2_512>, u32x16<Sve2_512>) {
            let _ = simd;
            a.interleave(b)
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_fma_sqrt_256(simd: Sve2_256, a: f64x4<Sve2_256>, b: f64x4<Sve2_256>, c: f64x4<Sve2_256>) -> f64x4<Sve2_256> {
            let _ = simd;
            a.mul_add(b, c).sqrt()
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_integer_sum_512(simd: Sve2_512, a: u32x16<Sve2_512>, b: u32x16<Sve2_512>) -> u32 {
            let _ = simd;
            (a * b).reduce_sum()
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_subwidth_128(simd: Sve2_128, a: f32x4<Sve2_128>, b: f32x4<Sve2_128>) -> f32x4<Sve2_128> {
            let _ = simd;
            (a + b).reverse().zip_high(b)
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_subwidth_256(simd: Sve2_256, a: f32x4<Sve2_256>, b: f32x4<Sve2_256>) -> f32x4<Sve2_256> {
            let _ = simd;
            (a + b).reverse().zip_high(b)
        }
    }

    kernel! {
        #[unsafe(no_mangle)]
        pub fn sve2_subwidth_512(simd: Sve2_512, a: f32x4<Sve2_512>, b: f32x4<Sve2_512>) -> f32x4<Sve2_512> {
            let _ = simd;
            (a + b).reverse().zip_high(b)
        }
    }
}

fn main() {
    println!("{:?}", fearless_simd::Level::new());
}
