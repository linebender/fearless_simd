#![cfg_attr(not(feature = "std"), no_std)]
use simd_research_macros::simd;

#[simd]
pub fn mix(x: u32) -> u32 {
    x.wrapping_mul(0x9e3779b9).rotate_left(7)
}

#[inline]
pub fn ordinary(x: u32) -> u32 {
    x.wrapping_add(13)
}

#[simd]
#[inline(never)]
pub fn opaque_mix(x: u32) -> u32 {
    x.wrapping_mul(0x9e3779b9).rotate_left(7)
}
