// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//@ compile-flags: --emit=link

use fearless_simd::SimdBase;

fn main() {
    let value = fearless_simd::u32x4::from_slice(fearless_simd::Fallback::new(), &[10, 20, 30, 40]);
    let _ = fearless_simd::simd_swizzle!(value, [0, 1, 2]);
    //~^ ERROR: swizzle index count must equal
}
