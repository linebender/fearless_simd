// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

fn main() {
    let value = fearless_simd::u32x4::from_slice(fearless_simd::Fallback::new(), &[10, 20, 30, 40]);
    let indices = [0, 1, 2, 3];
    let _ = fearless_simd::simd_swizzle!(value, indices);
    //~^ ERROR: attempt to use a non-constant value in a constant
}

use fearless_simd::SimdBase;
