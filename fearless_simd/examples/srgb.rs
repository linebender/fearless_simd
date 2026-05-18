// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    missing_docs,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

use fearless_simd::{Level, dispatch, f32x4, prelude::*};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{float32x4_t, vcopyq_laneq_f32};
#[cfg(target_arch = "x86")]
use core::arch::x86::{__m128, _mm_blend_ps};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m128, _mm_blend_ps};

fearless_simd::kernel! {
    #[inline]
    fn copy_alpha_neon(neon: Neon, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vcopyq_laneq_f32::<3, 3>(a, b)
    }
}

fearless_simd::kernel! {
    #[inline]
    fn copy_alpha_sse4_2(sse4_2: Sse4_2, a: __m128, b: __m128) -> __m128 {
        _mm_blend_ps::<8>(a, b)
    }
}

#[inline(always)]
fn copy_alpha<S: Simd>(a: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(sse4_2) = a.simd.level().as_sse4_2() {
        return copy_alpha_sse4_2(sse4_2, a.into(), b.into()).simd_into(a.simd);
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = a.simd.level().as_neon() {
        return copy_alpha_neon(neon, a.into(), b.into()).simd_into(a.simd);
    }

    let mut result = a;
    result[3] = b[3];
    result
}

#[inline(always)]
fn to_srgb<S: Simd>(simd: S, rgba: [f32; 4]) -> [f32; 4] {
    let v: f32x4<S> = rgba.simd_into(simd);
    let vabs = v.abs();
    let x = vabs - 5.358_626_4e-4;
    let x2 = x * x;
    let even1 = x * -9.127_959e-1 + -2.881_431_4e-2;
    let even2 = x2 * -7.291_929e-1 + even1;
    let odd1 = x * 1.061_331_7 + 1.401_945_4;
    let odd2 = x2 * 2.077_583e-1 + odd1;
    let poly = odd2 * x.sqrt() + even2;
    let lin = vabs * 12.92;
    let z = vabs.simd_gt(0.0031308).select(poly, lin);
    let z_signed = z.copysign(v);
    let result = copy_alpha(z_signed, v);
    result.into()
}

fn main() {
    let level = Level::new();
    let rgba = [0.1, -0.2, 0.001, 0.4];
    let srgb = dispatch!(level, simd=> to_srgb(simd, rgba));
    println!("{srgb:?}");
}
