// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    missing_docs,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

use fearless_simd::{Level, Select, Simd, SimdFloat, SimdInto, dispatch, f32x4};

// This block shows how to use safe wrappers for compile-time enforcement
// of using valid SIMD intrinsics.
#[cfg(feature = "safe_wrappers")]
#[inline(always)]
fn copy_alpha<S: Simd>(a: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    // #[cfg(target_arch = "x86_64")]
    // if let Some(avx2) = a.simd.level().as_avx2() {
    //     return avx2
    //         .sse4_1
    //         ._mm_blend_ps::<8>(a.into(), b.into())
    //         .simd_into(a.simd);
    // }
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = a.simd.level().as_neon() {
        return neon
            .neon
            .vcopyq_laneq_f32::<3, 3>(a.into(), b.into())
            .simd_into(a.simd);
    }
    let mut result = a;
    result[3] = b[3];
    result
}

// This block lets the example compile without safe wrappers.
#[cfg(not(feature = "safe_wrappers"))]
#[inline(always)]
fn copy_alpha<S: Simd>(a: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    #[cfg(target_arch = "aarch64")]
    if let Some(_neon) = a.simd.level().as_neon() {
        unsafe {
            return core::arch::aarch64::vcopyq_laneq_f32::<3, 3>(a.into(), b.into())
                .simd_into(a.simd);
        }
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
