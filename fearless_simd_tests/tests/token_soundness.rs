// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

fearless_simd::kernel!(
    fn create_neon_token(_neon: Neon) -> fearless_simd::Neon {
        fearless_simd::Neon::assume_supported()
    }
);

fearless_simd::kernel!(
    fn create_wasm_simd128_token(_wasm_simd128: WasmSimd128) -> fearless_simd::WasmSimd128 {
        fearless_simd::WasmSimd128::assume_supported()
    }
);

fearless_simd::kernel!(
    fn create_sse2_token(_sse2: Sse2) -> fearless_simd::Sse2 {
        fearless_simd::Sse2::assume_supported()
    }
);

fearless_simd::kernel!(
    fn create_sse4_2_token(_sse4_2: Sse4_2) -> fearless_simd::Sse4_2 {
        fearless_simd::Sse4_2::assume_supported()
    }
);

fearless_simd::kernel!(
    fn create_avx2_token(_avx2: Avx2) -> fearless_simd::Avx2 {
        fearless_simd::Avx2::assume_supported()
    }
);

fearless_simd::kernel!(
    fn create_avx512_token(_avx512: Avx512) -> fearless_simd::Avx512 {
        fearless_simd::Avx512::assume_supported()
    }
);

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_kernel_features_allow_safe_token_creation() {
    let Some(neon) = fearless_simd::Level::new().as_neon() else {
        return;
    };

    let _ = create_neon_token(neon);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn wasm_simd128_kernel_allows_safe_token_creation() {
    let wasm_simd128 = fearless_simd::Level::new()
        .as_wasm_simd128()
        .expect("WASM SIMD128 should be available when +simd128 is enabled");

    let _ = create_wasm_simd128_token(wasm_simd128);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_kernel_features_allow_safe_token_creation() {
    let Some(sse2) = fearless_simd::Level::new().as_sse2() else {
        return;
    };

    let _ = create_sse2_token(sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse4_2_kernel_features_allow_safe_token_creation() {
    let Some(sse4_2) = fearless_simd::Level::new().as_sse4_2() else {
        return;
    };

    let _ = create_sse4_2_token(sse4_2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_kernel_features_allow_safe_token_creation() {
    let Some(avx2) = fearless_simd::Level::new().as_avx2() else {
        return;
    };

    let _ = create_avx2_token(avx2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_kernel_features_allow_safe_token_creation() {
    let Some(avx512) = fearless_simd::Level::new().as_avx512() else {
        return;
    };

    let _ = create_avx512_token(avx512);
}
