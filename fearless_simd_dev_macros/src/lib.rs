// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Development macros for `fearless_simd`.

use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

/// Create test for checking consistency between different SIMD backends.
#[proc_macro_attribute]
pub fn simd_test(_: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let input_fn_name = input_fn.sig.ident.clone();

    let get_ident =
        |name: &str| Ident::new(&format!("{input_fn_name}_{name}"), input_fn_name.span());

    let fallback_name = get_ident("fallback");
    let neon_name = get_ident("neon");
    let sse4_name = get_ident("sse4");
    let avx2_name = get_ident("avx2");
    let wasm_name = get_ident("wasm");

    let include_fallback = !exclude_fallback(&input_fn_name.to_string());
    #[cfg(target_arch = "aarch64")]
    let include_neon = std::arch::is_aarch64_feature_detected!("neon")
        && !exclude_neon(&input_fn_name.to_string());
    #[cfg(not(target_arch = "aarch64"))]
    let include_neon = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let include_sse4 =
        std::arch::is_x86_feature_detected!("sse4.2") && !exclude_sse4(&input_fn_name.to_string());
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let include_sse4 = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let include_avx2 = std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && !exclude_avx2(&input_fn_name.to_string());
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let include_avx2 = false;
    // Note that we cannot feature-gate this with `target_arch`. If we run
    // `wasm-pack test --headless --chrome`, then the `target_arch` will still be set to
    // the operating system you are running on. Because of this, we instead add the `target_arch`
    // feature gate to the actual test.
    let include_wasm = !exclude_wasm(&input_fn_name.to_string());

    let fallback_snippet = if include_fallback {
        quote! {
            #[test]
            #[cfg_attr(all(target_arch = "wasm32", target_feature = "simd128"), wasm_bindgen_test::wasm_bindgen_test)]
            fn #fallback_name() {
                let fallback = fearless_simd::Fallback::new();
                #input_fn_name(fallback);
            }
        }
    } else {
        quote! {}
    };

    let neon_snippet = if include_neon {
        quote! {
            #[cfg(target_arch = "aarch64")]
            #[test]
            fn #neon_name() {
                let neon = unsafe { fearless_simd::aarch64::Neon::new_unchecked() };
                #input_fn_name(neon);
            }
        }
    } else {
        quote! {}
    };

    let sse4_snippet = if include_sse4 {
        quote! {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[test]
            fn #sse4_name() {
                let sse4 = unsafe { fearless_simd::x86::Sse4_2::new_unchecked() };
                #input_fn_name(sse4);
            }
        }
    } else {
        quote! {}
    };

    let avx2_snippet = if include_avx2 {
        quote! {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            #[test]
            fn #avx2_name() {
                let avx2 = unsafe { fearless_simd::x86::Avx2::new_unchecked() };
                #input_fn_name(avx2);
            }
        }
    } else {
        quote! {}
    };

    let wasm_snippet = if include_wasm {
        quote! {
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            #[wasm_bindgen_test::wasm_bindgen_test]
            fn #wasm_name() {
                let wasm = unsafe { fearless_simd::wasm32::WasmSimd128::new_unchecked() };
                #input_fn_name(wasm);
            }
        }
    } else {
        quote! {}
    };

    quote! {
        #input_fn

        #fallback_snippet
        #neon_snippet
        #wasm_snippet
        #sse4_snippet
        #avx2_snippet
    }
    .into()
}

// You can update below functions if you want to exclude certain tests from different architectures
// (for example because they haven't been implemented yet).

#[allow(dead_code, reason = "on purpose.")]
#[allow(unused_variables, reason = "on purpose.")]
fn exclude_neon(name: &str) -> bool {
    false
}

#[allow(dead_code, reason = "on purpose.")]
#[allow(unused_variables, reason = "on purpose.")]
fn exclude_fallback(name: &str) -> bool {
    false
}

#[allow(dead_code, reason = "on purpose.")]
#[allow(unused_variables, reason = "on purpose.")]
fn exclude_sse4(name: &str) -> bool {
    matches!(
        name,
        // works incorrectly for any values larger than i32::MAX and smaller than 0.
        "cvt_u32_f32x4" | "cvt_f32_u32x4" | "saturate_float_to_int",
    ) || name.contains("precise")
}

#[allow(dead_code, reason = "on purpose.")]
#[allow(unused_variables, reason = "on purpose.")]
fn exclude_avx2(name: &str) -> bool {
    matches!(
        name,
        // works incorrectly for any values larger than i32::MAX and smaller than 0.
        "cvt_u32_f32x4" | "cvt_f32_u32x4" | "saturate_float_to_int",
    ) || name.contains("precise")
}

#[allow(dead_code, reason = "on purpose.")]
#[allow(unused_variables, reason = "on purpose.")]
fn exclude_wasm(name: &str) -> bool {
    matches!(
        name,
        "min_precise_f32x4_with_nan" | "max_precise_f32x4_with_nan"
    )
}
