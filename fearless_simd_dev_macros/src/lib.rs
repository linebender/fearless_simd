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

    let ignore_attr = |f: fn(&str) -> bool| {
        let should_ignore = input_fn
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("ignore"))
            || f(&input_fn_name.to_string());
        if should_ignore {
            quote! { #[ignore] }
        } else {
            quote! {}
        }
    };

    let ignore_fallback = ignore_attr(exclude_fallback);
    let ignore_neon = ignore_attr(exclude_neon);
    let ignore_sse4 = ignore_attr(exclude_sse4);
    let ignore_avx2 = ignore_attr(exclude_avx2);
    let ignore_wasm = ignore_attr(exclude_wasm);

    let fallback_snippet = quote! {
        #[test]
        #ignore_fallback
        fn #fallback_name() {
            let fallback = fearless_simd::Fallback::new();
            #input_fn_name(fallback);
        }
    };

    // All of the architecture-specific tests need to be included every time, and #[cfg]'d out depending on the target
    // architecture. We can't use `CARGO_CFG_TARGET_ARCH` to conditionally omit them because it's not available when
    // proc macros are evaluated.

    // There is currently no way to conditionally ignore a test at runtime (see
    // https://internals.rust-lang.org/t/pre-rfc-skippable-tests/14611). Instead, we'll just pass the tests if the
    // target features aren't supported. This is not ideal, since it may mislead you into thinking tests have passed
    // when they haven't even been run, but some CI runners don't support all target features and we don't want failures
    // as a result of that.

    let neon_snippet = quote! {
        #[cfg(target_arch = "aarch64")]
        #[test]
        #ignore_neon
        fn #neon_name() {
            if std::arch::is_aarch64_feature_detected!("neon") {
                let neon = unsafe { fearless_simd::aarch64::Neon::new_unchecked() };
                #input_fn_name(neon);
            }
        }
    };

    let sse4_snippet = quote! {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[test]
        #ignore_sse4
        fn #sse4_name() {
            if std::arch::is_x86_feature_detected!("sse4.2") {
                let sse4 = unsafe { fearless_simd::x86::Sse4_2::new_unchecked() };
                #input_fn_name(sse4);
            }
        }
    };

    let avx2_snippet = quote! {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[test]
        #ignore_avx2
        fn #avx2_name() {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                let avx2 = unsafe { fearless_simd::x86::Avx2::new_unchecked() };
                #input_fn_name(avx2);
            }
        }
    };

    let wasm_snippet = quote! {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        #[test]
        #ignore_wasm
        fn #wasm_name() {
            let wasm = unsafe { fearless_simd::wasm32::WasmSimd128::new_unchecked() };
            #input_fn_name(wasm);
        }
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

fn exclude_neon(_test_name: &str) -> bool {
    false
}

fn exclude_fallback(_test_name: &str) -> bool {
    false
}

fn exclude_sse4(_test_name: &str) -> bool {
    false
}

fn exclude_avx2(_test_name: &str) -> bool {
    false
}

fn exclude_wasm(_test_name: &str) -> bool {
    false
}
