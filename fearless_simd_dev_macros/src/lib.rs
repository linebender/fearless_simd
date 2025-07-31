//! Development macros for fearless_simd.

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

    let include_fallback = !exclude_fallback(&input_fn_name.to_string());
    let include_neon = {
        #[cfg(not(target_arch = "aarch64"))]
        let temp = false;
        #[cfg(target_arch = "aarch64")]
        let temp = std::arch::is_aarch64_feature_detected!("neon");

        temp & !exclude_neon(&input_fn_name.to_string())
    };

    let fallback_snippet = if include_fallback {
        quote! {
            #[test]
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
            #[test]
            fn #neon_name() {
                let neon = unsafe { fearless_simd::Neon::new_unchecked() };
                #input_fn_name(neon);
            }
        }
    } else {
        quote! {}
    };

    quote! {
        #input_fn

        #fallback_snippet
        #neon_snippet
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
fn exclude_wasm(name: &str) -> bool {
    false
}
