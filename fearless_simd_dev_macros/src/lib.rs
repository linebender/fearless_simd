//! Development macros for fearless_simd.

use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Create test for checking consistency between different SIMD backends. 
#[proc_macro_attribute]
pub fn simd_test(_: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    let input_fn_name = input_fn.sig.ident.clone();
    let scalar_name = Ident::new(
        &format!("{input_fn_name}_scalar"),
        input_fn_name.span(),
    );
    let neon_name = Ident::new(
        &format!("{input_fn_name}_scalar"),
        input_fn_name.span(),
    );
    
    let scalar_snippet =  quote! {
        #input_fn
        
        #[test]
        fn #scalar_name() {
            let fallback = fearless_simd::Fallback::new();
            #input_fn_name(fallback);
        }
    };
    
    quote! {
        #scalar_snippet
    }.into()
}