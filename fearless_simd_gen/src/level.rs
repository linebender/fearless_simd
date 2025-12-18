// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

use crate::types::{SIMD_TYPES, VecType};

pub(crate) trait Level {
    fn name(&self) -> &'static str;

    fn token(&self) -> Ident {
        Ident::new(self.name(), Span::call_site())
    }

    fn impl_arch_types(
        &self,
        max_block_size: usize,
        arch_ty: &dyn Fn(&VecType) -> TokenStream,
    ) -> TokenStream {
        let mut assoc_types = vec![];
        for vec_ty in SIMD_TYPES {
            let ty_ident = vec_ty.rust();
            let wrapper_ty = vec_ty.aligned_wrapper_ty(arch_ty, max_block_size);
            assoc_types.push(quote! {
                type #ty_ident = #wrapper_ty;
            });
        }
        let level_tok = self.token();

        quote! {
            impl ArchTypes for #level_tok {
                #( #assoc_types )*
            }
        }
    }
}
