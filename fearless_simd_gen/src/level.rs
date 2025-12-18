// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

use crate::{
    generic::generic_op,
    ops::{Op, ops_for_type},
    types::{SIMD_TYPES, ScalarType, VecType},
};

pub(crate) trait Level {
    fn name(&self) -> &'static str;
    fn native_width(&self) -> usize;
    fn make_vectorize_body(&self) -> TokenStream;
    fn make_method(&self, op: Op, vec_ty: &VecType) -> TokenStream;

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

    fn make_level_body(&self) -> TokenStream {
        let level_tok = self.token();

        quote! {
            Level::#level_tok(self)
        }
    }

    fn mk_simd_impl(&self) -> TokenStream {
        let level_tok = self.token();
        let native_width = self.native_width();
        let mut methods = vec![];
        for vec_ty in SIMD_TYPES {
            for op in ops_for_type(vec_ty) {
                if op.sig.should_use_generic_op(vec_ty, native_width) {
                    methods.push(generic_op(&op, vec_ty));
                    continue;
                }

                let method = self.make_method(op, vec_ty);
                methods.push(method);
            }
        }

        let vectorize_body = self.make_vectorize_body();
        let level_body = self.make_level_body();

        let mut assoc_types = vec![];
        for (scalar, scalar_bits) in [
            (ScalarType::Float, 32),
            (ScalarType::Float, 64),
            (ScalarType::Unsigned, 8),
            (ScalarType::Int, 8),
            (ScalarType::Unsigned, 16),
            (ScalarType::Int, 16),
            (ScalarType::Unsigned, 32),
            (ScalarType::Int, 32),
            (ScalarType::Mask, 8),
            (ScalarType::Mask, 16),
            (ScalarType::Mask, 32),
            (ScalarType::Mask, 64),
        ] {
            let native_width_ty = VecType::new(scalar, scalar_bits, native_width / scalar_bits);
            let name = native_width_ty.rust();
            let native_width_name = scalar.native_width_name(scalar_bits);
            assoc_types.push(quote! {
                type #native_width_name = #name<Self>;
            });
        }

        quote! {
            impl Simd for #level_tok {
                #( #assoc_types )*

                #[inline(always)]
                fn level(self) -> Level {
                    #level_body
                }

                #[inline]
                fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                    #vectorize_body
                }

                #( #methods )*
            }
        }
    }
}
