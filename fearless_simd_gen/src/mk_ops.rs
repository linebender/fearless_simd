// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

use crate::{
    ops::{CoreOpTrait, OpSig, TyFlavor, overloaded_ops_for},
    types::{SIMD_TYPES, type_imports},
};

pub(crate) fn mk_ops() -> TokenStream {
    let imports = type_imports();

    let mut impls = vec![];

    for ty in SIMD_TYPES {
        let simd = ty.rust();
        for (op, sig, doc) in overloaded_ops_for(ty.scalar) {
            let opfn = op.op_fn();
            let trait_name = op.trait_name();
            let simd_name = op.simd_name();
            let op_assign_fn = format_ident!("{opfn}_assign");
            let trait_id = Ident::new(trait_name, Span::call_site());
            let trait_assign_id = format_ident!("{trait_name}Assign");
            let simd_fn_name = format!("{simd_name}_{}", ty.rust_name());
            let simd_fn = Ident::new(&simd_fn_name, Span::call_site());
            let opfn = Ident::new(opfn, Span::call_site());
            let doc = sig.format_docstring(doc, TyFlavor::VecImpl);

            match op {
                CoreOpTrait::ShrVectored => {
                    impls.push(quote! {
                        impl<S: Simd> core::ops::#trait_id for #simd<S> {
                            type Output = Self;
                            #[doc = #doc]
                            #[inline(always)]
                            fn #opfn(self, rhs: Self) -> Self::Output {
                                self.simd.#simd_fn(self, rhs)
                            }
                        }

                        impl<S: Simd> core::ops::#trait_assign_id for #simd<S> {
                            #[doc = #doc]
                            #[inline(always)]
                            fn #op_assign_fn(&mut self, rhs: Self) {
                                *self = self.simd.#simd_fn(*self, rhs);
                            }
                        }
                    });
                }
                CoreOpTrait::Shl | CoreOpTrait::Shr => {
                    impls.push(quote! {
                        impl<S: Simd> core::ops::#trait_id<u32> for #simd<S> {
                            type Output = Self;
                            #[doc = #doc]
                            #[inline(always)]
                            fn #opfn(self, rhs: u32) -> Self::Output {
                                self.simd.#simd_fn(self, rhs)
                            }
                        }

                        impl<S: Simd> core::ops::#trait_assign_id<u32> for #simd<S> {
                            #[inline(always)]
                            fn #op_assign_fn(&mut self, rhs: u32) {
                                *self = self.simd.#simd_fn(*self, rhs);
                            }
                        }
                    });
                }
                _ if matches!(sig, OpSig::Unary) => {
                    impls.push(quote! {
                        impl<S: Simd> core::ops::#trait_id for #simd<S> {
                            type Output = Self;
                            #[doc = #doc]
                            #[inline(always)]
                            fn #opfn(self) -> Self::Output {
                                self.simd.#simd_fn(self)
                            }
                        }
                    });
                }
                _ => {
                    let scalar = ty.scalar.rust(ty.scalar_bits);
                    impls.push(quote! {
                        impl<S: Simd> core::ops::#trait_id for #simd<S> {
                            type Output = Self;
                            #[doc = #doc]
                            #[inline(always)]
                            fn #opfn(self, rhs: Self) -> Self::Output {
                                self.simd.#simd_fn(self, rhs)
                            }
                        }

                        impl<S: Simd> core::ops::#trait_assign_id for #simd<S> {
                            #[doc = #doc]
                            #[inline(always)]
                            fn #op_assign_fn(&mut self, rhs: Self) {
                                *self = self.simd.#simd_fn(*self, rhs);
                            }
                        }

                        impl<S: Simd> core::ops::#trait_id<#scalar> for #simd<S> {
                            type Output = Self;
                            #[doc = #doc]
                            #[inline(always)]
                            fn #opfn(self, rhs: #scalar) -> Self::Output {
                                self.simd.#simd_fn(self, rhs.simd_into(self.simd))
                            }
                        }

                        impl<S: Simd> core::ops::#trait_assign_id<#scalar> for #simd<S> {
                            #[doc = #doc]
                            #[inline(always)]
                            fn #op_assign_fn(&mut self, rhs: #scalar) {
                                *self = self.simd.#simd_fn(*self, rhs.simd_into(self.simd));
                            }
                        }

                        impl<S: Simd> core::ops::#trait_id<#simd<S>> for #scalar {
                            type Output = #simd<S>;
                            #[doc = #doc]
                            #[inline(always)]
                            fn #opfn(self, rhs: #simd<S>) -> Self::Output {
                                rhs.simd.#simd_fn(self.simd_into(rhs.simd), rhs)
                            }
                        }
                    });
                }
            }
        }
    }

    quote! {
        use crate::{Simd, SimdInto};
        #imports
        #( #impls )*
    }
}
