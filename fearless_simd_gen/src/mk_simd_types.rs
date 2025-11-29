// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{format_ident, quote};

use crate::{
    generic::generic_op_name,
    ops::{CORE_OPS, OpSig, TyFlavor, ops_for_type},
    types::{SIMD_TYPES, ScalarType, VecType},
};

pub(crate) fn mk_simd_types() -> TokenStream {
    let mut result = quote! {
        use crate::{Bytes, Select, Simd, SimdFrom, SimdInto, SimdCvtFloat, SimdCvtTruncate};
    };
    for ty in SIMD_TYPES {
        let name = ty.rust();
        let align = ty.n_bits() / 8;
        let align_lit = Literal::usize_unsuffixed(align);
        let len = Literal::usize_unsuffixed(ty.len);
        let rust_scalar = ty.scalar.rust(ty.scalar_bits);
        let select = Ident::new(&format!("select_{}", ty.rust_name()), Span::call_site());
        let bytes = VecType::new(ScalarType::Unsigned, 8, align).rust();
        let mask = ty.mask_ty().rust();
        let scalar_impl = {
            let splat = Ident::new(&format!("splat_{}", ty.rust_name()), Span::call_site());
            quote! {
                impl<S: Simd> SimdFrom<#rust_scalar, S> for #name<S> {
                    #[inline(always)]
                    fn simd_from(value: #rust_scalar, simd: S) -> Self {
                        simd.#splat(value)
                    }
                }

                impl<S: Simd> core::ops::Index<usize> for #name<S> {
                    type Output = #rust_scalar;
                    #[inline(always)]
                    fn index(&self, i: usize) -> &Self::Output {
                        &self.val[i]
                    }
                }

                impl<S: Simd> core::ops::IndexMut<usize> for #name<S> {
                    #[inline(always)]
                    fn index_mut (&mut self, i: usize) -> &mut Self::Output {
                        &mut self.val[i]
                    }
                }
            }
        };
        let impl_block = simd_impl(ty);
        let simd_from_items = make_list(
            (0..ty.len)
                .map(|idx| quote! { val[#idx] })
                .collect::<Vec<_>>(),
        );
        let mut conditional_impls = Vec::new();
        // TODO: Relax `if` clauses once 64-bit integer or 16-bit floats vectors are implemented
        match ty.scalar {
            ScalarType::Float if ty.scalar_bits == 32 => {
                for src_scalar in [ScalarType::Unsigned, ScalarType::Int] {
                    let src_ty = VecType {
                        scalar: src_scalar,
                        ..*ty
                    };
                    let method = format_ident!(
                        "cvt_{}_{}",
                        ty.scalar.rust_name(ty.scalar_bits),
                        src_ty.rust_name()
                    );
                    let src_ty = src_ty.rust();
                    conditional_impls.push(quote! {
                        impl<S: Simd> SimdCvtFloat<#src_ty<S>> for #name<S> {
                            fn float_from(x: #src_ty<S>) -> Self {
                                x.simd.#method(x)
                            }
                        }
                    });
                }
            }
            ScalarType::Int | ScalarType::Unsigned if ty.scalar_bits == 32 => {
                let src_ty = VecType {
                    scalar: ScalarType::Float,
                    ..*ty
                };
                let method = format_ident!(
                    "cvt_{}_{}",
                    ty.scalar.rust_name(ty.scalar_bits),
                    src_ty.rust_name()
                );
                let src_ty = src_ty.rust();
                conditional_impls.push(quote! {
                    impl<S: Simd> SimdCvtTruncate<#src_ty<S>> for #name<S> {
                        fn truncate_from(x: #src_ty<S>) -> Self {
                            x.simd.#method(x)
                        }
                    }
                });
            }
            _ => {}
        }
        if ty.n_bits() > 128 {
            let n2 = ty.len / 2;
            let split_ty = VecType::new(ty.scalar, ty.scalar_bits, n2);
            let split_ty_rust = split_ty.rust();
            let split_method = generic_op_name("split", ty);
            conditional_impls.push(quote! {
                impl<S: Simd> crate::SimdSplit<#rust_scalar, S> for #name<S> {
                    type Split = #split_ty_rust<S>;

                    #[inline(always)]
                    fn split(self) -> (Self::Split, Self::Split) {
                        self.simd.#split_method(self)
                    }
                }
            });
        }
        if ty.n_bits() < 512 {
            let n2 = ty.len * 2;
            let combine_ty = VecType::new(ty.scalar, ty.scalar_bits, n2);
            let combine_ty_rust = combine_ty.rust();
            let combine_method = generic_op_name("combine", ty);
            conditional_impls.push(quote! {
                impl<S: Simd> crate::SimdCombine<#rust_scalar, S> for #name<S> {
                    type Combined = #combine_ty_rust<S>;

                    #[inline(always)]
                    fn combine(self, rhs: Self) -> Self::Combined {
                        self.simd.#combine_method(self, rhs)
                    }
                }
            });
        }
        result.extend(quote! {
            #[derive(Clone, Copy, Debug)]
            #[repr(C, align(#align_lit))]
            pub struct #name<S: Simd> {
                pub val: [#rust_scalar; #len],
                pub simd: S,
            }

            impl<S: Simd> SimdFrom<[#rust_scalar; #len], S> for #name<S> {
                #[inline(always)]
                fn simd_from(val: [#rust_scalar; #len], simd: S) -> Self {
                    // Note: Previously, we would just straight up copy `val`. However, at least on
                    // ARM, this would always lead to it being compiled to a `memset_pattern16`, at least
                    // for scalar f32x4, which significantly slowed down the `render_strips` benchmark.
                    // Assigning each index individually seems to circumvent this quirk.
                    // TODO: Investigate whether this has detrimental effects for other numeric
                    // types.
                    Self { val: #simd_from_items, simd }
                }
            }

            impl<S: Simd> From<#name<S>> for [#rust_scalar; #len] {
                #[inline(always)]
                fn from(value: #name<S>) -> Self {
                    value.val
                }
            }

            impl<S: Simd> core::ops::Deref for #name<S> {
                type Target = [#rust_scalar; #len];
                #[inline(always)]
                fn deref(&self) -> &Self::Target {
                    &self.val
                }
            }

            impl<S: Simd> core::ops::DerefMut for #name<S> {
                #[inline(always)]
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.val
                }
            }

            #scalar_impl

            impl<S: Simd> Select<#name<S>> for #mask<S> {
                #[inline(always)]
                fn select(self, if_true: #name<S>, if_false: #name<S>) -> #name<S> {
                    self.simd.#select(self, if_true, if_false)
                }
            }

            impl<S: Simd> Bytes for #name<S> {
                type Bytes = #bytes<S>;

                #[inline(always)]
                fn to_bytes(self) -> Self::Bytes {
                    unsafe {
                        #bytes {
                            val: core::mem::transmute(self.val),
                            simd: self.simd,
                        }
                    }
                }

                #[inline(always)]
                fn from_bytes(value: Self::Bytes) -> Self {
                    unsafe {
                        Self {
                            val: core::mem::transmute(value.val),
                            simd: value.simd,
                        }
                    }
                }
            }

            #impl_block

            #( #conditional_impls )*
        });
    }
    result
}

/// Create the impl block for the type
///
/// This may go away, as possibly all methods will be subsumed by the `vec_impl`.
fn simd_impl(ty: &VecType) -> TokenStream {
    let name = ty.rust();
    let mut methods = vec![];
    for (method, sig) in ops_for_type(ty, true) {
        // Eventually, we should get rid of all inherent methods. Right now, we keep the "convert" operations (used for
        // converting between floats and integers; there's a trait for this, but type inference makes it annoying to
        // use).
        if matches!(sig, OpSig::Cvt { .. })
            && let Some(args) = sig.vec_trait_args()
        {
            let method_name = Ident::new(method, Span::call_site());
            let trait_method = generic_op_name(method, ty);
            let ret_ty = sig.ret_ty(ty, TyFlavor::VecImpl);
            let call_args = match sig {
                OpSig::Cvt { .. } => quote! { self },
                _ => unimplemented!(),
            };
            methods.push(quote! {
                #[inline(always)]
                pub fn #method_name(#args) -> #ret_ty {
                    self.simd.#trait_method(#call_args)
                }
            });
        }
    }

    let vec_impl = simd_vec_impl(ty);
    quote! {
        impl<S: Simd> #name<S> {
            #( #methods )*
        }
        #vec_impl
    }
}

fn simd_vec_impl(ty: &VecType) -> TokenStream {
    let name = ty.rust();
    let scalar = ty.scalar.rust(ty.scalar_bits);
    let len = Literal::usize_unsuffixed(ty.len);
    let vec_trait = match ty.scalar {
        ScalarType::Float => "SimdFloat",
        ScalarType::Unsigned | ScalarType::Int => "SimdInt",
        ScalarType::Mask => "SimdMask",
    };
    let zero = match ty.scalar {
        ScalarType::Float => quote! { 0.0 },
        _ => quote! { 0 },
    };
    let vec_trait_id = Ident::new(vec_trait, Span::call_site());
    let splat = generic_op_name("splat", ty);
    let mut methods = vec![];
    for (method, sig) in ops_for_type(ty, false) {
        if CORE_OPS.contains(&method) || matches!(sig, OpSig::Combine) {
            continue;
        }
        let method_name = Ident::new(method, Span::call_site());
        let trait_method = generic_op_name(method, ty);
        if let Some(args) = sig.vec_trait_args() {
            let ret_ty = sig.ret_ty(ty, TyFlavor::VecImpl);
            let call_args = match sig {
                OpSig::Unary => quote! { self },
                OpSig::Binary
                | OpSig::Compare
                | OpSig::Combine
                | OpSig::Zip { .. }
                | OpSig::Unzip { .. } => {
                    quote! { self, rhs.simd_into(self.simd) }
                }
                OpSig::Ternary => {
                    quote! { self, op1.simd_into(self.simd), op2.simd_into(self.simd) }
                }
                _ => quote! { todo!() },
            };
            methods.push(quote! {
                #[inline(always)]
                fn #method_name(#args) -> #ret_ty {
                    self.simd.#trait_method(#call_args)
                }
            });
        }
    }
    let mask_ty = ty.mask_ty().rust();
    let block_ty = VecType::new(ty.scalar, ty.scalar_bits, 128 / ty.scalar_bits).rust();
    let block_splat_body = match ty.n_bits() {
        128 => quote! {
            block
        },
        256 => {
            let n2 = ty.len / 2;
            let combine = generic_op_name("combine", &VecType::new(ty.scalar, ty.scalar_bits, n2));
            quote! {
                block.simd.#combine(block, block)
            }
        }
        512 => {
            let n2 = ty.len / 2;
            let combine2 = generic_op_name("combine", &VecType::new(ty.scalar, ty.scalar_bits, n2));
            let n4 = ty.len / 4;
            let combine4 = generic_op_name("combine", &VecType::new(ty.scalar, ty.scalar_bits, n4));
            quote! {
                let block2 = block.simd.#combine4(block, block);
                block2.simd.#combine2(block2, block2)
            }
        }
        _ => unreachable!(),
    };
    quote! {
        impl<S: Simd> crate::SimdBase<#scalar, S> for #name<S> {
            const N: usize = #len;
            type Mask = #mask_ty<S>;
            type Block = #block_ty<S>;

            #[inline(always)]
            fn witness(&self) -> S {
                self.simd
            }

            #[inline(always)]
            fn as_slice(&self) -> &[#scalar] {
                &self.val
            }

            #[inline(always)]
            fn as_mut_slice(&mut self) -> &mut [#scalar] {
                &mut self.val
            }

            #[inline(always)]
            fn from_slice(simd: S, slice: &[#scalar]) -> Self {
                let mut val = [#zero; #len];
                val.copy_from_slice(slice);
                Self { val, simd }
            }

            #[inline(always)]
            fn splat(simd: S, val: #scalar) -> Self {
                simd.#splat(val)
            }

            #[inline(always)]
            fn block_splat(block: Self::Block) -> Self {
                #block_splat_body
            }

            #[inline(always)]
            fn from_fn(simd: S, f: impl FnMut(usize) -> #scalar) -> Self {
                Self {
                    val: core::array::from_fn(f),
                    simd,
                }
            }

        }
        impl<S: Simd> crate::#vec_trait_id<#scalar, S> for #name<S> {
            #( #methods )*
        }
    }
}

fn make_list(items: Vec<TokenStream>) -> TokenStream {
    quote!([#( #items, )*])
}
