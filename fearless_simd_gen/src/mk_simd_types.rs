// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{format_ident, quote};

use crate::{
    generic::generic_op_name,
    ops::{
        F32_TO_I32, F32_TO_I32_PRECISE, F32_TO_U32, F32_TO_U32_PRECISE, I32_TO_F32, Op, TyFlavor,
        U32_TO_F32, vec_trait_ops_for,
    },
    types::{SIMD_TYPES, ScalarType, VecType},
};

pub(crate) fn mk_simd_types() -> TokenStream {
    let mut result = quote! {
        use crate::{Bytes, Select, Simd, SimdBase, SimdFrom, SimdInto, SimdGather, SimdScatter, SimdCvtFloat, SimdCvtTruncate};
    };
    for ty in SIMD_TYPES {
        let name = ty.rust();
        let name_str = ty.rust_name();
        let doc = ty.docstring();
        let align = ty.n_bits() / 8;
        let align_lit = Literal::usize_unsuffixed(align);
        let len = Literal::usize_unsuffixed(ty.len);
        let rust_scalar = ty.scalar.rust(ty.scalar_bits);
        let select = generic_op_name("select", ty);
        let from_array_op = generic_op_name("load_array", ty);
        let as_array_op = generic_op_name("as_array", ty);
        let as_array_ref_op = generic_op_name("as_array_ref", ty);
        let as_array_mut_op = generic_op_name("as_array_mut", ty);
        let from_bytes_op = generic_op_name("cvt_from_bytes", ty);
        let to_bytes_op = generic_op_name("cvt_to_bytes", ty);
        let bytes = VecType::new(ScalarType::Unsigned, 8, align).rust();
        let mask = ty.mask_ty().rust();

        let scalar_impl = {
            let splat = Ident::new(&format!("splat_{}", ty.rust_name()), Span::call_site());
            quote! {
                impl<S: Simd> SimdFrom<#rust_scalar, S> for #name<S> {
                    #[inline(always)]
                    fn simd_from(simd: S, value: #rust_scalar) -> Self {
                        simd.#splat(value)
                    }
                }

                impl<S: Simd> core::ops::Index<usize> for #name<S> {
                    type Output = #rust_scalar;
                    #[inline(always)]
                    fn index(&self, i: usize) -> &Self::Output {
                        &self.simd.#as_array_ref_op(self)[i]
                    }
                }

                impl<S: Simd> core::ops::IndexMut<usize> for #name<S> {
                    #[inline(always)]
                    fn index_mut (&mut self, i: usize) -> &mut Self::Output {
                        &mut self.simd.#as_array_mut_op(self)[i]
                    }
                }
            }
        };
        let impl_block = simd_vec_impl(ty);
        let mut conditional_impls = Vec::new();

        // Conversion operations
        // TODO: Relax `if` clauses once 64-bit integer or 16-bit floats vectors are implemented
        match ty.scalar {
            ScalarType::Float if ty.scalar_bits == 32 => {
                for src_scalar in [ScalarType::Unsigned, ScalarType::Int] {
                    let src_ty = ty.cast(src_scalar);
                    let method = format_ident!(
                        "cvt_{}_{}",
                        ty.scalar.rust_name(ty.scalar_bits),
                        src_ty.rust_name()
                    );
                    let src_ty = src_ty.rust();
                    let op = match src_scalar {
                        ScalarType::Unsigned => U32_TO_F32,
                        ScalarType::Int => I32_TO_F32,
                        _ => unreachable!(),
                    };
                    let doc = op.format_docstring(TyFlavor::VecImpl);
                    conditional_impls.push(quote! {
                        impl<S: Simd> SimdCvtFloat<#src_ty<S>> for #name<S> {
                            #[doc = #doc]
                            #[inline(always)]
                            fn float_from(x: #src_ty<S>) -> Self {
                                x.simd.#method(x)
                            }
                        }
                    });
                }
            }
            ScalarType::Int | ScalarType::Unsigned if ty.scalar_bits == 32 => {
                let src_ty = ty.cast(ScalarType::Float);
                let method = format_ident!(
                    "cvt_{}_{}",
                    ty.scalar.rust_name(ty.scalar_bits),
                    src_ty.rust_name()
                );
                let op = match ty.scalar {
                    ScalarType::Unsigned => F32_TO_U32,
                    ScalarType::Int => F32_TO_I32,
                    _ => unreachable!(),
                };
                let doc = op.format_docstring(TyFlavor::VecImpl);
                let method_precise = format_ident!(
                    "cvt_{}_precise_{}",
                    ty.scalar.rust_name(ty.scalar_bits),
                    src_ty.rust_name()
                );
                let op_precise = match ty.scalar {
                    ScalarType::Unsigned => F32_TO_U32_PRECISE,
                    ScalarType::Int => F32_TO_I32_PRECISE,
                    _ => unreachable!(),
                };
                let doc_precise = op_precise.format_docstring(TyFlavor::VecImpl);
                let src_ty = src_ty.rust();
                conditional_impls.push(quote! {
                    impl<S: Simd> SimdCvtTruncate<#src_ty<S>> for #name<S> {
                        #[doc = #doc]
                        #[inline(always)]
                        fn truncate_from(x: #src_ty<S>) -> Self {
                            x.simd.#method(x)
                        }
                        #[doc = #doc_precise]
                        #[inline(always)]
                        fn truncate_from_precise(x: #src_ty<S>) -> Self {
                            x.simd.#method_precise(x)
                        }
                    }
                });
            }
            _ => {}
        }

        // Scatter/gather operations
        if ty.scalar == ScalarType::Unsigned {
            let min_method = generic_op_name("min", ty);
            conditional_impls.push(quote! {
                impl<S: Simd> SimdGather<S> for #name<S> {
                    type Gathered<T> = [T; #len];

                    #[inline(always)]
                    fn gather<T: Copy>(self, src: &[T]) -> Self::Gathered<T> {
                        assert!(!src.is_empty(), "gather: source slice must not be empty");

                        // Check if the element type is small enough that the slice's length could (and then does)
                        // exceed its maximum value. The `size_of` check ensures that `Self::Element::MAX as usize` will
                        // never truncate/wrap.
                        let inbounds = if core::mem::size_of::<Self::Element>() < core::mem::size_of::<usize>() &&
                            src.len() > Self::Element::MAX as usize
                        {
                            // The slice is big enough to accept any index. For instance, if this is a vector of `u8`s,
                            // `Self::Element::MAX` is 255, so the slice must be at least 256 elements long.
                            self
                        } else {
                            // No `max(0)`; we do not implement `SimdGather` for signed integers.
                            //
                            // Converting `src.len() - 1` to `Self::Element` will not wrap, because if `src.len() - 1 >=
                            // Self::Element::MAX`, that means that `src.len() > Self::Element::MAX`, and we take the
                            // above branch instead.
                            self.simd.#min_method(self, ((src.len() - 1) as Self::Element).simd_into(self.simd))
                        };

                        core::array::from_fn(|i| unsafe {
                            // Safety: All elements of `inbounds` are in [0, src.len()). 0 is a valid index, because we
                            // asserted that `src` is not empty. Therefore, the index into `src` is valid. `i` will be
                            // between [0, Self::N), so the index into `inbounds` is valid.
                            *src.get_unchecked(*inbounds.get_unchecked(i) as usize)
                        })
                    }

                    #[inline(always)]
                    fn gather_into<T: Copy>(self, src: &[T], dst: &mut [T]) {
                        assert_eq!(Self::N, dst.len(), "gather_into: destination slice must have the same element count as the vector type");
                        assert!(!src.is_empty(), "gather_into: source slice must not be empty");

                        // Same logic as for `gather`. See the comments there.
                        let inbounds = if core::mem::size_of::<Self::Element>() < core::mem::size_of::<usize>() &&
                            src.len() > Self::Element::MAX as usize
                        {
                            self
                        } else {
                            self.simd.#min_method(self, ((src.len() - 1) as Self::Element).simd_into(self.simd))
                        };

                        for i in 0..Self::N {
                            unsafe {
                                // Safety: All elements of `inbounds` are in [0, src.len()). 0 is a valid index, because
                                // we asserted that `src` is not empty. Therefore, the index into `src` is valid. `i`
                                // will be between [0, Self::N), so the index into `inbounds` is valid. The index into
                                // `dst` is also valid, since we asserted above that `dst.len() == Self::N`.
                                *dst.get_unchecked_mut(i) = *src.get_unchecked(*inbounds.get_unchecked(i) as usize);
                            }
                        }
                    }
                }

                impl<S: Simd> SimdScatter<S> for #name<S> {
                    #[inline(always)]
                    fn scatter<T: Copy>(self, src: &[T], dst: &mut [T]) {
                        assert_eq!(Self::N, src.len(), "scatter: source slice must have the same element count as the vector type");
                        assert!(!dst.is_empty(), "scatter: destination slice must not be empty");

                        // Same logic as for `gather`, but for `dst`. See the comments there.
                        let inbounds = if core::mem::size_of::<Self::Element>() < core::mem::size_of::<usize>() &&
                            dst.len() > Self::Element::MAX as usize
                        {
                            self
                        } else {
                            self.simd.#min_method(self, ((dst.len() - 1) as Self::Element).simd_into(self.simd))
                        };

                        for i in 0..Self::N {
                            unsafe {
                                // Safety: All elements of `inbounds` are in [0, dst.len()). 0 is a valid index, because
                                // we asserted that `dst` is not empty. Therefore, the index into `dst` is valid. `i`
                                // will be between [0, Self::N), so the index into `inbounds` is valid. The index into
                                // `src` is also valid, since we asserted above that `src.len() == Self::N`.
                                *dst.get_unchecked_mut(*inbounds.get_unchecked(i) as usize) = *src.get_unchecked(i);
                            }
                        }
                    }
                }
            });
        }

        // Split/combine operations
        if let Some(half_ty) = ty.split_operand() {
            let half_ty_rust = half_ty.rust();
            let split_method = generic_op_name("split", ty);
            conditional_impls.push(quote! {
                impl<S: Simd> crate::SimdSplit<S> for #name<S> {
                    type Split = #half_ty_rust<S>;

                    #[inline(always)]
                    fn split(self) -> (Self::Split, Self::Split) {
                        self.simd.#split_method(self)
                    }
                }
            });
        }
        if let Some(combined_ty) = ty.combine_operand() {
            let combined_ty_rust = combined_ty.rust();
            let combine_method = generic_op_name("combine", ty);
            conditional_impls.push(quote! {
                impl<S: Simd> crate::SimdCombine<S> for #name<S> {
                    type Combined = #combined_ty_rust<S>;

                    #[inline(always)]
                    fn combine(self, rhs: impl SimdInto<Self, S>) -> Self::Combined {
                        self.simd.#combine_method(self, rhs.simd_into(self.simd))
                    }
                }
            });
        }

        result.extend(quote! {
            #[doc = #doc]
            #[derive(Clone, Copy)]
            #[repr(C, align(#align_lit))]
            pub struct #name<S: Simd> {
                pub(crate) val: S::#name,
                pub simd: S,
            }

            impl<S: Simd> SimdFrom<[#rust_scalar; #len], S> for #name<S> {
                #[inline(always)]
                fn simd_from(simd: S, val: [#rust_scalar; #len]) -> Self {
                    simd.#from_array_op(val)
                }
            }

            impl<S: Simd> From<#name<S>> for [#rust_scalar; #len] {
                #[inline(always)]
                fn from(value: #name<S>) -> Self {
                    value.simd.#as_array_op(value)
                }
            }

            impl<S: Simd> core::ops::Deref for #name<S> {
                type Target = [#rust_scalar; #len];
                #[inline(always)]
                fn deref(&self) -> &Self::Target {
                    self.simd.#as_array_ref_op(self)
                }
            }

            impl<S: Simd> core::ops::DerefMut for #name<S> {
                #[inline(always)]
                fn deref_mut(&mut self) -> &mut Self::Target {
                    self.simd.#as_array_mut_op(self)
                }
            }

            impl<S: Simd + core::fmt::Debug> core::fmt::Debug for #name<S> {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    crate::support::simd_debug_impl(f, #name_str, &self.simd, self.simd.#as_array_ref_op(self))
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
                    self.simd.#to_bytes_op(self)
                }

                #[inline(always)]
                fn from_bytes(value: Self::Bytes) -> Self {
                    value.simd.#from_bytes_op(value)
                }
            }

            #impl_block

            #( #conditional_impls )*
        });
    }
    result
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
    let vec_trait_id = Ident::new(vec_trait, Span::call_site());
    let splat = generic_op_name("splat", ty);
    let mut methods = vec![];
    for op in vec_trait_ops_for(ty.scalar) {
        let Op { sig, method, .. } = op;
        let trait_method = generic_op_name(method, ty);
        if let Some(method_sig) = op.vec_trait_method_sig() {
            let call_args = sig
                .forwarding_call_args()
                .expect("this method can be forwarded to a specific Simd function");
            methods.push(quote! {
                #[inline(always)]
                #method_sig {
                    self.simd.#trait_method(#call_args)
                }
            });
        }
    }
    let mask_ty = ty.mask_ty().rust();
    let block_ty = ty.block_ty().rust();
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
    let from_array_op = generic_op_name("load_array", ty);
    let from_array_ref_op = generic_op_name("load_array_ref", ty);
    let store_array_op = generic_op_name("store_array", ty);
    let as_array_ref_op = generic_op_name("as_array_ref", ty);
    let as_array_mut_op = generic_op_name("as_array_mut", ty);
    quote! {
        impl<S: Simd> SimdBase<S> for #name<S> {
            type Element = #scalar;
            const N: usize = #len;
            type Mask = #mask_ty<S>;
            type Block = #block_ty<S>;
            type Array = [#scalar; #len];

            #[inline(always)]
            fn witness(&self) -> S {
                self.simd
            }

            #[inline(always)]
            fn as_slice(&self) -> &[#scalar] {
                self.simd.#as_array_ref_op(self).as_slice()
            }

            #[inline(always)]
            fn as_mut_slice(&mut self) -> &mut [#scalar] {
                self.simd.#as_array_mut_op(self).as_mut_slice()
            }

            #[inline(always)]
            fn from_slice(simd: S, slice: &[#scalar]) -> Self {
                simd.#from_array_ref_op(slice.try_into().unwrap())
            }

            #[inline(always)]
            fn store_slice(&self, slice: &mut [#scalar]) {
                self.simd.#store_array_op(*self, slice.try_into().unwrap());
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
                simd.#from_array_op(core::array::from_fn(f))
            }

        }
        impl<S: Simd> crate::#vec_trait_id<S> for #name<S> {
            #( #methods )*
        }
    }
}
