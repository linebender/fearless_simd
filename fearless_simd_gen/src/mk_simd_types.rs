// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    unreachable_pub,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{format_ident, quote};

use crate::{
    ops::{CORE_OPS, OpSig, TyFlavor, ops_for_type},
    types::{SIMD_TYPES, ScalarType, VecType},
};

pub fn mk_simd_types() -> TokenStream {
    let mut result = quote! {
        use crate::{Bytes, Select, Simd, SimdFrom, SimdInto, SimdCvtFloat, SimdCvtTruncate, Scalar};
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
            }
        };
        let impl_block = simd_impl(ty);
        let simd_from_items = make_list(
            (0..ty.len)
                .map(|idx| quote! { val[#idx] })
                .collect::<Vec<_>>(),
        );
        let mut cvt_impls = Vec::new();
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
                    cvt_impls.push(quote! {
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
                cvt_impls.push(quote! {
                    impl<S: Simd> SimdCvtTruncate<#src_ty<S>> for #name<S> {
                        fn truncate_from(x: #src_ty<S>) -> Self {
                            x.simd.#method(x)
                        }
                    }
                });
            }
            _ => {}
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

            #( #cvt_impls )*
        });
    }

    for ty in [ScalarType::Unsigned, ScalarType::Int, ScalarType::Float] {
        for bits in [8, 16, 32] {
            if ty == ScalarType::Float && ![32, 64].contains(&bits) {
                continue;
            }
            result.extend(scalar_impl(ty, bits));
        }
    }
    for bits in [8, 16, 32] {
        let ty = ScalarType::Int;
        let scalar = ty.rust(bits);
        result.extend(quote! {
            impl crate::SimdMask<#scalar, Scalar> for #scalar {
                fn simd_eq(self, rhs: impl SimdInto<Self, Scalar>) -> Self::Mask {
                    -((self == rhs.simd_into(Scalar)) as #scalar)
                }
            }
        });
    }

    result
}

fn scalar_impl(ty: ScalarType, bits: usize) -> TokenStream {
    let scalar = ty.rust(bits);
    let block_ty = VecType::new(ty, bits, 128 / bits).rust();
    let mask = ScalarType::Int.rust(bits);
    let bytes = ScalarType::Unsigned.rust(bits);
    let to_bytes = match ty {
        ScalarType::Float => quote! { self.to_bits() },
        ScalarType::Int => quote! { self as #bytes },
        ScalarType::Unsigned | ScalarType::Mask => quote! { self },
    };
    let from_bytes = match ty {
        ScalarType::Float => quote! { #scalar::from_bits(value) },
        ScalarType::Int => quote! { value as Self },
        ScalarType::Unsigned | ScalarType::Mask => quote! { value },
    };
    let cvt_float = match (ty, bits) {
        (ScalarType::Int | ScalarType::Unsigned, 32) => quote! {
            impl crate::SimdCvtTruncate<f32> for #scalar {
                fn truncate_from(x: f32) -> Self { x as Self }
            }
            impl crate::SimdCvtFloat<#scalar> for f32 {
                fn float_from(x: #scalar) -> Self {
                    x as Self
                }
            }
        },
        _ => quote!(),
    };
    let common = quote! {
        #[inline(always)]
        fn simd_eq(self, rhs: impl SimdInto<Self, Scalar>) -> Self::Mask {
            -((self == rhs.simd_into(Scalar)) as #mask)
        }
        #[inline(always)]
        fn simd_lt(self, rhs: impl SimdInto<Self, Scalar>) -> Self::Mask {
            -((self < rhs.simd_into(Scalar)) as #mask)
        }
        #[inline(always)]
        fn simd_le(self, rhs: impl SimdInto<Self, Scalar>) -> Self::Mask {
            -((self <= rhs.simd_into(Scalar)) as #mask)
        }
        #[inline(always)]
        fn simd_ge(self, rhs: impl SimdInto<Self, Scalar>) -> Self::Mask {
            -((self >= rhs.simd_into(Scalar)) as #mask)
        }
        #[inline(always)]
        fn simd_gt(self, rhs: impl SimdInto<Self, Scalar>) -> Self::Mask {
            -((self > rhs.simd_into(Scalar)) as #mask)
        }
        #[inline(always)]
        fn zip_low(self, _rhs: impl SimdInto<Self, Scalar>) -> Self {
            self
        }
        #[inline(always)]
        fn zip_high(self, _rhs: impl SimdInto<Self, Scalar>) -> Self {
            self
        }
        #[inline(always)]
        fn unzip_low(self, _rhs: impl SimdInto<Self, Scalar>) -> Self {
            self
        }
        #[inline(always)]
        fn unzip_high(self, _rhs: impl SimdInto<Self, Scalar>) -> Self {
            self
        }
    };
    let ty_impl = match ty {
        ScalarType::Int | ScalarType::Unsigned => quote! {
            impl crate::SimdInt<#scalar, Scalar> for #scalar {
                #common

                #[inline(always)]
                fn min(self, rhs: impl SimdInto<Self, Scalar>) -> Self {
                    Ord::min(self, rhs.simd_into(Scalar))
                }
                #[inline(always)]
                fn max(self, rhs: impl SimdInto<Self, Scalar>) -> Self {
                    Ord::max(self, rhs.simd_into(Scalar))
                }
            }
        },
        ScalarType::Float => quote! {
            impl crate::SimdFloat<#scalar, Scalar> for #scalar {
                #common

                #[inline(always)]
                fn abs(self) -> Self {
                    #scalar::abs(self)
                }
                #[inline(always)]
                fn sqrt(self) -> Self {
                    #scalar::sqrt(self)
                }
                #[inline(always)]
                fn copysign(self, rhs: impl SimdInto<Self, Scalar>) -> Self {
                    #scalar::copysign(self, rhs.simd_into(Scalar))
                }

                #[inline(always)]
                fn max(self, rhs: impl SimdInto<Self, Scalar>) -> Self {
                    #scalar::max(self, rhs.simd_into(Scalar))
                }
                #[inline(always)]
                fn max_precise(self, rhs: impl SimdInto<Self, Scalar>) -> Self {
                    #scalar::max(self, rhs.simd_into(Scalar))
                }
                #[inline(always)]
                fn min(self, rhs: impl SimdInto<Self, Scalar>) -> Self {
                    #scalar::min(self, rhs.simd_into(Scalar))
                }
                #[inline(always)]
                fn min_precise(self, rhs: impl SimdInto<Self, Scalar>) -> Self {
                    #scalar::min(self, rhs.simd_into(Scalar))
                }
                #[inline(always)]
                fn madd(
                    self,
                    op1: impl SimdInto<Self, Scalar>,
                    op2: impl SimdInto<Self, Scalar>,
                ) -> Self {
                    self.mul_add(op1.simd_into(Scalar), op2.simd_into(Scalar))
                }
                #[inline(always)]
                fn msub(
                    self,
                    op1: impl SimdInto<Self, Scalar>,
                    op2: impl SimdInto<Self, Scalar>,
                ) -> Self {
                    self.mul_add(op1.simd_into(Scalar), -op2.simd_into(Scalar))
                }
                #[inline(always)]
                fn floor(self) -> Self {
                    #scalar::floor(self)
                }
                #[inline(always)]
                fn fract(self) -> Self {
                    #scalar::fract(self)
                }
                #[inline(always)]
                fn trunc(self) -> Self {
                    #scalar::trunc(self)
                }
            }
        },
        _ => quote!(),
    };
    quote! {
        impl crate::Bytes for #scalar {
            type Bytes = #bytes;
            fn to_bytes(self) -> #bytes { #to_bytes }
            fn from_bytes(value: #bytes) -> Self { #from_bytes }
        }

        impl crate::SimdBase<#scalar, Scalar> for #scalar {
            const N: usize = 1;
            type Mask = #mask;
            type Block = #block_ty<Scalar>;

            #[inline(always)]
            fn witness(&self) -> Scalar {
                Scalar
            }

            #[inline(always)]
            fn as_slice(&self) -> &[#scalar] {
                core::slice::from_ref(self)
            }

            #[inline(always)]
            fn as_mut_slice(&mut self) -> &mut [#scalar] {
                core::slice::from_mut(self)
            }

            #[inline(always)]
            fn from_slice(Scalar: Scalar, slice: &[#scalar]) -> Self {
                slice[0]
            }

            #[inline(always)]
            fn splat(Scalar: Scalar, val: #scalar) -> Self {
                val
            }

            #[inline(always)]
            fn block_splat(block: Self::Block) -> Self {
                block.as_slice()[0]
            }
        }

        #ty_impl

        impl crate::Select<#scalar> for #mask {
            fn select(self, if_true: #scalar, if_false: #scalar) -> #scalar {
                if self != 0 { if_true } else { if_false }
            }
        }

        #cvt_float
    }
}

/// Create the impl block for the type
///
/// This may go away, as possibly all methods will be subsumed by the `vec_impl`.
fn simd_impl(ty: &VecType) -> TokenStream {
    let name = ty.rust();
    let ty_name = ty.rust_name();
    let mut methods = vec![];
    for (method, sig) in ops_for_type(ty, true) {
        let method_name = Ident::new(method, Span::call_site());
        let trait_method = Ident::new(&format!("{method}_{ty_name}"), Span::call_site());
        if matches!(
            sig,
            OpSig::Unary
                | OpSig::Binary
                | OpSig::Compare
                | OpSig::Combine
                | OpSig::Cvt(_, _)
                | OpSig::Reinterpret(_, _)
                | OpSig::Shift
        ) {
            if let Some(args) = sig.vec_trait_args() {
                let ret_ty = sig.ret_ty(ty, TyFlavor::VecImpl);
                let call_args = match sig {
                    OpSig::Unary | OpSig::Cvt(_, _) | OpSig::Reinterpret(_, _) => quote! { self },
                    OpSig::Binary | OpSig::Compare | OpSig::Combine => {
                        quote! { self, rhs.simd_into(self.simd) }
                    }
                    OpSig::Shift => {
                        quote! { self, shift }
                    }
                    OpSig::Ternary => {
                        quote! { self, op1.simd_into(self.simd), op2.simd_into(self.simd) }
                    }
                    _ => quote! { todo!() },
                };
                methods.push(quote! {
                    #[inline(always)]
                    pub fn #method_name(#args) -> #ret_ty {
                        self.simd.#trait_method(#call_args)
                    }
                });
            }
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
    let ty_name = ty.rust_name();
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
    let splat = Ident::new(&format!("splat_{}", ty.rust_name()), Span::call_site());
    let mut methods = vec![];
    for (method, sig) in ops_for_type(ty, false) {
        if CORE_OPS.contains(&method) || matches!(sig, OpSig::Combine) {
            continue;
        }
        let method_name = Ident::new(method, Span::call_site());
        let trait_method = Ident::new(&format!("{method}_{ty_name}"), Span::call_site());
        if let Some(args) = sig.vec_trait_args() {
            let ret_ty = sig.ret_ty(ty, TyFlavor::VecImpl);
            let call_args = match sig {
                OpSig::Unary => quote! { self },
                OpSig::Binary
                | OpSig::Compare
                | OpSig::Combine
                | OpSig::Zip(_)
                | OpSig::Unzip(_) => {
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
        64 => quote! {
            block.split().0
        },
        128 => quote! {
            block
        },
        256 => quote! {
            block.combine(block)
        },
        512 => quote! {
            let block2 = block.combine(block);
            block2.combine(block2)
        },
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

        }
        impl<S: Simd> crate::#vec_trait_id<#scalar, S> for #name<S> {
            #( #methods )*
        }
    }
}

fn make_list(items: Vec<TokenStream>) -> TokenStream {
    quote!([#( #items, )*])
}
