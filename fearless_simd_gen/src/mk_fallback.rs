// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::fallback;
use crate::generic::{generic_from_bytes, generic_op_name, generic_to_bytes};
use crate::level::Level;
use crate::ops::{Op, OpSig, RefKind, valid_reinterpret};
use crate::types::{ScalarType, VecType};
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Clone, Copy)]
pub(crate) struct Fallback;

impl Level for Fallback {
    fn name(&self) -> &'static str {
        "Fallback"
    }

    fn native_width(&self) -> usize {
        128
    }

    fn max_block_size(&self) -> usize {
        512
    }

    fn arch_ty(&self, vec_ty: &VecType) -> TokenStream {
        let scalar_rust = vec_ty.scalar.rust(vec_ty.scalar_bits);
        let len = vec_ty.len;
        quote!([#scalar_rust; #len])
    }

    fn token_doc(&self) -> &'static str {
        r#"The SIMD token for the "fallback" level."#
    }

    fn token_inner(&self) -> TokenStream {
        quote!(crate::core_arch::fallback::Fallback)
    }

    fn make_module_prelude(&self) -> TokenStream {
        quote! {
            use core::ops::*;

            #[cfg(all(feature = "libm", not(feature = "std")))]
            trait FloatExt {
                fn floor(self) -> Self;
                fn ceil(self) -> Self;
                fn round_ties_even(self) -> Self;
                fn fract(self) -> Self;
                fn sqrt(self) -> Self;
                fn trunc(self) -> Self;
            }
            #[cfg(all(feature = "libm", not(feature = "std")))]
            impl FloatExt for f32 {
                #[inline(always)]
                fn floor(self) -> f32 {
                    libm::floorf(self)
                }
                #[inline(always)]
                fn ceil(self) -> f32 {
                    libm::ceilf(self)
                }
                #[inline(always)]
                fn round_ties_even(self) -> f32 {
                    libm::rintf(self)
                }
                #[inline(always)]
                fn sqrt(self) -> f32 {
                    libm::sqrtf(self)
                }
                #[inline(always)]
                fn fract(self) -> f32 {
                    self - self.trunc()
                }
                #[inline(always)]
                fn trunc(self) -> f32 {
                    libm::truncf(self)
                }
            }

            #[cfg(all(feature = "libm", not(feature = "std")))]
            impl FloatExt for f64 {
                #[inline(always)]
                fn floor(self) -> f64 {
                    libm::floor(self)
                }
                #[inline(always)]
                fn ceil(self) -> f64 {
                    libm::ceil(self)
                }
                #[inline(always)]
                fn round_ties_even(self) -> f64 {
                    libm::rint(self)
                }
                #[inline(always)]
                fn sqrt(self) -> f64 {
                    libm::sqrt(self)
                }
                #[inline(always)]
                fn fract(self) -> f64 {
                    self - self.trunc()
                }
                #[inline(always)]
                fn trunc(self) -> f64 {
                    libm::trunc(self)
                }
            }
        }
    }

    fn make_vectorize_body(&self) -> TokenStream {
        quote! {
            f()
        }
    }

    fn make_level_body(&self) -> TokenStream {
        let level_tok = Self.token();
        quote! {
            #[cfg(feature = "force_support_fallback")]
            return Level::#level_tok(self);
            #[cfg(not(feature = "force_support_fallback"))]
            Level::baseline()
        }
    }

    fn make_impl_body(&self) -> TokenStream {
        quote! {
            #[inline]
            pub const fn new() -> Self {
                Self {
                    fallback: crate::core_arch::fallback::Fallback::new(),
                }
            }
        }
    }

    fn make_method(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let Op { sig, method, .. } = op;
        let method_sig = op.simd_trait_method_sig(vec_ty);

        match sig {
            OpSig::Splat => {
                let num_elements = vec_ty.len;
                quote! {
                    #method_sig {
                        [val; #num_elements].simd_into(self)
                    }
                }
            }
            OpSig::Unary => {
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            let args = [quote! { a[#idx] }];
                            let expr = fallback::expr(method, vec_ty, &args);
                            quote! { #expr }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #items.simd_into(self)
                    }
                }
            }
            OpSig::WidenNarrow { target_ty } => {
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            let scalar_ty = target_ty.scalar.rust(target_ty.scalar_bits);
                            quote! { a[#idx] as #scalar_ty }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #items.simd_into(self)
                    }
                }
            }
            OpSig::Binary => {
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            let b = if fallback::translate_op(
                                method,
                                vec_ty.scalar == ScalarType::Float,
                            )
                            .map(rhs_reference)
                            .unwrap_or(true)
                            {
                                quote! { &b[#idx] }
                            } else {
                                quote! { b[#idx] }
                            };

                            let args = [quote! { a[#idx] }, quote! { #b }];
                            let expr = fallback::expr(method, vec_ty, &args);
                            quote! { #expr }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #items.simd_into(self)
                    }
                }
            }
            OpSig::Shift => {
                let rust_scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            let args = [quote! { a[#idx] }, quote! { shift as #rust_scalar }];
                            let expr = fallback::expr(method, vec_ty, &args);
                            quote! { #expr }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #items.simd_into(self)
                    }
                }
            }
            OpSig::Ternary => {
                if method == "mul_add" {
                    quote! {
                        #method_sig {
                            a.mul(b).add(c)
                        }
                    }
                } else if method == "mul_sub" {
                    quote! {
                        #method_sig {
                            a.mul(b).sub(c)
                        }
                    }
                } else {
                    let args = [
                        quote! { a.into() },
                        quote! { b.into() },
                        quote! { c.into() },
                    ];

                    let expr = fallback::expr(method, vec_ty, &args);
                    quote! {
                        #method_sig {
                            #expr.simd_into(self)
                        }
                    }
                }
            }
            OpSig::Compare => {
                let mask_type = vec_ty.cast(ScalarType::Mask);
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx: usize| {
                            let args = [quote! { &a[#idx] }, quote! { &b[#idx] }];
                            let expr = fallback::expr(method, vec_ty, &args);
                            let mask_ty = mask_type.scalar.rust(vec_ty.scalar_bits);
                            quote! { -(#expr as #mask_ty) }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #items.simd_into(self)
                    }
                }
            }
            OpSig::Select => {
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            quote! { if a[#idx] != 0 { b[#idx] } else { c[#idx] } }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #items.simd_into(self)
                    }
                }
            }
            OpSig::Combine { combined_ty } => {
                let n = vec_ty.len;
                let n2 = combined_ty.len;
                let default = match vec_ty.scalar {
                    ScalarType::Float => quote! { 0.0 },
                    _ => quote! { 0 },
                };
                quote! {
                    #method_sig {
                        let mut result = [#default; #n2];
                        result[0..#n].copy_from_slice(&a.val.0);
                        result[#n..#n2].copy_from_slice(&b.val.0);
                        result.simd_into(self)
                    }
                }
            }
            OpSig::Split { half_ty } => {
                let n = vec_ty.len;
                let nhalf = half_ty.len;
                let default = match vec_ty.scalar {
                    ScalarType::Float => quote! { 0.0 },
                    _ => quote! { 0 },
                };
                quote! {
                    #method_sig {
                        let mut b0 = [#default; #nhalf];
                        let mut b1 = [#default; #nhalf];
                        b0.copy_from_slice(&a.val.0[0..#nhalf]);
                        b1.copy_from_slice(&a.val.0[#nhalf..#n]);
                        (b0.simd_into(self), b1.simd_into(self))
                    }
                }
            }
            OpSig::Zip { select_low } => {
                let indices = if select_low {
                    0..vec_ty.len / 2
                } else {
                    (vec_ty.len / 2)..vec_ty.len
                };

                let zip = make_list(
                    indices
                        .map(|idx| {
                            quote! {a[#idx], b[#idx] }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #zip.simd_into(self)
                    }
                }
            }
            OpSig::Unzip { select_even } => {
                let indices = if select_even {
                    (0..vec_ty.len).step_by(2)
                } else {
                    (1..vec_ty.len).step_by(2)
                };

                let unzip = make_list(
                    indices
                        .clone()
                        .map(|idx| {
                            quote! {a[#idx]}
                        })
                        .chain(indices.map(|idx| {
                            quote! {b[#idx]}
                        }))
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #unzip.simd_into(self)
                    }
                }
            }
            OpSig::Cvt {
                target_ty,
                scalar_bits,
                precise,
            } => {
                if precise {
                    let non_precise =
                        generic_op_name(method.strip_suffix("_precise").unwrap(), vec_ty);
                    quote! {
                        #method_sig {
                            self.#non_precise(a)
                        }
                    }
                } else {
                    let to_ty = vec_ty.reinterpret(target_ty, scalar_bits);
                    let scalar = to_ty.scalar.rust(scalar_bits);
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                quote! { a[#idx] as #scalar }
                            })
                            .collect::<Vec<_>>(),
                    );
                    quote! {
                        #method_sig {
                            #items.simd_into(self)
                        }
                    }
                }
            }
            OpSig::Reinterpret {
                target_ty,
                scalar_bits,
            } => {
                if valid_reinterpret(vec_ty, target_ty, scalar_bits) {
                    quote! {
                        #method_sig {
                            a.bitcast()
                        }
                    }
                } else {
                    quote! {}
                }
            }
            OpSig::MaskReduce {
                quantifier,
                condition,
            } => {
                let indices = (0..vec_ty.len).map(|idx| quote! { #idx });
                let check = if condition {
                    quote! { != }
                } else {
                    quote! { == }
                };

                let expr = match quantifier {
                    crate::ops::Quantifier::Any => {
                        quote! { #(a[#indices] #check 0)||* }
                    }
                    crate::ops::Quantifier::All => {
                        quote! { #(a[#indices] #check 0)&&* }
                    }
                };

                quote! {
                    #method_sig {
                        #expr
                    }
                }
            }
            OpSig::LoadInterleaved {
                block_size,
                block_count,
            } => {
                let len = (block_size * block_count) as usize / vec_ty.scalar_bits;
                let items =
                    interleave_indices(len, block_count as usize, |idx| quote! { src[#idx] });

                quote! {
                    #method_sig {
                        #items.simd_into(self)
                    }
                }
            }
            OpSig::StoreInterleaved {
                block_size,
                block_count,
            } => {
                let len = (block_size * block_count) as usize / vec_ty.scalar_bits;
                let items =
                    interleave_indices(len, len / block_count as usize, |idx| quote! { a[#idx] });

                quote! {
                    #method_sig {
                        *dest = #items;
                    }
                }
            }
            OpSig::FromArray { kind } => {
                let vec_rust = vec_ty.rust();
                let wrapper = vec_ty.aligned_wrapper();
                let expr = match kind {
                    RefKind::Value => quote! { val },
                    RefKind::Ref | RefKind::Mut => quote! { *val },
                };
                quote! {
                    #method_sig {
                        #vec_rust { val: #wrapper(#expr), simd: self }
                    }
                }
            }
            OpSig::AsArray { kind } => {
                let ref_tok = kind.token();
                quote! {
                    #method_sig {
                        #ref_tok a.val.0
                    }
                }
            }
            OpSig::FromBytes => generic_from_bytes(method_sig, vec_ty),
            OpSig::ToBytes => generic_to_bytes(method_sig, vec_ty),
        }
    }

    fn make_type_impl(&self) -> TokenStream {
        TokenStream::new()
    }
}

fn interleave_indices(
    len: usize,
    stride: usize,
    func: impl FnMut(usize) -> TokenStream,
) -> TokenStream {
    let indices = {
        let indices = (0..len).collect::<Vec<_>>();
        interleave(&indices, stride)
    };

    make_list(indices.into_iter().map(func).collect::<Vec<_>>())
}

/// Whether the second argument of the function needs to be passed by reference.
fn rhs_reference(method: &str) -> bool {
    !matches!(
        method,
        "copysign" | "min" | "max" | "wrapping_sub" | "wrapping_mul" | "wrapping_add"
    )
}

fn make_list(items: Vec<TokenStream>) -> TokenStream {
    quote!([#( #items, )*])
}

fn interleave(input: &[usize], width: usize) -> Vec<usize> {
    let height = input.len() / width;

    let mut output = Vec::with_capacity(input.len());
    for col in 0..width {
        for row in 0..height {
            output.push(input[row * width + col]);
        }
    }
    output
}
