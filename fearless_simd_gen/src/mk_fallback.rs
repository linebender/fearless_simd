// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::fallback;
use crate::generic::{generic_combine, generic_op, generic_split};
use crate::ops::{Op, OpSig, ops_for_type, valid_reinterpret};
use crate::types::{SIMD_TYPES, ScalarType, VecType, type_imports};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

#[derive(Clone, Copy)]
pub(crate) struct Level;

impl Level {
    fn name(self) -> &'static str {
        "Fallback"
    }

    fn token(self) -> TokenStream {
        let ident = Ident::new(self.name(), Span::call_site());
        quote! { #ident }
    }
}

pub(crate) fn mk_fallback_impl() -> TokenStream {
    let imports = type_imports();
    let simd_impl = mk_simd_impl();

    quote! {
        use core::ops::*;
        use crate::{Bytes, seal::Seal, Level, Simd, SimdInto};

        #imports

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

        /// The SIMD token for the "fallback" level.
        #[derive(Clone, Copy, Debug)]
        pub struct Fallback {
            pub fallback: crate::core_arch::fallback::Fallback,
        }

        impl Fallback {
            #[inline]
            pub const fn new() -> Self {
                Fallback {
                    fallback: crate::core_arch::fallback::Fallback::new(),
                }
            }
        }

        impl Seal for Fallback {}

        #simd_impl
    }
}

fn mk_simd_impl() -> TokenStream {
    let level_tok = Level.token();
    let mut methods = vec![];
    for vec_ty in SIMD_TYPES {
        let scalar_bits = vec_ty.scalar_bits;
        let ty_name = vec_ty.rust_name();
        for Op { method, sig, .. } in ops_for_type(vec_ty) {
            let b1 = (vec_ty.n_bits() > 128 && !matches!(method, "split" | "narrow"))
                || vec_ty.n_bits() > 256;
            let b2 = !matches!(method, "load_interleaved_128")
                && !matches!(method, "store_interleaved_128");

            if b1 && b2 {
                methods.push(generic_op(method, sig, vec_ty));
                continue;
            }
            let method_name = format!("{method}_{ty_name}");
            let method_sig = sig.simd_trait_method_sig(vec_ty, &method_name);
            let method_sig = quote! {
                #[inline(always)]
                #method_sig
            };

            let method = match sig {
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
                    let arch_ty = fallback::arch_ty(vec_ty);
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                let args = [quote! { a[#idx] }, quote! { shift as #arch_ty }];
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
                        // TODO: This is has slightly different semantics than a fused multiply-add,
                        // since we are not actually fusing it, should this be documented?
                        quote! {
                            #method_sig {
                               a.mul(b).add(c)
                            }
                        }
                    } else if method == "mul_sub" {
                        // TODO: Same as above
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
                    let mask_type = VecType::new(ScalarType::Mask, vec_ty.scalar_bits, vec_ty.len);
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx: usize| {
                                let args = [quote! { &a[#idx] }, quote! { &b[#idx] }];
                                let expr = fallback::expr(method, vec_ty, &args);
                                let mask_ty = mask_type.scalar.rust(scalar_bits);
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
                OpSig::Combine { combined_ty } => generic_combine(vec_ty, &combined_ty),
                OpSig::Split { half_ty } => generic_split(vec_ty, &half_ty),
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
                } => {
                    let to_ty = &VecType::new(target_ty, scalar_bits, vec_ty.len);
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
                    let items = interleave_indices(
                        len,
                        len / block_count as usize,
                        |idx| quote! { a[#idx] },
                    );

                    quote! {
                        #method_sig {
                            *dest = #items;
                        }
                    }
                }
            };
            methods.push(method);
        }
    }

    // Note: the `vectorize` implementation is pretty boilerplate and should probably
    // be factored out for DRY.
    quote! {
        impl Simd for #level_tok {
            type f32s = f32x4<Self>;
            type f64s = f64x2<Self>;
            type u8s = u8x16<Self>;
            type i8s = i8x16<Self>;
            type u16s = u16x8<Self>;
            type i16s = i16x8<Self>;
            type u32s = u32x4<Self>;
            type i32s = i32x4<Self>;
            type mask8s = mask8x16<Self>;
            type mask16s = mask16x8<Self>;
            type mask32s = mask32x4<Self>;
            type mask64s = mask64x2<Self>;
            #[inline(always)]
            fn level(self) -> Level {
                #[cfg(feature = "force_support_fallback")]
                return Level::#level_tok(self);
                #[cfg(not(feature = "force_support_fallback"))]
                Level::baseline()
            }

            #[inline]
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                f()
            }

            #( #methods )*
        }
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
