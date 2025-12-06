// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

use crate::{
    ops::OpSig,
    types::{ScalarType, VecType},
};

/// Implementation of combine based on `copy_from_slice`
pub(crate) fn generic_combine(ty: &VecType, combined_ty: &VecType) -> TokenStream {
    let n = ty.len;
    let n2 = combined_ty.len;
    let ty_rust = ty.rust();
    let result = combined_ty.rust();
    let name = Ident::new(&format!("combine_{}", ty.rust_name()), Span::call_site());
    let default = match ty.scalar {
        ScalarType::Float => quote! { 0.0 },
        _ => quote! { 0 },
    };
    quote! {
        #[inline(always)]
        fn #name(self, a: #ty_rust<Self>, b: #ty_rust<Self>) -> #result<Self> {
            let mut result = [#default; #n2];
            result[0..#n].copy_from_slice(&a.val);
            result[#n..#n2].copy_from_slice(&b.val);
            result.simd_into(self)
        }
    }
}

/// Implementation of split based on `copy_from_slice`
pub(crate) fn generic_split(ty: &VecType, half_ty: &VecType) -> TokenStream {
    let n = ty.len;
    let nhalf = half_ty.len;
    let ty_rust = ty.rust();
    let result = half_ty.rust();
    let name = Ident::new(&format!("split_{}", ty.rust_name()), Span::call_site());
    let default = match ty.scalar {
        ScalarType::Float => quote! { 0.0 },
        _ => quote! { 0 },
    };
    quote! {
        #[inline(always)]
        fn #name(self, a: #ty_rust<Self>) -> (#result<Self>, #result<Self>) {
            let mut b0 = [#default; #nhalf];
            let mut b1 = [#default; #nhalf];
            b0.copy_from_slice(&a.val[0..#nhalf]);
            b1.copy_from_slice(&a.val[#nhalf..#n]);
            (b0.simd_into(self), b1.simd_into(self))
        }
    }
}

pub(crate) fn generic_op_name(op: &str, ty: &VecType) -> Ident {
    Ident::new(&format!("{op}_{}", ty.rust_name()), Span::call_site())
}

/// Implementation based on split/combine
///
/// Only suitable for lane-wise and block-wise operations
pub(crate) fn generic_op(op: &str, sig: OpSig, ty: &VecType) -> TokenStream {
    let name = format!("{op}_{}", ty.rust_name());
    let split = Ident::new(&format!("split_{}", ty.rust_name()), Span::call_site());
    let half = VecType::new(ty.scalar, ty.scalar_bits, ty.len / 2);
    let combine = Ident::new(&format!("combine_{}", half.rust_name()), Span::call_site());
    let do_half = Ident::new(&format!("{op}_{}", half.rust_name()), Span::call_site());
    let method_sig = sig.simd_trait_method_sig(ty, &name);
    let method_sig = quote! {
        #[inline(always)]
        #method_sig
    };
    match sig {
        OpSig::Splat => {
            quote! {
                #method_sig {
                    let half = self.#do_half(val);
                    self.#combine(half, half)
                }
            }
        }
        OpSig::Unary => {
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::Binary => {
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    self.#combine(self.#do_half(a0, b0), self.#do_half(a1, b1))
                }
            }
        }
        OpSig::Shift => {
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0, shift), self.#do_half(a1, shift))
                }
            }
        }
        OpSig::Ternary => {
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    let (c0, c1) = self.#split(c);
                    self.#combine(self.#do_half(a0, b0, c0), self.#do_half(a1, b1, c1))
                }
            }
        }
        OpSig::Compare => {
            let half_mask = VecType::new(ScalarType::Mask, ty.scalar_bits, ty.len / 2);
            let combine_mask = Ident::new(
                &format!("combine_{}", half_mask.rust_name()),
                Span::call_site(),
            );
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    self.#combine_mask(self.#do_half(a0, b0), self.#do_half(a1, b1))
                }
            }
        }
        OpSig::Select => {
            let mask_ty = VecType::new(ScalarType::Mask, ty.scalar_bits, ty.len);
            let split_mask =
                Ident::new(&format!("split_{}", mask_ty.rust_name()), Span::call_site());
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split_mask(a);
                    let (b0, b1) = self.#split(b);
                    let (c0, c1) = self.#split(c);
                    self.#combine(self.#do_half(a0, b0, c0), self.#do_half(a1, b1, c1))
                }
            }
        }
        OpSig::Zip { select_low } => {
            let (e1, e2, e3) = if select_low {
                (
                    quote! {
                        (a0, _)
                    },
                    quote! {
                        (b0, _)
                    },
                    quote! {
                        a0, b0
                    },
                )
            } else {
                (
                    quote! {
                        (_, a1)
                    },
                    quote! {
                        (_, b1)
                    },
                    quote! {
                        a1, b1
                    },
                )
            };

            let zip_low_half =
                Ident::new(&format!("zip_low_{}", half.rust_name()), Span::call_site());
            let zip_high_half =
                Ident::new(&format!("zip_high_{}", half.rust_name()), Span::call_site());

            quote! {
                #method_sig {
                    let #e1 = self.#split(a);
                    let #e2 = self.#split(b);
                    self.#combine(self.#zip_low_half(#e3), self.#zip_high_half(#e3))
                }
            }
        }
        OpSig::Unzip { .. } => {
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    self.#combine(self.#do_half(a0, a1), self.#do_half(b0, b1))
                }
            }
        }
        OpSig::Cvt {
            target_ty,
            scalar_bits,
        } => {
            let half = VecType::new(target_ty, scalar_bits, ty.len / 2);
            let combine = Ident::new(&format!("combine_{}", half.rust_name()), Span::call_site());
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::Reinterpret {
            target_ty,
            scalar_bits,
        } => {
            let mut half = ty.reinterpret(target_ty, scalar_bits);
            half.len /= 2;
            let combine = Ident::new(&format!("combine_{}", half.rust_name()), Span::call_site());
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::WidenNarrow { mut target_ty } => {
            target_ty.len /= 2;
            let combine = Ident::new(
                &format!("combine_{}", target_ty.rust_name()),
                Span::call_site(),
            );
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::Split { half_ty } => generic_split(ty, &half_ty),
        OpSig::Combine { combined_ty } => generic_combine(ty, &combined_ty),
        OpSig::MaskReduce { quantifier, .. } => {
            let combine_op = quantifier.bool_op();
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#do_half(a0) #combine_op self.#do_half(a1)
                }
            }
        }
        OpSig::LoadInterleaved {
            block_size,
            block_count,
        } => {
            let split_len = (block_size * block_count) as usize / (ty.scalar_bits * 2);
            let delegate = format_ident!(
                "{op}_{}",
                VecType {
                    len: ty.len / 2,
                    ..*ty
                }
                .rust_name()
            );
            quote! {
                #method_sig {
                    let (chunks, _) = src.as_chunks::<#split_len>();
                    unsafe {
                        core::mem::transmute([self.#delegate(&chunks[0]), self.#delegate(&chunks[1])])
                    }
                }
            }
        }
        OpSig::StoreInterleaved { .. } => {
            quote! {
                #method_sig {
                    todo!()
                }
            }
        }
    }
}

pub(crate) fn scalar_binary(name: &Ident, f: TokenStream, ty: &VecType) -> TokenStream {
    let ty_rust = ty.rust();
    quote! {
        #[inline(always)]
        fn #name(self, a: #ty_rust<Self>, b: #ty_rust<Self>) -> #ty_rust<Self> {
            core::array::from_fn(|i| #f(a.val[i], b.val[i])).simd_into(self)
        }
    }
}
