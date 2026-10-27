// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Experimental exact-width SVE2 backend with bounded memory accesses.
//! Arrays are the ABI; scalable values are locals.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::{
    generic::fallback_method,
    level::Level,
    mk_fallback::Fallback,
    ops::{Op, OpSig},
    types::{ScalarType, VecType},
};

pub(crate) struct Sve2(pub usize);

impl Sve2 {
    fn supports(&self, op: &Op, ty: &VecType) -> bool {
        if ty.scalar == ScalarType::Mask || ty.n_bits() > self.0 {
            return false;
        }
        match op.method {
            "add" | "sub" | "mul" => true,
            "div" | "sqrt" | "mul_add" => ty.scalar == ScalarType::Float,
            "reduce_sum" => ty.scalar != ScalarType::Float,
            "reverse" | "zip_low" | "zip_high" | "unzip_low" | "unzip_high" | "interleave"
            | "deinterleave" => true,
            "swizzle_dyn"
            | "swizzle_dyn_precise"
            | "swizzle_dyn_within_blocks"
            | "concat_swizzle_dyn"
            | "concat_swizzle_dyn_precise" => *ty == ty.bytes_ty(),
            _ => false,
        }
    }
}

impl Level for Sve2 {
    fn name(&self) -> &'static str {
        match self.0 {
            128 => "Sve2_128",
            256 => "Sve2_256",
            512 => "Sve2_512",
            _ => unreachable!(),
        }
    }

    fn native_width(&self) -> usize {
        self.0
    }
    fn max_block_size(&self) -> usize {
        512
    }
    fn enabled_target_features(&self) -> Option<&'static str> {
        Some("sve2")
    }
    fn arch_ty(&self, ty: &VecType) -> TokenStream {
        Fallback.arch_ty(ty)
    }
    fn make_type_impl(&self) -> TokenStream {
        TokenStream::new()
    }

    fn token_doc(&self) -> &'static str {
        "Experimental exact-width SVE2 token with aligned array storage and bounded memory accesses.\n\n\
         Known prototype soundness hole: tokens and vectors implement `Send` and `Sync`,\n\
         but the active SVE vector length is per-thread. Safe detection does not protect\n\
         against transfer to a thread with a different width. See `assume_supported`\n\
         and `Level::detect_sve2` for the execution-context requirements."
    }

    fn make_module_prelude(&self) -> TokenStream {
        let fallback = Fallback.make_module_prelude();
        quote! {
            use core::arch::aarch64::*;
            #fallback
        }
    }

    fn make_impl_body(&self) -> TokenStream {
        let bits = self.0;
        quote! {
            /// Native vector width and required exact active SVE vector length, in bits.
            pub const VECTOR_BITS: usize = #bits;

            /// Construct a token without checking SVE2 availability or active vector length.
            ///
            /// # Safety
            ///
            /// Every use of this token, any copy, or any vector carrying it must occur
            /// on a thread with SVE2 enabled and an active non-streaming SVE vector
            /// length of exactly `Self::VECTOR_BITS`. This includes uses after transfer
            /// through `Send` or `Sync`. The length must remain unchanged during use.
            /// Enabling the `sve2` target feature alone does not establish the width.
            #[inline]
            pub const unsafe fn assume_supported() -> Self {
                Self { _private: () }
            }
        }
    }

    fn should_use_generic_op(&self, op: &Op, ty: &VecType) -> bool {
        // Unimplemented operations use the same 128-bit leaves as Fallback.
        !self.supports(op, ty) && op.sig.should_use_generic_op(ty, 128)
    }

    fn make_method(&self, op: Op, ty: &VecType) -> TokenStream {
        if !self.supports(&op, ty) {
            return fallback_method(op, ty);
        }
        let suffix = format!(
            "{}{}",
            match ty.scalar {
                ScalarType::Float => "f",
                ScalarType::Int => "s",
                ScalarType::Unsigned => "u",
                ScalarType::Mask => unreachable!(),
            },
            ty.scalar_bits
        );
        let load = format_ident!("svld1_{suffix}");
        let store = format_ident!("svst1_{suffix}");
        let ptrue = format_ident!("svptrue_pat_b{}", ty.scalar_bits);
        // Keep memory accesses bounded even at the exact native width: this makes
        // their fixed size visible to LLVM and helps eliminate intermediate arrays.
        let pattern = format_ident!("SV_VL{}", ty.len);
        let is_native = ty.n_bits() == self.0;
        let vec = ty.rust();
        let scalar = ty.scalar.rust(ty.scalar_bits);
        let len = ty.len;
        let wrap = ty.aligned_wrapper();
        let level = self.token();

        self.kernel_method(op, ty, |token| {
            let args = op.sig.simd_trait_arg_names();
            let loads = args.iter().filter(|name| **name != "indices").map(|name| {
                let arg = format_ident!("{name}");
                quote! { let #arg = unsafe { #load(pg, #arg.as_array().as_ptr()) }; }
            });
            let intrinsic = |name: &str| format_ident!("sv{name}_{suffix}");
            let index = format_ident!("svindex_u{}", ty.scalar_bits);
            let index_scalar = ScalarType::Unsigned.rust(ty.scalar_bits);
            let tbl = intrinsic("tbl");
            let zip1 = intrinsic("zip1");
            let half_len = ty.len / 2;
            let half_pattern = format_ident!("SV_VL{half_len}");
            let zip_high = if is_native {
                let zip2 = intrinsic("zip2");
                quote! { #zip2(a, b) }
            } else {
                quote! {
                    {
                        // ZIP2 starts at half the *physical* width. Gather the logical
                        // upper halves into the low lanes before using ZIP1 instead.
                        let indices = #index(#half_len as #index_scalar, 1);
                        #zip1(#tbl(a, indices), #tbl(b, indices))
                    }
                }
            };
            let unzip = |odd: bool| {
                if is_native {
                    let uzp = intrinsic(if odd { "uzp2" } else { "uzp1" });
                    return quote! { #uzp(a, b) };
                }
                let first = usize::from(odd);
                let lane_mask = len - 1;
                let and = format_ident!("svand_n_u{}_x", ty.scalar_bits);
                let select = intrinsic("sel");
                quote! {
                    {
                        // The two logical half-vectors must meet at len/2, not VL/2.
                        let indices = #and(pg, #index(#first as #index_scalar, 2), #lane_mask as #index_scalar);
                        let first_half = #ptrue::<{ svpattern::#half_pattern }>();
                        #select(first_half, #tbl(a, indices), #tbl(b, indices))
                    }
                }
            };
            let result = match op.method {
                "add" | "sub" | "mul" | "div" => {
                    // Predication suffix follows the element type in ACLE names.
                    let f = format_ident!("sv{}_{suffix}_x", op.method);
                    quote! { #f(pg, a, b) }
                }
                "sqrt" => {
                    let f = format_ident!("svsqrt_{suffix}_x");
                    quote! { #f(pg, a) }
                }
                "mul_add" => {
                    let f = format_ident!("svmla_{suffix}_x");
                    quote! { #f(pg, c, a, b) }
                }
                "reduce_sum" => {
                    let f = intrinsic("addv");
                    quote! { #f(pg, a) as #scalar }
                }
                "reverse" if is_native => {
                    let rev = intrinsic("rev");
                    quote! { #rev(a) }
                }
                "reverse" => {
                    let last = len - 1;
                    // An unsigned all-ones step encodes -1 for INDEX.
                    quote! { #tbl(a, #index(#last as #index_scalar, #index_scalar::MAX)) }
                }
                "zip_low" => quote! { #zip1(a, b) },
                "zip_high" => zip_high.clone(),
                "unzip_low" => unzip(false),
                "unzip_high" => unzip(true),
                "interleave" | "deinterleave" => {
                    let (lo, hi) = if op.method == "interleave" {
                        (quote! { #zip1(a, b) }, zip_high.clone())
                    } else {
                        (unzip(false), unzip(true))
                    };
                    quote! { (store(#token, pg, #lo), store(#token, pg, #hi)) }
                }
                // Bounded loads zero every lane beyond the logical table. This
                // also makes precise out-of-range lookups zero on wider hardware.
                "swizzle_dyn" | "swizzle_dyn_precise" => quote! { svtbl_u8(a, indices) },
                "concat_swizzle_dyn" | "concat_swizzle_dyn_precise" if is_native => {
                    quote! { svtbl2_u8(svcreate2_u8(a, b), indices) }
                }
                "concat_swizzle_dyn" | "concat_swizzle_dyn_precise" => {
                    // TBL2 places b at the physical width, not the logical width.
                    // Each lookup is zero outside its logical table; combine them.
                    // For indices < len, the subtraction wraps to high (zero) lanes.
                    quote! { svorr_u8_x(pg, svtbl_u8(a, indices),
                        svtbl_u8(b, svsub_n_u8_x(pg, indices, #len as u8))) }
                }
                "swizzle_dyn_within_blocks" => quote! {
                    svtbl_u8(a, svadd_u8_x(pg,
                        svand_n_u8_x(pg, svindex_u8(0, 1), 0xf0),
                        svand_n_u8_x(pg, indices, 0x0f)))
                },
                _ => unreachable!(),
            };
            let indices = args.contains(&"indices").then(|| {
                quote! {
                    let indices = unsafe { svld1_u8(pg, indices.as_array().as_ptr()) };
                }
            });
            let native = format_ident!(
                "sv{}{}_t",
                match ty.scalar {
                    ScalarType::Float => "float",
                    ScalarType::Int => "int",
                    ScalarType::Unsigned => "uint",
                    ScalarType::Mask => unreachable!(),
                },
                ty.scalar_bits
            );
            let finish = if matches!(
                op.sig,
                OpSig::Reduce { .. } | OpSig::Interleave | OpSig::Deinterleave
            ) {
                result
            } else {
                quote! { store(#token, pg, #result) }
            };
            let store_helper = (!matches!(op.sig, OpSig::Reduce { .. })).then(|| {
                quote! {
                    #[inline]
                    #[target_feature(enable = "sve2")]
                    fn store(token: #level, pg: svbool_t, value: #native) -> #vec<#level> {
                        let mut out = core::mem::MaybeUninit::<[#scalar; #len]>::uninit();
                        // SAFETY: the exact-width token guarantees the predicate activates
                        // all logical lanes, and no lanes beyond this array.
                        unsafe {
                            #store(pg, out.as_mut_ptr().cast::<#scalar>(), value);
                            #vec { val: #wrap(out.assume_init()), simd: token }
                        }
                    }
                }
            });
            quote! {
                #store_helper
                let pg = #ptrue::<{ svpattern::#pattern }>();
                #(#loads)*
                #indices
                #finish
            }
        })
    }
}
