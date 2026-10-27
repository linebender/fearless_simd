// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Experimental exact-VL SVE2 backend. Arrays are the ABI; scalable values are locals.

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
            // These instructions use the physical register's boundaries. Narrower
            // logical vectors must use the fallback, even with bounded loads/stores.
            "reverse" | "zip_low" | "zip_high" | "unzip_low" | "unzip_high" | "interleave"
            | "deinterleave" => ty.n_bits() == self.0,
            "swizzle_dyn"
            | "swizzle_dyn_precise"
            | "swizzle_dyn_within_blocks"
            | "concat_swizzle_dyn"
            | "concat_swizzle_dyn_precise" => ty.n_bits() == self.0 && *ty == ty.bytes_ty(),
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
        "Experimental exact-width SVE2 token with aligned array storage.\n\n\
         Known prototype soundness hole: this token and its vectors implement `Send` and `Sync`,\n\
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
            /// Required active SVE vector length, in bits.
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
        // Exact native width allows an all-true predicate. Unlike a fixed lane
        // count, LLVM knows this leaves no inactive lanes in scalable temporaries.
        let pattern = if ty.n_bits() == self.0 {
            format_ident!("SV_ALL")
        } else {
            format_ident!("SV_VL{}", ty.len)
        };
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
                "reverse" => {
                    let f = intrinsic("rev");
                    quote! { #f(a) }
                }
                "zip_low" | "zip_high" | "unzip_low" | "unzip_high" => {
                    let f = intrinsic(match op.method {
                        "zip_low" => "zip1",
                        "zip_high" => "zip2",
                        "unzip_low" => "uzp1",
                        _ => "uzp2",
                    });
                    quote! { #f(a, b) }
                }
                "interleave" | "deinterleave" => {
                    let (lo, hi) = if op.method == "interleave" {
                        (intrinsic("zip1"), intrinsic("zip2"))
                    } else {
                        (intrinsic("uzp1"), intrinsic("uzp2"))
                    };
                    quote! { (store(#token, pg, #lo(a, b)), store(#token, pg, #hi(a, b))) }
                }
                "swizzle_dyn" | "swizzle_dyn_precise" => quote! { svtbl_u8(a, indices) },
                "concat_swizzle_dyn" | "concat_swizzle_dyn_precise" => {
                    quote! { svtbl2_u8(svcreate2_u8(a, b), indices) }
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
                        // SAFETY: the exact-VL token guarantees the predicate activates
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
