// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{ToTokens as _, format_ident, quote};

use crate::arch::neon::load_intrinsic;
use crate::generic::{
    generic_as_array, generic_from_array, generic_from_bytes, generic_op_name, generic_to_bytes,
};
use crate::level::Level;
use crate::ops::{Op, valid_reinterpret};
use crate::{
    arch::neon::{self, arch_ty, cvt_intrinsic, simple_intrinsic, split_intrinsic},
    ops::OpSig,
    types::{SIMD_TYPES, ScalarType, VecType, type_imports},
};

#[derive(Clone, Copy)]
pub(crate) struct Neon;

impl Level for Neon {
    fn name(&self) -> &'static str {
        "Neon"
    }

    fn native_width(&self) -> usize {
        128
    }

    fn make_vectorize_body(&self) -> TokenStream {
        quote! {
        #[target_feature(enable = "neon")]
        #[inline]
        // unsafe not needed here with tf11, but can be justified
        unsafe fn vectorize_neon<F: FnOnce() -> R, R>(f: F) -> R {
            f()
        }
        unsafe { vectorize_neon(f) }
        }
    }

    fn make_method(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let Op { sig, method, .. } = op;
        let method_sig = op.simd_trait_method_sig(vec_ty);
        let method_sig = quote! {
            #[inline(always)]
            #method_sig
        };

        match sig {
            OpSig::Splat => {
                let expr = neon::expr(method, vec_ty, &[quote! { val }]);
                quote! {
                    #method_sig {
                        unsafe {
                            #expr.simd_into(self)
                        }
                    }
                }
            }
            OpSig::Shift => {
                let dup_type = vec_ty.cast(ScalarType::Int);
                let scalar = dup_type.scalar.rust(dup_type.scalar_bits);
                let dup_intrinsic = split_intrinsic("vdup", "n", &dup_type);
                let shift = if method == "shr" {
                    quote! { -(shift as #scalar) }
                } else {
                    quote! { shift as #scalar }
                };
                let expr = neon::expr(
                    method,
                    vec_ty,
                    &[quote! { a.into() }, quote! { #dup_intrinsic ( #shift ) }],
                );
                quote! {
                    #method_sig {
                        unsafe {
                            #expr.simd_into(self)
                        }
                    }
                }
            }
            OpSig::Unary => {
                let args = [quote! { a.into() }];

                let expr = neon::expr(method, vec_ty, &args);

                quote! {
                    #method_sig {
                        unsafe {
                            #expr.simd_into(self)
                        }
                    }
                }
            }
            OpSig::LoadInterleaved {
                block_size,
                block_count,
            } => {
                assert_eq!(block_count, 4, "only count of 4 is currently supported");
                let intrinsic = {
                    // The function expects 64-bit or 128-bit
                    let ty = VecType::new(
                        vec_ty.scalar,
                        vec_ty.scalar_bits,
                        block_size as usize / vec_ty.scalar_bits,
                    );
                    simple_intrinsic("vld4", &ty)
                };

                quote! {
                    #method_sig {
                        unsafe {
                            #intrinsic(src.as_ptr()).simd_into(self)
                        }
                    }
                }
            }
            OpSig::StoreInterleaved {
                block_size,
                block_count,
            } => {
                assert_eq!(block_count, 4, "only count of 4 is currently supported");
                let intrinsic = {
                    // The function expects 64-bit or 128-bit
                    let ty = VecType::new(
                        vec_ty.scalar,
                        vec_ty.scalar_bits,
                        block_size as usize / vec_ty.scalar_bits,
                    );
                    simple_intrinsic("vst4", &ty)
                };

                quote! {
                    #method_sig {
                        unsafe {
                            #intrinsic(dest.as_mut_ptr(), a.into())
                        }
                    }
                }
            }
            OpSig::WidenNarrow { target_ty } => {
                let vec_scalar_ty = vec_ty.scalar.rust(vec_ty.scalar_bits);
                let target_scalar_ty = target_ty.scalar.rust(target_ty.scalar_bits);

                if method == "narrow" {
                    let arch = neon::arch_ty(vec_ty);

                    let id1 = Ident::new(&format!("vmovn_{}", vec_scalar_ty), Span::call_site());
                    let id2 =
                        Ident::new(&format!("vcombine_{}", target_scalar_ty), Span::call_site());

                    quote! {
                        #method_sig {
                            unsafe {
                                let converted: #arch = a.into();
                                let low = #id1(converted.0);
                                let high = #id1(converted.1);

                                #id2(low, high).simd_into(self)
                            }
                        }
                    }
                } else {
                    let arch = neon::arch_ty(&target_ty);
                    let id1 = Ident::new(&format!("vmovl_{}", vec_scalar_ty), Span::call_site());
                    let id2 = Ident::new(&format!("vget_low_{}", vec_scalar_ty), Span::call_site());
                    let id3 =
                        Ident::new(&format!("vget_high_{}", vec_scalar_ty), Span::call_site());

                    quote! {
                        #method_sig {
                            unsafe {
                                let low = #id1(#id2(a.into()));
                                let high = #id1(#id3(a.into()));

                                #arch(low, high).simd_into(self)
                            }
                        }
                    }
                }
            }
            OpSig::Binary => {
                let expr = match method {
                    "shlv" | "shrv" => {
                        let mut args = if vec_ty.scalar == ScalarType::Int {
                            // Signed case
                            [quote! { a.into() }, quote! { b.into() }]
                        } else {
                            // Unsigned case
                            let bits = vec_ty.scalar_bits;
                            let reinterpret = format_ident!("vreinterpretq_s{bits}_u{bits}");
                            [quote! { a.into() }, quote! { #reinterpret(b.into()) }]
                        };

                        // For a right shift, we need to negate the shift amount
                        if method == "shrv" {
                            let neg = simple_intrinsic("vneg", &vec_ty.cast(ScalarType::Int));
                            let arg1 = &args[1];
                            args[1] = quote! { #neg(#arg1) };
                        }

                        let expr = neon::expr(method, vec_ty, &args);
                        quote! {
                            #expr.simd_into(self)
                        }
                    }
                    "copysign" => {
                        let shift_amt = Literal::usize_unsuffixed(vec_ty.scalar_bits - 1);
                        let unsigned_ty = vec_ty.cast(ScalarType::Unsigned);
                        let sign_mask =
                            neon::expr("splat", &unsigned_ty, &[quote! { 1 << #shift_amt }]);
                        let vbsl = simple_intrinsic("vbsl", vec_ty);

                        quote! {
                            let sign_mask = #sign_mask;
                            #vbsl(sign_mask, b.into(), a.into()).simd_into(self)
                        }
                    }
                    _ => {
                        let args = [quote! { a.into() }, quote! { b.into() }];
                        let expr = neon::expr(method, vec_ty, &args);
                        quote! {
                            #expr.simd_into(self)
                        }
                    }
                };

                quote! {
                    #method_sig {
                        unsafe {
                            #expr
                        }
                    }
                }
            }
            OpSig::Ternary => {
                let args = match method {
                    "mul_add" | "mul_sub" => [
                        quote! { c.into() },
                        quote! { b.into() },
                        quote! { a.into() },
                    ],
                    _ => [
                        quote! { a.into() },
                        quote! { b.into() },
                        quote! { c.into() },
                    ],
                };

                let mut expr = neon::expr(method, vec_ty, &args);
                if method == "mul_sub" {
                    // -(c - a * b) = (a * b - c)
                    let neg = simple_intrinsic("vneg", vec_ty);
                    expr = quote! { #neg(#expr) };
                }
                quote! {
                    #method_sig {
                        unsafe {
                            #expr.simd_into(self)
                        }
                    }
                }
            }
            OpSig::Compare => {
                let args = [quote! { a.into() }, quote! { b.into() }];
                let expr = neon::expr(method, vec_ty, &args);
                let opt_q = crate::arch::neon::opt_q(vec_ty);
                let scalar_bits = vec_ty.scalar_bits;
                let reinterpret_str = format!("vreinterpret{opt_q}_s{scalar_bits}_u{scalar_bits}");
                let reinterpret = Ident::new(&reinterpret_str, Span::call_site());
                quote! {
                    #method_sig {
                        unsafe {
                            #reinterpret(#expr).simd_into(self)
                        }
                    }
                }
            }
            OpSig::Select => {
                let opt_q = crate::arch::neon::opt_q(vec_ty);
                let scalar_bits = vec_ty.scalar_bits;
                let reinterpret_str = format!("vreinterpret{opt_q}_u{scalar_bits}_s{scalar_bits}");
                let reinterpret = Ident::new(&reinterpret_str, Span::call_site());
                let vbsl = simple_intrinsic("vbsl", vec_ty);
                quote! {
                    #method_sig {
                        unsafe {
                            #vbsl(#reinterpret(a.into()), b.into(), c.into()).simd_into(self)
                        }
                    }
                }
            }
            OpSig::Combine { combined_ty } => {
                let combined_wrapper = combined_ty.aligned_wrapper();
                let combined_arch_ty = arch_ty(&combined_ty);
                let combined_rust = combined_ty.rust();
                let expr = match combined_ty.n_bits() {
                    512 => quote! {
                        #combined_rust {val: #combined_wrapper(#combined_arch_ty(a.val.0.0, a.val.0.1, b.val.0.0, b.val.0.1)), simd: self }
                    },
                    256 => quote! {
                        #combined_rust {val: #combined_wrapper(#combined_arch_ty(a.val.0, b.val.0)), simd: self }
                    },
                    _ => unimplemented!(),
                };
                quote! {
                    #method_sig {
                        #expr
                    }
                }
            }
            OpSig::Split { half_ty } => {
                let split_wrapper = half_ty.aligned_wrapper();
                let split_arch_ty = arch_ty(&half_ty);
                let half_rust = half_ty.rust();
                let expr = match half_ty.n_bits() {
                    256 => quote! {
                        (
                            #half_rust { val: #split_wrapper(#split_arch_ty(a.val.0.0, a.val.0.1)), simd: self },
                            #half_rust { val: #split_wrapper(#split_arch_ty(a.val.0.2, a.val.0.3)), simd: self },
                        )
                    },
                    128 => quote! {
                        (
                            #half_rust { val: #split_wrapper(a.val.0.0), simd: self },
                            #half_rust { val: #split_wrapper(a.val.0.1), simd: self },
                        )
                    },
                    _ => unimplemented!(),
                };
                quote! {
                    #method_sig {
                        #expr
                    }
                }
            }
            OpSig::Zip { select_low } => {
                let neon = if select_low { "vzip1" } else { "vzip2" };
                let zip = simple_intrinsic(neon, vec_ty);
                quote! {
                    #method_sig {
                        let x = a.into();
                        let y = b.into();
                        unsafe {
                            #zip(x, y).simd_into(self)
                        }
                    }
                }
            }
            OpSig::Unzip { select_even } => {
                let neon = if select_even { "vuzp1" } else { "vuzp2" };
                let zip = simple_intrinsic(neon, vec_ty);
                quote! {
                    #method_sig {
                        let x = a.into();
                        let y = b.into();
                        unsafe {
                            #zip(x, y).simd_into(self)
                        }
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
                    let to_ty = &vec_ty.reinterpret(target_ty, scalar_bits);
                    let neon = cvt_intrinsic("vcvt", to_ty, vec_ty);
                    quote! {
                        #method_sig {
                            unsafe {
                                #neon(a.into()).simd_into(self)
                            }
                        }
                    }
                }
            }
            OpSig::Reinterpret {
                target_ty,
                scalar_bits,
            } => {
                if valid_reinterpret(vec_ty, target_ty, scalar_bits) {
                    let to_ty = vec_ty.reinterpret(target_ty, scalar_bits);
                    let neon = cvt_intrinsic("vreinterpret", &to_ty, vec_ty);

                    quote! {
                        #method_sig {
                            unsafe {
                                #neon(a.into()).simd_into(self)
                            }
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
                let (reduction, target) = match (quantifier, condition) {
                    (crate::ops::Quantifier::Any, true) => ("vmaxv", quote! { != 0 }),
                    (crate::ops::Quantifier::Any, false) => ("vminv", quote! { != 0xffffffff }),
                    (crate::ops::Quantifier::All, true) => ("vminv", quote! { == 0xffffffff }),
                    (crate::ops::Quantifier::All, false) => ("vmaxv", quote! { == 0 }),
                };

                let u32_ty = vec_ty.reinterpret(ScalarType::Unsigned, 32);
                let min_max = simple_intrinsic(reduction, &u32_ty);
                let reinterpret = format_ident!("vreinterpretq_u32_s{}", vec_ty.scalar_bits);
                quote! {
                    #method_sig {
                        unsafe {
                            #min_max(#reinterpret(a.into())) #target
                        }
                    }
                }
            }
            OpSig::FromArray { kind } => {
                generic_from_array(method_sig, vec_ty, kind, 512, load_intrinsic)
            }
            OpSig::AsArray { kind } => generic_as_array(method_sig, vec_ty, kind, 512, arch_ty),
            OpSig::FromBytes => generic_from_bytes(method_sig, vec_ty),
            OpSig::ToBytes => generic_to_bytes(method_sig, vec_ty),
        }
    }
}

pub(crate) fn mk_neon_impl(level: &dyn Level) -> TokenStream {
    let imports = type_imports();
    let arch_types_impl = level.impl_arch_types(512, &|vec_ty| arch_ty(vec_ty).into_token_stream());
    let simd_impl = level.mk_simd_impl();
    let ty_impl = mk_type_impl();

    quote! {
        use core::arch::aarch64::*;

        use crate::{seal::Seal, arch_types::ArchTypes, Level, Simd, SimdFrom, SimdInto};

        #imports

        /// The SIMD token for the "neon" level.
        #[derive(Clone, Copy, Debug)]
        pub struct Neon {
            pub neon: crate::core_arch::aarch64::Neon,
            }

        impl Neon {
            #[inline]
            pub const unsafe fn new_unchecked() -> Self {
                Neon {
                    neon: unsafe { crate::core_arch::aarch64::Neon::new_unchecked() },
                }
                }
            }

        impl Seal for Neon {}

        #simd_impl

        #arch_types_impl

        #ty_impl
    }
}

fn mk_type_impl() -> TokenStream {
    let mut result = vec![];
    for ty in SIMD_TYPES {
        let n_bits = ty.n_bits();
        if !(n_bits == 64 || n_bits == 128 || n_bits == 256 || n_bits == 512) {
            continue;
        }
        let simd = ty.rust();
        let arch = neon::arch_ty(ty);
        result.push(quote! {
            impl<S: Simd> SimdFrom<#arch, S> for #simd<S> {
                #[inline(always)]
                fn simd_from(arch: #arch, simd: S) -> Self {
                    Self {
                        val: unsafe { core::mem::transmute_copy(&arch) },
                        simd
                    }
                }
            }
            impl<S: Simd> From<#simd<S>> for #arch {
                #[inline(always)]
                fn from(value: #simd<S>) -> Self {
                    unsafe { core::mem::transmute_copy(&value.val) }
                }
            }
        });
    }
    quote! {
        #( #result )*
    }
}
