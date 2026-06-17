// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{ToTokens as _, format_ident, quote};

use crate::generic::{
    generic_as_array, generic_from_array, generic_from_bytes, generic_mask_set, generic_op_name,
    generic_store_array, generic_to_bytes, integer_lane_mask_splat_arg,
};
use crate::level::Level;
use crate::ops::{Op, SlideGranularity, valid_reinterpret};
use crate::{
    arch::neon::{self, cvt_intrinsic, simple_intrinsic, split_intrinsic},
    ops::OpSig,
    types::{ScalarType, VecType},
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

    fn max_block_size(&self) -> usize {
        512
    }

    fn enabled_target_features(&self) -> Option<&'static str> {
        Some("neon")
    }

    fn arch_ty(&self, vec_ty: &VecType) -> TokenStream {
        let scalar = match vec_ty.scalar {
            ScalarType::Float => "float",
            ScalarType::Unsigned => "uint",
            ScalarType::Int | ScalarType::Mask => "int",
        };
        let name = if vec_ty.n_bits() == 256 {
            format!("{}{}x{}x2_t", scalar, vec_ty.scalar_bits, vec_ty.len / 2)
        } else if vec_ty.n_bits() == 512 {
            format!("{}{}x{}x4_t", scalar, vec_ty.scalar_bits, vec_ty.len / 4)
        } else {
            format!("{}{}x{}_t", scalar, vec_ty.scalar_bits, vec_ty.len)
        };
        Ident::new(&name, Span::call_site()).into_token_stream()
    }

    fn token_doc(&self) -> &'static str {
        r#"A token for Neon intrinsics on aarch64, representing the "neon" level."#
    }

    fn make_module_prelude(&self) -> TokenStream {
        quote! {
            use core::arch::aarch64::*;
        }
    }

    fn make_module_footer(&self) -> TokenStream {
        mk_slide_helpers()
    }

    fn make_impl_body(&self) -> TokenStream {
        quote! {
            #[inline]
            pub const unsafe fn new_unchecked() -> Self {
                Neon { _private: () }
            }
        }
    }

    fn make_method(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let Op { sig, method, .. } = op;
        let method_sig = op.simd_trait_method_sig(vec_ty);

        match sig {
            OpSig::Splat => {
                let expr = neon::expr(method, vec_ty, &[quote! { val }]);
                let normalize_mask = integer_lane_mask_splat_arg(vec_ty);
                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        #normalize_mask
                        #expr.simd_into(#token)
                    }
                })
            }
            OpSig::Shift => {
                let dup_type = vec_ty.cast(ScalarType::Int);
                let scalar = dup_type.scalar.rust(dup_type.scalar_bits);
                let dup_intrinsic = split_intrinsic("vdup", "n", &dup_type);
                // The shift argument is `u32`. If the target is `i32`, use `cast_signed()`, else
                // `as`-casting.
                let shift = match (vec_ty.scalar_bits, method) {
                    (32, "shr") => quote! { -shift.cast_signed() },
                    (32, _) => quote! { shift.cast_signed() },
                    (_, "shr") => quote! { -(shift as #scalar) },
                    (_, _) => quote! { shift as #scalar },
                };
                let expr = neon::expr(
                    method,
                    vec_ty,
                    &[quote! { a.into() }, quote! { #dup_intrinsic ( #shift ) }],
                );
                self.kernel_method(op, vec_ty, |token| {
                    quote! { #expr.simd_into(#token) }
                })
            }
            OpSig::Unary => {
                let args = [quote! { a.into() }];

                let expr = neon::expr(method, vec_ty, &args);

                self.kernel_method(op, vec_ty, |token| {
                    quote! { #expr.simd_into(#token) }
                })
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
                    let arch = self.arch_ty(vec_ty);

                    let id1 = Ident::new(&format!("vmovn_{}", vec_scalar_ty), Span::call_site());
                    let id2 =
                        Ident::new(&format!("vcombine_{}", target_scalar_ty), Span::call_site());

                    self.kernel_method(op, vec_ty, |token| {
                        quote! {
                            let converted: #arch = a.into();
                            let low = #id1(converted.0);
                            let high = #id1(converted.1);

                            #id2(low, high).simd_into(#token)
                        }
                    })
                } else {
                    let arch = self.arch_ty(&target_ty);
                    let id1 = Ident::new(&format!("vmovl_{}", vec_scalar_ty), Span::call_site());
                    let id2 = Ident::new(&format!("vget_low_{}", vec_scalar_ty), Span::call_site());
                    let id3 =
                        Ident::new(&format!("vget_high_{}", vec_scalar_ty), Span::call_site());

                    self.kernel_method(op, vec_ty, |token| {
                        quote! {
                            let low = #id1(#id2(a.into()));
                            let high = #id1(#id3(a.into()));

                            #arch(low, high).simd_into(#token)
                        }
                    })
                }
            }
            OpSig::Binary => self.kernel_method(op, vec_ty, |token| match method {
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
                        #expr.simd_into(#token)
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
                        #vbsl(sign_mask, b.into(), a.into()).simd_into(#token)
                    }
                }
                _ => {
                    let args = [quote! { a.into() }, quote! { b.into() }];
                    let expr = neon::expr(method, vec_ty, &args);
                    quote! {
                        #expr.simd_into(#token)
                    }
                }
            }),
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
                self.kernel_method(op, vec_ty, |token| {
                    quote! { #expr.simd_into(#token) }
                })
            }
            OpSig::Compare => {
                let args = [quote! { a.into() }, quote! { b.into() }];
                let expr = neon::expr(method, vec_ty, &args);
                let opt_q = neon::opt_q(vec_ty);
                let scalar_bits = vec_ty.scalar_bits;
                let reinterpret_str = format!("vreinterpret{opt_q}_s{scalar_bits}_u{scalar_bits}");
                let reinterpret = Ident::new(&reinterpret_str, Span::call_site());
                self.kernel_method(
                    op,
                    vec_ty,
                    |token| quote! { #reinterpret(#expr).simd_into(#token) },
                )
            }
            OpSig::Select => {
                let opt_q = neon::opt_q(vec_ty);
                let scalar_bits = vec_ty.scalar_bits;
                let reinterpret_str = format!("vreinterpret{opt_q}_u{scalar_bits}_s{scalar_bits}");
                let reinterpret = Ident::new(&reinterpret_str, Span::call_site());
                let vbsl = simple_intrinsic("vbsl", vec_ty);
                self.kernel_method(op, vec_ty, |token| {
                    quote! { #vbsl(#reinterpret(a.into()), b.into(), c.into()).simd_into(#token) }
                })
            }
            OpSig::Combine { combined_ty } => {
                let combined_wrapper = combined_ty.aligned_wrapper();
                let combined_arch_ty = self.arch_ty(&combined_ty);
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
                let split_arch_ty = self.arch_ty(&half_ty);
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
                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        let x = a.into();
                        let y = b.into();
                        #zip(x, y).simd_into(#token)
                    }
                })
            }
            OpSig::Unzip { select_even } => {
                let neon = if select_even { "vuzp1" } else { "vuzp2" };
                let zip = simple_intrinsic(neon, vec_ty);
                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        let x = a.into();
                        let y = b.into();
                        #zip(x, y).simd_into(#token)
                    }
                })
            }
            OpSig::Slide { granularity } => {
                use SlideGranularity::*;

                let block_wrapper = vec_ty.aligned_wrapper();
                let bytes_ty = vec_ty.reinterpret(ScalarType::Unsigned, 8);
                let combined_bytes = bytes_ty.rust();
                let scalar_bytes = vec_ty.scalar_bits / 8;
                let num_items = vec_ty.len;
                let to_bytes = generic_op_name("cvt_to_bytes", vec_ty);
                let from_bytes = generic_op_name("cvt_from_bytes", vec_ty);

                let byte_shift = if scalar_bytes == 1 {
                    quote! { SHIFT }
                } else {
                    quote! { SHIFT * #scalar_bytes }
                };

                let bytes_expr = match (granularity, vec_ty.n_bits()) {
                    (WithinBlocks, 128) => {
                        panic!("This should have been handled by generic_op");
                    }
                    (WithinBlocks, _) | (_, 128) => {
                        quote! {
                            unsafe {
                                dyn_vext_128(self.#to_bytes(a).val.0, self.#to_bytes(b).val.0, #byte_shift)
                            }
                        }
                    }
                    (AcrossBlocks, 256 | 512) => {
                        let num_blocks = vec_ty.n_bits() / 128;

                        // Ranges are not `Copy`, so we need to create a new range iterator for each usage
                        let blocks = (0..num_blocks).map(Literal::usize_unsuffixed);
                        let blocks2 = blocks.clone();
                        let blocks3 = blocks.clone();
                        let bytes_arch_ty = self.arch_ty(&bytes_ty);

                        quote! {
                            unsafe {
                                let a_bytes = self.#to_bytes(a).val.0;
                                let b_bytes = self.#to_bytes(b).val.0;
                                let a_blocks = [#( a_bytes.#blocks ),*];
                                let b_blocks = [#( b_bytes.#blocks2 ),*];

                                let shift_bytes = #byte_shift;
                                #bytes_arch_ty(#({
                                    let [lo, hi] = crate::support::cross_block_slide_blocks_at(&a_blocks, &b_blocks, #blocks3, shift_bytes);
                                    dyn_vext_128(lo, hi, shift_bytes % 16)
                                }),*)
                            }
                        }
                    }
                    _ => unimplemented!(),
                };

                quote! {
                    #method_sig {
                        if SHIFT >= #num_items {
                            return b;
                        }

                        let result = #bytes_expr;
                        self.#from_bytes(#combined_bytes { val: #block_wrapper(result), simd: self })
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
                    self.kernel_method(
                        op,
                        vec_ty,
                        |token| quote! { #neon(a.into()).simd_into(#token) },
                    )
                }
            }
            OpSig::Reinterpret {
                target_ty,
                scalar_bits,
            } => {
                if valid_reinterpret(vec_ty, target_ty, scalar_bits) {
                    let to_ty = vec_ty.reinterpret(target_ty, scalar_bits);
                    let neon = cvt_intrinsic("vreinterpret", &to_ty, vec_ty);

                    self.kernel_method(
                        op,
                        vec_ty,
                        |token| quote! { #neon(a.into()).simd_into(#token) },
                    )
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
                self.kernel_method(
                    op,
                    vec_ty,
                    |_| quote! { #min_max(#reinterpret(a.into())) #target },
                )
            }
            OpSig::MaskFromBitmask => self.handle_mask_from_bitmask(op, vec_ty),
            OpSig::MaskToBitmask => self.handle_mask_to_bitmask(op, vec_ty),
            OpSig::MaskSet => generic_mask_set(method_sig, vec_ty),
            OpSig::FromArray { kind } => generic_from_array(method_sig, vec_ty, kind),
            OpSig::AsArray { kind } => {
                generic_as_array(method_sig, vec_ty, kind, self.max_block_size(), |vec_ty| {
                    self.arch_ty(vec_ty)
                })
            }
            OpSig::StoreArray => generic_store_array(method_sig, vec_ty),
            OpSig::FromBytes => generic_from_bytes(method_sig, vec_ty),
            OpSig::ToBytes => generic_to_bytes(method_sig, vec_ty),
            OpSig::Interleave => {
                let zip_low = generic_op_name("zip_low", vec_ty);
                let zip_high = generic_op_name("zip_high", vec_ty);
                quote! {
                    #method_sig {
                        (self.#zip_low(a, b), self.#zip_high(a, b))
                    }
                }
            }
            OpSig::Deinterleave => {
                let unzip_low = generic_op_name("unzip_low", vec_ty);
                let unzip_high = generic_op_name("unzip_high", vec_ty);
                quote! {
                    #method_sig {
                        (self.#unzip_low(a, b), self.#unzip_high(a, b))
                    }
                }
            }
        }
    }
}

impl Neon {
    fn handle_mask_from_bitmask(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "mask bitmask conversion only operates on masks"
        );
        assert_eq!(
            vec_ty.n_bits(),
            self.native_width(),
            "wide masks should use the generic split implementation"
        );

        self.kernel_method(op, vec_ty, |token| match vec_ty.scalar_bits {
            8 => quote! {
                let shifts =
                    crate::transmute::checked_transmute_copy::<[i16; 8], int16x8_t>(
                        &[15, 14, 13, 12, 11, 10, 9, 8],
                    );
                let lo = vshlq_u16(vdupq_n_u16(bits as u16), shifts);
                let hi = vshlq_u16(vdupq_n_u16((bits >> 8) as u16), shifts);
                let lo = vcltq_s16(vreinterpretq_s16_u16(lo), vdupq_n_s16(0));
                let hi = vcltq_s16(vreinterpretq_s16_u16(hi), vdupq_n_s16(0));
                vcombine_s8(
                    vmovn_s16(vreinterpretq_s16_u16(lo)),
                    vmovn_s16(vreinterpretq_s16_u16(hi)),
                ).simd_into(#token)
            },
            16 => quote! {
                let shifts =
                    crate::transmute::checked_transmute_copy::<[i16; 8], int16x8_t>(
                        &[15, 14, 13, 12, 11, 10, 9, 8],
                    );
                let shifted = vshlq_u16(vdupq_n_u16(bits as u16), shifts);
                let mask = vcltq_s16(vreinterpretq_s16_u16(shifted), vdupq_n_s16(0));
                vreinterpretq_s16_u16(mask).simd_into(#token)
            },
            32 => quote! {
                let shifts =
                    crate::transmute::checked_transmute_copy::<[i32; 4], int32x4_t>(
                        &[31, 30, 29, 28],
                    );
                let shifted = vshlq_u32(vdupq_n_u32(bits as u32), shifts);
                let mask = vcltq_s32(vreinterpretq_s32_u32(shifted), vdupq_n_s32(0));
                vreinterpretq_s32_u32(mask).simd_into(#token)
            },
            64 => quote! {
                let shifts =
                    crate::transmute::checked_transmute_copy::<[i64; 2], int64x2_t>(
                        &[63, 62],
                    );
                let shifted = vshlq_u64(vdupq_n_u64(bits), shifts);
                let mask = vcltq_s64(vreinterpretq_s64_u64(shifted), vdupq_n_s64(0));
                vreinterpretq_s64_u64(mask).simd_into(#token)
            },
            _ => unimplemented!(),
        })
    }

    fn handle_mask_to_bitmask(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "mask bitmask conversion only operates on masks"
        );
        assert_eq!(
            vec_ty.n_bits(),
            self.native_width(),
            "wide masks should use the generic split implementation"
        );

        self.kernel_method(op, vec_ty, |_| match vec_ty.scalar_bits {
            8 => quote! {
                let weights =
                    crate::transmute::checked_transmute_copy::<[u8; 16], uint8x16_t>(
                        &[
                            1, 2, 4, 8, 16, 32, 64, 128,
                            1, 2, 4, 8, 16, 32, 64, 128,
                        ],
                    );
                let bits = vandq_u8(vreinterpretq_u8_s8(a.into()), weights);
                let lo = vaddv_u8(vget_low_u8(bits)) as u64;
                let hi = vaddv_u8(vget_high_u8(bits)) as u64;
                lo | (hi << 8)
            },
            16 => quote! {
                let weights =
                    crate::transmute::checked_transmute_copy::<[u16; 8], uint16x8_t>(
                        &[1, 2, 4, 8, 16, 32, 64, 128],
                    );
                let bits = vandq_u16(vreinterpretq_u16_s16(a.into()), weights);
                vaddvq_u16(bits) as u64
            },
            32 => quote! {
                let weights =
                    crate::transmute::checked_transmute_copy::<[u32; 4], uint32x4_t>(
                        &[1, 2, 4, 8],
                    );
                let bits = vandq_u32(vreinterpretq_u32_s32(a.into()), weights);
                vaddvq_u32(bits) as u64
            },
            64 => quote! {
                let weights =
                    crate::transmute::checked_transmute_copy::<[u64; 2], uint64x2_t>(
                        &[1, 2],
                    );
                let bits = vandq_u64(vreinterpretq_u64_s64(a.into()), weights);
                vaddvq_u64(bits)
            },
            _ => unimplemented!(),
        })
    }
}

fn mk_slide_helpers() -> TokenStream {
    let shifts = (0_usize..16).map(|shift| {
        let shift_i32 = i32::try_from(shift).unwrap();
        quote! { #shift => vextq_u8::<#shift_i32>(a, b) }
    });

    quote! {
        /// This is a version of the `vext` intrinsic that takes a non-const shift argument. The shift is still
        /// expected to be constant in practice, so the match statement will be optimized out. This exists because
        /// Rust doesn't currently let you do math on const generics.
        #[inline(always)]
        unsafe fn dyn_vext_128(a: uint8x16_t, b: uint8x16_t, shift: usize) -> uint8x16_t {
            unsafe {
                match shift {
                    #(#shifts,)*
                    _ => unreachable!()
                }
            }
        }
    }
}
