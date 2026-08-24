// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{ToTokens as _, format_ident, quote};

use crate::generic::{
    fallback_method, generic_mask_set, generic_op_name, integer_lane_mask_splat_arg,
};
use crate::level::Level;
use crate::ops::{CoreOpTrait, NarrowingMode, Op, SlideGranularity, relaxed_narrow_method};
use crate::{
    arch::neon::{self, cvt_intrinsic, simple_intrinsic, split_intrinsic},
    ops::OpSig,
    types::{ScalarType, VecType},
};

#[derive(Clone, Copy)]
pub(crate) struct Neon;

fn neon_multi_vector_ty(vec_ty: &VecType, count: u16) -> Ident {
    let scalar = match vec_ty.scalar {
        ScalarType::Float => "float",
        ScalarType::Unsigned => "uint",
        ScalarType::Int => "int",
        ScalarType::Mask => unreachable!("interleaved operations are not defined for masks"),
    };
    Ident::new(
        &format!("{scalar}{}x{}x{count}_t", vec_ty.scalar_bits, vec_ty.len),
        Span::call_site(),
    )
}

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
        let features = self
            .enabled_target_features()
            .expect("Neon always enables target features");

        quote! {
            /// Create a SIMD token proving that Neon is available.
            ///
            /// This function can be called safely from a function with the `neon` target feature
            /// enabled.
            ///
            /// # Safety
            ///
            /// When invoking this function through an `unsafe` block, the caller must ensure that
            /// the current CPU supports `neon`.
            #[inline]
            #[target_feature(enable = #features)]
            pub const fn assume_supported() -> Self {
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
                assert_eq!(
                    vec_ty.n_bits(),
                    block_size as usize,
                    "NEON interleaved loads require full block-sized vectors"
                );
                let intrinsic = simple_intrinsic(&format!("vld{block_count}"), vec_ty);
                let fields = (0..block_count).map(Literal::u16_unsuffixed);

                quote! {
                    #method_sig {
                        let native = unsafe { #intrinsic(src.as_ptr()) };
                        [#(native.#fields.simd_into(self)),*]
                    }
                }
            }
            OpSig::StoreInterleaved {
                block_size,
                block_count,
            } => {
                assert_eq!(
                    vec_ty.n_bits(),
                    block_size as usize,
                    "NEON interleaved stores require full block-sized vectors"
                );
                let intrinsic = simple_intrinsic(&format!("vst{block_count}"), vec_ty);
                let aggregate_ty = neon_multi_vector_ty(vec_ty, block_count);
                let indices = 0..block_count as usize;
                let native_ty = self.arch_ty(vec_ty);
                let native_values = (0..block_count as usize)
                    .map(|idx| format_ident!("v{idx}"))
                    .collect::<Vec<_>>();
                let native_decls = native_values.iter().zip(indices).map(|(value, idx)| {
                    quote! { let #value: #native_ty = vectors[#idx].into(); }
                });

                quote! {
                    #method_sig {
                        #(#native_decls)*
                        unsafe { #intrinsic(dest.as_mut_ptr(), #aggregate_ty(#(#native_values),*)); }
                    }
                }
            }
            OpSig::Widen { target_ty: _ } => {
                if vec_ty.scalar == ScalarType::Float {
                    return self.kernel_method(op, vec_ty, |token| {
                        quote! {
                            (
                                vcvt_f64_f32(vget_low_f32(a.into())).simd_into(#token),
                                vcvt_high_f64_f32(a.into()).simd_into(#token),
                            )
                        }
                    });
                }

                let src_scalar = match vec_ty.scalar {
                    ScalarType::Int => format!("s{}", vec_ty.scalar_bits),
                    ScalarType::Unsigned => format!("u{}", vec_ty.scalar_bits),
                    _ => unreachable!(),
                };
                let movl = Ident::new(&format!("vmovl_{src_scalar}"), Span::call_site());
                let get_low = Ident::new(&format!("vget_low_{src_scalar}"), Span::call_site());
                let get_high = Ident::new(&format!("vget_high_{src_scalar}"), Span::call_site());
                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        (
                            #movl(#get_low(a.into())).simd_into(#token),
                            #movl(#get_high(a.into())).simd_into(#token),
                        )
                    }
                })
            }
            OpSig::Narrow { target_ty, mode } => {
                if mode == NarrowingMode::Relaxed {
                    return relaxed_narrow_method(op, vec_ty, target_ty, "narrow");
                }

                if vec_ty.scalar == ScalarType::Float {
                    if mode == NarrowingMode::Saturate {
                        let narrow = generic_op_name("narrow", vec_ty);
                        return quote! {
                            #method_sig {
                                self.#narrow(a, b)
                            }
                        };
                    }

                    return self.kernel_method(op, vec_ty, |token| {
                        quote! {
                            vcvt_high_f32_f64(vcvt_f32_f64(a.into()), b.into()).simd_into(#token)
                        }
                    });
                }

                let prefix = match vec_ty.scalar {
                    ScalarType::Int => "s",
                    ScalarType::Unsigned => "u",
                    _ => unreachable!(),
                };
                let src_scalar = format!("{prefix}{}", vec_ty.scalar_bits);
                let target_scalar = format!("{prefix}{}", target_ty.scalar_bits);
                let method_prefix = if mode == NarrowingMode::Saturate {
                    "vqmovn"
                } else {
                    "vmovn"
                };
                let narrow =
                    Ident::new(&format!("{method_prefix}_{src_scalar}"), Span::call_site());
                let combine = Ident::new(&format!("vcombine_{target_scalar}"), Span::call_site());
                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        #combine(#narrow(a.into()), #narrow(b.into())).simd_into(#token)
                    }
                })
            }
            OpSig::Binary => {
                if vec_ty.scalar_bits == 64
                    && matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned)
                    && matches!(method, "mul" | "min" | "max")
                {
                    return fallback_method(op, vec_ty);
                }

                self.kernel_method(op, vec_ty, |token| match method {
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
                })
            }
            OpSig::Ternary => {
                if method == "mul_add_precise" {
                    let mul_add = generic_op_name("mul_add", vec_ty);
                    return quote! {
                        #method_sig {
                            self.#mul_add(a, b, c)
                        }
                    };
                }
                if method == "mul_sub" {
                    let mul_add = generic_op_name("mul_add", vec_ty);
                    return quote! {
                        #method_sig {
                            self.#mul_add(a, b, -c)
                        }
                    };
                }
                if method == "mul_sub_precise" {
                    let mul_sub = generic_op_name("mul_sub", vec_ty);
                    return quote! {
                        #method_sig {
                            self.#mul_sub(a, b, c)
                        }
                    };
                }

                let args = [
                    quote! { c.into() },
                    quote! { b.into() },
                    quote! { a.into() },
                ];
                let expr = neon::expr(method, vec_ty, &args);
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
                            dyn_vext_128(
                                self,
                                Bytes::to_bytes(a).val.0,
                                Bytes::to_bytes(b).val.0,
                                #byte_shift,
                            )
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
                            {
                                let a_bytes = Bytes::to_bytes(a).val.0;
                                let b_bytes = Bytes::to_bytes(b).val.0;
                                let a_blocks = [#( a_bytes.#blocks ),*];
                                let b_blocks = [#( b_bytes.#blocks2 ),*];

                                let shift_bytes = #byte_shift;
                                #bytes_arch_ty(#({
                                    let [lo, hi] = crate::support::cross_block_slide_blocks_at(&a_blocks, &b_blocks, #blocks3, shift_bytes);
                                    dyn_vext_128(self, lo, hi, shift_bytes % 16)
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
                        Bytes::from_bytes(#combined_bytes {
                            val: #block_wrapper(result),
                            simd: self,
                        })
                    }
                }
            }
            OpSig::SwizzleDynWithinBlocks => {
                assert_eq!(
                    vec_ty.n_bits(),
                    self.native_width(),
                    "wide swizzles should use the generic split implementation"
                );

                let bytes_ty = vec_ty.bytes_ty();
                let bytes = bytes_ty.rust();
                let wrapper = bytes_ty.aligned_wrapper();

                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        let result =
                            vqtbl1q_u8(Bytes::to_bytes(a).val.0, indices.into());
                        Bytes::from_bytes(#bytes {
                            val: #wrapper(result),
                            simd: #token,
                        })
                    }
                })
            }
            OpSig::SwizzleDyn => {
                let precise = generic_op_name("swizzle_dyn_precise", vec_ty);
                quote! {
                    #method_sig {
                        self.#precise(a, indices)
                    }
                }
            }
            OpSig::SwizzleDynPrecise => {
                let bytes_ty = vec_ty.bytes_ty();
                let bytes = bytes_ty.rust();
                let wrapper = bytes_ty.aligned_wrapper();

                self.kernel_method(op, vec_ty, |token| {
                    let body = match vec_ty.n_bits() {
                        128 => quote! {
                            let result = vqtbl1q_u8(Bytes::to_bytes(a).val.0, indices.into());
                        },
                        256 => quote! {
                            let table = Bytes::to_bytes(a).val.0;
                            let indices: uint8x16x2_t = indices.into();
                            let result = uint8x16x2_t(
                                vqtbl2q_u8(table, indices.0),
                                vqtbl2q_u8(table, indices.1),
                            );
                        },
                        512 => quote! {
                            let table = Bytes::to_bytes(a).val.0;
                            let indices: uint8x16x4_t = indices.into();
                            let result = uint8x16x4_t(
                                vqtbl4q_u8(table, indices.0),
                                vqtbl4q_u8(table, indices.1),
                                vqtbl4q_u8(table, indices.2),
                                vqtbl4q_u8(table, indices.3),
                            );
                        },
                        _ => unreachable!(),
                    };

                    quote! {
                        #body
                        Bytes::from_bytes(#bytes { val: #wrapper(result), simd: #token })
                    }
                })
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
            OpSig::BitwiseReduction { op: combine_op } => {
                self.handle_bitwise_reduction(op, vec_ty, combine_op)
            }
            OpSig::MaskFromBitmask => self.handle_mask_from_bitmask(op, vec_ty),
            OpSig::MaskToBitmask => self.handle_mask_to_bitmask(op, vec_ty),
            OpSig::MaskSet => generic_mask_set(method_sig, vec_ty),
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

    fn should_use_generic_op(&self, op: &Op, vec_ty: &VecType) -> bool {
        if matches!(op.sig, OpSig::MaskToBitmask) {
            false
        } else {
            op.sig.should_use_generic_op(vec_ty, self.native_width())
        }
    }
}

impl Neon {
    fn handle_bitwise_reduction(
        &self,
        op: Op,
        vec_ty: &VecType,
        combine_op: CoreOpTrait,
    ) -> TokenStream {
        assert_eq!(vec_ty.n_bits(), 128);
        assert!(matches!(
            combine_op,
            CoreOpTrait::BitAnd | CoreOpTrait::BitOr | CoreOpTrait::BitXor
        ));

        let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
        let scalar_prefix = match vec_ty.scalar {
            ScalarType::Int => "s",
            ScalarType::Unsigned => "u",
            _ => unreachable!("bitwise reductions only operate on integers"),
        };
        let bits = if vec_ty.scalar == ScalarType::Unsigned && vec_ty.scalar_bits == 64 {
            quote! { a.into() }
        } else {
            let reinterpret =
                format_ident!("vreinterpretq_u64_{}{}", scalar_prefix, vec_ty.scalar_bits);
            quote! { #reinterpret(a.into()) }
        };
        let vector_combine = match combine_op {
            CoreOpTrait::BitAnd => quote! { vand_u64 },
            CoreOpTrait::BitOr => quote! { vorr_u64 },
            CoreOpTrait::BitXor => quote! { veor_u64 },
            _ => unreachable!(),
        };
        let shifts = (vec_ty.scalar_bits.ilog2()..6)
            .rev()
            .map(|power| Literal::usize_unsuffixed(1 << power))
            .collect::<Vec<_>>();
        let reduced_init = if shifts.is_empty() {
            quote! { let reduced = vget_lane_u64::<0>(halves); }
        } else {
            quote! { let mut reduced = vget_lane_u64::<0>(halves); }
        };
        let scalar_steps = match combine_op {
            CoreOpTrait::BitAnd => quote! { #(reduced &= reduced >> #shifts;)* },
            CoreOpTrait::BitOr => quote! { #(reduced |= reduced >> #shifts;)* },
            CoreOpTrait::BitXor => quote! { #(reduced ^= reduced >> #shifts;)* },
            _ => unreachable!(),
        };

        self.kernel_method(op, vec_ty, |_| {
            quote! {
                let bits = #bits;
                let halves = #vector_combine(vget_low_u64(bits), vget_high_u64(bits));
                #reduced_init
                #scalar_steps
                reduced as #scalar
            }
        })
    }

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

        let body = match (vec_ty.n_bits(), vec_ty.scalar_bits) {
            (128, 8) => quote! {
                let weights =
                    crate::transmute::checked_transmute_copy::<[u8; 16], uint8x16_t>(
                        &[
                            1, 2, 4, 8, 16, 32, 64, 128,
                            1, 2, 4, 8, 16, 32, 64, 128,
                        ],
                    );
                let bits = vandq_u8(vreinterpretq_u8_s8(a.into()), weights);
                let rotated = vextq_u8::<8>(bits, bits);
                let paired = vzip1q_u8(bits, rotated);
                vaddvq_u16(vreinterpretq_u16_u8(paired)) as u64
            },
            (128, 16) => quote! {
                let weights =
                    crate::transmute::checked_transmute_copy::<[u16; 8], uint16x8_t>(
                        &[1, 2, 4, 8, 16, 32, 64, 128],
                    );
                let bits = vandq_u16(vreinterpretq_u16_s16(a.into()), weights);
                vaddvq_u16(bits) as u64
            },
            (128, 32) => quote! {
                let weights =
                    crate::transmute::checked_transmute_copy::<[u32; 4], uint32x4_t>(
                        &[1, 2, 4, 8],
                    );
                let bits = vandq_u32(vreinterpretq_u32_s32(a.into()), weights);
                vaddvq_u32(bits) as u64
            },
            (128, 64) => quote! {
                let weights =
                    crate::transmute::checked_transmute_copy::<[u64; 2], uint64x2_t>(
                        &[1, 2],
                    );
                let bits = vandq_u64(vreinterpretq_u64_s64(a.into()), weights);
                vaddvq_u64(bits)
            },
            // Pack 32 canonical mask bytes in two vector narrowing stages. The first SHRN reduces
            // each adjacent byte pair to one byte; SLI arranges the retained bits so that the
            // second SHRN reduces each group of four lanes to the low nibble of one byte. Moving
            // those eight nibbles to a scalar register and merging them with a SWAR tree produces
            // the final 32-bit mask. Keeping the final stages scalar avoids a serial vector
            // reduction while preserving good throughput across both in-order and OoO cores.
            (256, 8) => quote! {
                let a: int8x16x2_t = a.into();
                let lo = vreinterpretq_u16_u8(vreinterpretq_u8_s8(a.0));
                let hi = vreinterpretq_u16_u8(vreinterpretq_u8_s8(a.1));

                let pairs_lo = vshrn_n_u16::<7>(lo);
                let pairs_hi = vshrn_n_u16::<7>(hi);
                let pairs = vcombine_u8(pairs_lo, pairs_hi);
                let pairs = vsliq_n_u8::<6>(pairs, pairs);
                let nibbles = vshrn_n_u16::<6>(vreinterpretq_u16_u8(pairs));

                let mut bits = vget_lane_u64::<0>(vreinterpret_u64_u8(nibbles));
                bits &= 0x0f0f_0f0f_0f0f_0f0f;
                bits = (bits | (bits >> 4)) & 0x00ff_00ff_00ff_00ff;
                bits = (bits | (bits >> 8)) & 0x0000_ffff_0000_ffff;
                (bits | (bits >> 16)) & 0xffff_ffff
            },
            // Pack 64 canonical mask bytes with a constant-free butterfly. Each UZP1/UZP2 pair
            // separates alternating lane groups, and SRI inserts their high mask bits into the
            // same byte: the three levels therefore pack groups of 2, then 4, then 8 lanes. The
            // final level deliberately operates on the distinct low and high D halves. Writing it
            // as UZP(x, x) makes LLVM expand the intended three-instruction stage into seven.
            (512, 8) => quote! {
                let a: int8x16x4_t = a.into();
                let a0 = vreinterpretq_u8_s8(a.0);
                let a1 = vreinterpretq_u8_s8(a.1);
                let a2 = vreinterpretq_u8_s8(a.2);
                let a3 = vreinterpretq_u8_s8(a.3);

                let lo = vsriq_n_u8::<1>(vuzp2q_u8(a0, a1), vuzp1q_u8(a0, a1));
                let hi = vsriq_n_u8::<1>(vuzp2q_u8(a2, a3), vuzp1q_u8(a2, a3));
                let packed = vsriq_n_u8::<2>(vuzp2q_u8(lo, hi), vuzp1q_u8(lo, hi));
                let lo = vget_low_u8(packed);
                let hi = vget_high_u8(packed);
                let packed = vsri_n_u8::<4>(vuzp2_u8(lo, hi), vuzp1_u8(lo, hi));
                vget_lane_u64::<0>(vreinterpret_u64_u8(packed))
            },
            (256 | 512, 16 | 32 | 64) => {
                return self.handle_wide_mask_to_bitmask(op, vec_ty);
            }
            _ => unimplemented!(),
        };

        self.kernel_method(op, vec_ty, |_| body)
    }

    fn handle_wide_mask_to_bitmask(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        // Canonical mask lanes contain either all zeroes or all ones, so retaining either half of
        // every lane preserves its value. Pairing adjacent Q registers with UZP1 therefore maps
        // (total width, element width) to (width / 2, element width / 2) without changing the
        // logical lane count or order. Recurse diagonally until reaching a native-width reduction
        // or one of the byte-mask packers above.
        //
        // Keep this expressed as UZP1 rather than XTN plus `vcombine`. Although the operations are
        // equivalent for canonical masks, LLVM 22.1.6 misses useful folds for the latter form.
        let reduced_ty = VecType::new(ScalarType::Mask, vec_ty.scalar_bits / 2, vec_ty.len);
        let reduced_method = generic_op_name("to_bitmask", &reduced_ty);
        let native_ty = self.arch_ty(vec_ty);
        let reduced_native_ty = self.arch_ty(&reduced_ty);
        let half_bits = vec_ty.scalar_bits / 2;
        let reinterpret = format_ident!("vreinterpretq_u{half_bits}_s{}", vec_ty.scalar_bits);
        let reinterpret_signed = format_ident!("vreinterpretq_s{half_bits}_u{half_bits}");
        let unzip_low = format_ident!("vuzp1q_u{half_bits}");

        self.kernel_method(op, vec_ty, |token| {
            let reduced = match vec_ty.n_bits() {
                256 => quote! {
                    let reduced = #reinterpret_signed(#unzip_low(
                        #reinterpret(a.0),
                        #reinterpret(a.1),
                    ));
                },
                512 => quote! {
                    let lo = #reinterpret_signed(#unzip_low(
                        #reinterpret(a.0),
                        #reinterpret(a.1),
                    ));
                    let hi = #reinterpret_signed(#unzip_low(
                        #reinterpret(a.2),
                        #reinterpret(a.3),
                    ));
                    let reduced = #reduced_native_ty(lo, hi);
                },
                _ => unreachable!(),
            };

            quote! {
                let a: #native_ty = a.into();
                #reduced
                #token.#reduced_method(reduced.simd_into(#token))
            }
        })
    }
}

fn mk_slide_helpers() -> TokenStream {
    let shifts = (0_usize..16).map(|shift| {
        let shift_i32 = i32::try_from(shift).unwrap();
        quote! { #shift => vextq_u8::<#shift_i32>(a, b) }
    });

    quote! {
        crate::kernel!(
            /// This is a version of the `vext` intrinsic that takes a non-const shift argument. The shift is still
            /// expected to be constant in practice, so the match statement will be optimized out. This exists because
            /// Rust doesn't currently let you do math on const generics.
            #[inline(always)]
            fn dyn_vext_128(neon: Neon, a: uint8x16_t, b: uint8x16_t, shift: usize) -> uint8x16_t {
                match shift {
                    #(#shifts,)*
                    _ => unreachable!()
                }
            }
        );
    }
}
