// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

use crate::arch::wasm::{arch_prefix, v128_intrinsic};
use crate::generic::{
    generic_as_array, generic_block_combine, generic_block_split, generic_from_array,
    generic_from_bytes, generic_op_name, generic_to_array, generic_to_bytes, scalar_binary,
};
use crate::level::Level;
use crate::ops::{Op, Quantifier, valid_reinterpret};
use crate::{
    arch::wasm::{self, simple_intrinsic},
    ops::OpSig,
    types::{ScalarType, VecType},
};

#[derive(Clone, Copy)]
pub(crate) struct WasmSimd128;

impl Level for WasmSimd128 {
    fn name(&self) -> &'static str {
        "WasmSimd128"
    }

    fn native_width(&self) -> usize {
        128
    }

    fn max_block_size(&self) -> usize {
        128
    }

    fn enabled_target_features(&self) -> Option<&'static str> {
        None
    }

    fn arch_ty(&self, _vec_ty: &VecType) -> TokenStream {
        quote! { v128 }
    }

    fn token_doc(&self) -> &'static str {
        r#"The SIMD token for the "wasm128" level."#
    }

    fn token_inner(&self) -> TokenStream {
        quote!(crate::core_arch::wasm32::WasmSimd128)
    }

    fn make_module_prelude(&self) -> TokenStream {
        quote! {
            use core::arch::wasm32::*;
        }
    }

    fn make_impl_body(&self) -> TokenStream {
        quote! {
            #[inline]
            pub const fn new_unchecked() -> Self {
                Self { wasmsimd128: crate::core_arch::wasm32::WasmSimd128::new() }
            }
        }
    }

    fn make_method(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let Op { sig, method, .. } = op;

        let method_sig = op.simd_trait_method_sig(vec_ty);
        match sig {
            OpSig::Splat => {
                let expr = wasm::expr(method, vec_ty, &[quote! { val }]);
                quote! {
                    #method_sig {
                        #expr.simd_into(self)
                    }
                }
            }
            OpSig::Unary => {
                let args = [quote! { a.into() }];
                let expr = if matches!(method, "fract") {
                    assert_eq!(
                        vec_ty.scalar,
                        ScalarType::Float,
                        "only float supports fract"
                    );

                    let trunc = generic_op_name("trunc", vec_ty);
                    let sub = generic_op_name("sub", vec_ty);
                    quote! {
                        self.#sub(a, self.#trunc(a))
                    }
                } else {
                    let expr = wasm::expr(method, vec_ty, &args);
                    quote! { #expr.simd_into(self) }
                };

                quote! {
                    #method_sig {
                        #expr
                    }
                }
            }
            OpSig::Binary => {
                let args = [quote! { a.into() }, quote! { b.into() }];
                let expr = match method {
                    "mul" if vec_ty.scalar_bits == 8 && vec_ty.len == 16 => {
                        let (extmul_low, extmul_high) = match vec_ty.scalar {
                            ScalarType::Unsigned => (
                                quote! { u16x8_extmul_low_u8x16 },
                                quote! { u16x8_extmul_high_u8x16 },
                            ),
                            ScalarType::Int => (
                                quote! { i16x8_extmul_low_i8x16 },
                                quote! { i16x8_extmul_high_i8x16 },
                            ),
                            _ => unreachable!(),
                        };

                        quote! {
                            let low = #extmul_low(a.into(), b.into());
                            let high = #extmul_high(a.into(), b.into());
                            u8x16_shuffle::<0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30>(low, high).simd_into(self)
                        }
                    }
                    "max_precise" | "min_precise" => {
                        let intrinsic = simple_intrinsic(
                            if method == "max_precise" {
                                "pmax"
                            } else {
                                "pmin"
                            },
                            vec_ty,
                        );
                        let compare_ne = simple_intrinsic("ne", vec_ty);
                        quote! {
                            let intermediate = #intrinsic(b.into(), a.into());

                            // See the x86 min_precise/max_precise code in `arch::x86` for more info on how this
                            // works.
                            let b_is_nan = #compare_ne(b.into(), b.into());
                            v128_bitselect(a.into(), intermediate, b_is_nan).simd_into(self)
                        }
                    }
                    "max" | "min" if vec_ty.scalar == ScalarType::Float => {
                        let expr = wasm::expr(method, vec_ty, &args);
                        let relaxed_intrinsic = simple_intrinsic(
                            if method == "max" {
                                "relaxed_max"
                            } else {
                                "relaxed_min"
                            },
                            vec_ty,
                        );
                        let relaxed_expr = quote! { #relaxed_intrinsic ( #( #args ),* ) };

                        quote! {
                            #[cfg(target_feature = "relaxed-simd")]
                            { #relaxed_expr.simd_into(self) }

                            #[cfg(not(target_feature = "relaxed-simd"))]
                            { #expr.simd_into(self) }
                        }
                    }
                    "shlv" => scalar_binary(quote!(core::ops::Shl::shl)),
                    "shrv" => scalar_binary(quote!(core::ops::Shr::shr)),
                    "copysign" => {
                        let splat = simple_intrinsic("splat", vec_ty);
                        let sign_mask_literal = match vec_ty.scalar_bits {
                            32 => quote! { -0.0_f32 },
                            64 => quote! { -0.0_f64 },
                            _ => unimplemented!(),
                        };
                        quote! {
                            let sign_mask = #splat(#sign_mask_literal);
                            let sign_bits = v128_and(b.into(), sign_mask.into());
                            let magnitude = v128_andnot(a.into(), sign_mask.into());
                            v128_or(magnitude, sign_bits).simd_into(self)
                        }
                    }
                    _ => {
                        let expr = wasm::expr(method, vec_ty, &args);
                        quote! { #expr.simd_into(self) }
                    }
                };

                quote! {
                    #method_sig {
                        #expr
                    }
                }
            }
            OpSig::Ternary => {
                if matches!(method, "mul_add" | "mul_sub") {
                    let add_sub =
                        generic_op_name(if method == "mul_add" { "add" } else { "sub" }, vec_ty);
                    let mul = generic_op_name("mul", vec_ty);

                    let c = if method == "mul_sub" {
                        // WebAssembly just... forgot fused multiply-subtract? It seems the
                        // initial proposal
                        // (https://github.com/WebAssembly/relaxed-simd/issues/27) confused it
                        // with negate multiply-add, and nobody ever resolved the confusion.
                        let negate = simple_intrinsic("neg", vec_ty);
                        quote! { #negate(c.into()) }
                    } else {
                        quote! { c.into() }
                    };
                    let relaxed_madd = simple_intrinsic("relaxed_madd", vec_ty);

                    quote! {
                        #method_sig {
                            #[cfg(target_feature = "relaxed-simd")]
                            { #relaxed_madd(a.into(), b.into(), #c).simd_into(self) }

                            #[cfg(not(target_feature = "relaxed-simd"))]
                            { self.#add_sub(self.#mul(a, b), c) }
                        }
                    }
                } else {
                    unimplemented!()
                }
            }
            OpSig::Compare => {
                let args = [quote! { a.into() }, quote! { b.into() }];
                let expr = wasm::expr(method, vec_ty, &args);
                quote! {
                    #method_sig {
                        #expr.simd_into(self)
                    }
                }
            }
            OpSig::Select => {
                // Rust includes unsigned versions of the lane select intrinsics, but they're
                // just aliases for the signed ones
                let lane_ty = vec_ty.cast(ScalarType::Int);
                let lane_select = simple_intrinsic("relaxed_laneselect", &lane_ty);

                quote! {
                    #method_sig {
                        #[cfg(target_feature = "relaxed-simd")]
                        { #lane_select(b.into(), c.into(), a.into()).simd_into(self) }

                        #[cfg(not(target_feature = "relaxed-simd"))]
                        { v128_bitselect(b.into(), c.into(), a.into()).simd_into(self) }
                    }
                }
            }
            OpSig::Combine { combined_ty } => generic_block_combine(method_sig, &combined_ty, 128),
            OpSig::Split { half_ty } => generic_block_split(method_sig, &half_ty, 128),
            OpSig::Zip { select_low } => {
                let (indices, shuffle_fn) = match vec_ty.scalar_bits {
                    8 => {
                        let indices = if select_low {
                            quote! { 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23 }
                        } else {
                            quote! { 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31 }
                        };
                        (indices, quote! { u8x16_shuffle })
                    }
                    16 => {
                        let indices = if select_low {
                            quote! { 0, 8, 1, 9, 2, 10, 3, 11 }
                        } else {
                            quote! { 4, 12, 5, 13, 6, 14, 7, 15 }
                        };
                        (indices, quote! { u16x8_shuffle })
                    }
                    32 => {
                        let indices = if select_low {
                            quote! { 0, 4, 1, 5 }
                        } else {
                            quote! { 2, 6, 3, 7 }
                        };
                        (indices, quote! { u32x4_shuffle })
                    }
                    64 => {
                        let indices = if select_low {
                            quote! { 0, 2 }
                        } else {
                            quote! { 1, 3 }
                        };
                        (indices, quote! { u64x2_shuffle })
                    }
                    _ => panic!("unsupported scalar_bits for zip operation"),
                };

                quote! {
                    #method_sig {
                        #shuffle_fn::<#indices>(a.into(), b.into()).simd_into(self)
                    }
                }
            }
            OpSig::Unzip { select_even } => {
                let (indices, shuffle_fn) = match vec_ty.scalar_bits {
                    8 => {
                        let indices = if select_even {
                            quote! { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 }
                        } else {
                            quote! { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 }
                        };
                        (indices, quote! { u8x16_shuffle })
                    }
                    16 => {
                        let indices = if select_even {
                            quote! { 0, 2, 4, 6, 8, 10, 12, 14 }
                        } else {
                            quote! { 1, 3, 5, 7, 9, 11, 13, 15 }
                        };
                        (indices, quote! { u16x8_shuffle })
                    }
                    32 => {
                        let indices = if select_even {
                            quote! { 0, 2, 4, 6 }
                        } else {
                            quote! { 1, 3, 5, 7 }
                        };
                        (indices, quote! { u32x4_shuffle })
                    }
                    64 => {
                        let indices = if select_even {
                            quote! { 0, 2 }
                        } else {
                            quote! { 1, 3 }
                        };
                        (indices, quote! { u64x2_shuffle })
                    }
                    _ => panic!("unsupported scalar_bits for unzip operation"),
                };
                quote! {
                    #method_sig {
                        #shuffle_fn::<#indices>(a.into(), b.into()).simd_into(self)
                    }
                }
            }
            OpSig::Shift => {
                let prefix = vec_ty.scalar.prefix();
                let shift_name = format!("{prefix}{}x{}_{method}", vec_ty.scalar_bits, vec_ty.len);
                let shift_fn = Ident::new(&shift_name, Span::call_site());

                quote! {
                    #method_sig {
                        #shift_fn(a.into(), shift).simd_into(self)
                    }
                }
            }
            OpSig::Reinterpret {
                target_ty,
                scalar_bits,
            } => {
                assert!(
                    valid_reinterpret(vec_ty, target_ty, scalar_bits),
                    "The underlying data for WASM SIMD is a v128, so a reinterpret is just that, a reinterpretation of the v128."
                );

                quote! {
                    #method_sig {
                        <v128>::from(a).simd_into(self)
                    }
                }
            }
            OpSig::Cvt {
                target_ty,
                scalar_bits,
                precise,
            } => {
                let (op, uses_relaxed) = match (vec_ty.scalar, target_ty, precise) {
                    (ScalarType::Float, ScalarType::Int | ScalarType::Unsigned, false) => {
                        ("relaxed_trunc", true)
                    }
                    (ScalarType::Float, ScalarType::Int | ScalarType::Unsigned, true) => {
                        ("trunc_sat", false)
                    }
                    (ScalarType::Int | ScalarType::Unsigned, ScalarType::Float, _) => {
                        ("convert", false)
                    }
                    _ => unimplemented!(),
                };
                let dst_ty = arch_prefix(&vec_ty.reinterpret(target_ty, scalar_bits));
                let src_ty = arch_prefix(vec_ty);
                let conversion_fn = format_ident!("{dst_ty}_{op}_{src_ty}");

                if uses_relaxed {
                    let precise = generic_op_name(&[method, "_precise"].join(""), vec_ty);
                    quote! {
                        #method_sig {
                            #[cfg(target_feature = "relaxed-simd")]
                            { #conversion_fn(a.into()).simd_into(self) }

                            #[cfg(not(target_feature = "relaxed-simd"))]
                            { self.#precise(a) }
                        }
                    }
                } else {
                    quote! {
                        #method_sig {
                            #conversion_fn(a.into()).simd_into(self)
                        }
                    }
                }
            }
            OpSig::WidenNarrow { target_ty } => {
                match method {
                    "widen" => {
                        assert_eq!(
                            vec_ty.rust_name(),
                            "u8x16",
                            "Currently only u8x16 -> u16x16 widening is supported"
                        );
                        assert_eq!(
                            target_ty.rust_name(),
                            "u16x16",
                            "Currently only u8x16 -> u16x16 widening is supported"
                        );
                        quote! {
                            #method_sig {
                                let low = u16x8_extend_low_u8x16(a.into());
                                let high = u16x8_extend_high_u8x16(a.into());
                                self.combine_u16x8(low.simd_into(self), high.simd_into(self))
                            }
                        }
                    }
                    "narrow" => {
                        assert_eq!(
                            vec_ty.rust_name(),
                            "u16x16",
                            "Currently only u16x16 -> u8x16 narrowing is supported"
                        );
                        assert_eq!(
                            target_ty.rust_name(),
                            "u8x16",
                            "Currently only u16x16 -> u8x16 narrowing is supported"
                        );
                        // WASM SIMD only has saturating narrowing instructions, so we emulate
                        // truncated narrowing by masking out the
                        quote! {
                            #method_sig {
                                let mask = u16x8_splat(0xFF);
                                let (low, high) = self.split_u16x16(a);
                                let low_masked = v128_and(low.into(), mask);
                                let high_masked = v128_and(high.into(), mask);
                                let result = u8x16_narrow_i16x8(low_masked, high_masked);
                                result.simd_into(self)
                            }
                        }
                    }
                    _ => unimplemented!(),
                }
            }
            OpSig::MaskReduce {
                quantifier,
                condition,
            } => {
                let (intrinsic, negate) = match (quantifier, condition) {
                    (Quantifier::Any, true) => (v128_intrinsic("any_true"), None),
                    (Quantifier::Any, false) => {
                        (simple_intrinsic("all_true", vec_ty), Some(quote! { ! }))
                    }
                    (Quantifier::All, true) => (simple_intrinsic("all_true", vec_ty), None),
                    (Quantifier::All, false) => (v128_intrinsic("any_true"), Some(quote! { ! })),
                };

                quote! {
                    #method_sig {
                        #negate #intrinsic(a.into())
                    }
                }
            }
            OpSig::LoadInterleaved {
                block_size,
                block_count,
            } => {
                assert_eq!(block_count, 4, "only count of 4 is currently supported");
                let elems_per_vec = block_size as usize / vec_ty.scalar_bits;

                // For WASM we need to simulate interleaving with shuffle, and we only have
                // access to 2, 4 and 16 lanes. So, for 64 u8's, we need to split and recombine
                // the vectors.
                let (i1, i2, i3, i4, shuffle_fn) = match vec_ty.scalar_bits {
                    8 => (
                        quote! { 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29 },
                        quote! { 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31 },
                        quote! { 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23 },
                        quote! { 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 },
                        quote! { u8x16_shuffle },
                    ),
                    16 => (
                        quote! { 0, 4, 8, 12, 1, 5, 9, 13 },
                        quote! { 2, 6, 10, 14, 3, 7, 11, 15 },
                        quote! { 0, 1, 2, 3,  8, 9, 10, 11 },
                        quote! { 4, 5, 6, 7, 12, 13, 14, 15 },
                        quote! { u16x8_shuffle },
                    ),
                    32 => (
                        quote! { 0, 4, 1, 5 },
                        quote! { 2, 6, 3, 7 },
                        quote! { 0, 1, 4, 5 },
                        quote! { 2, 3, 6, 7 },
                        quote! { u32x4_shuffle },
                    ),
                    _ => panic!("unsupported scalar_bits"),
                };

                let block_ty = vec_ty.block_ty();
                let block_ty_2x =
                    VecType::new(block_ty.scalar, block_ty.scalar_bits, block_ty.len * 2);

                let combine_method = generic_op_name("combine", &block_ty);
                let combine_method_2x = generic_op_name("combine", &block_ty_2x);

                let combine_code = quote! {
                    let combined_lower = self.#combine_method(out0.simd_into(self), out1.simd_into(self));
                    let combined_upper = self.#combine_method(out2.simd_into(self), out3.simd_into(self));
                    self.#combine_method_2x(combined_lower, combined_upper)
                };

                quote! {
                    #method_sig {
                            let v0: v128 = unsafe { v128_load(src[0 * #elems_per_vec..].as_ptr() as *const v128) };
                            let v1: v128 = unsafe { v128_load(src[1 * #elems_per_vec..].as_ptr() as *const v128) };
                            let v2: v128 = unsafe { v128_load(src[2 * #elems_per_vec..].as_ptr() as *const v128) };
                            let v3: v128 = unsafe { v128_load(src[3 * #elems_per_vec..].as_ptr() as *const v128) };

                            // InterleaveLowerLanes(v0, v2) and InterleaveLowerLanes(v1, v3)
                            let v01_lower = #shuffle_fn::<#i1>(v0, v1);
                            let v23_lower = #shuffle_fn::<#i1>(v2, v3);

                            // InterleaveUpperLanes(v0, v2) and InterleaveUpperLanes(v1, v3)
                            let v01_upper = #shuffle_fn::<#i2>(v0, v1);
                            let v23_upper = #shuffle_fn::<#i2>(v2, v3);

                            // Interleave lower and upper to get final result
                            let out0 = #shuffle_fn::<#i3>(v01_lower, v23_lower);
                            let out1 = #shuffle_fn::<#i4>(v01_lower, v23_lower);
                            let out2 = #shuffle_fn::<#i3>(v01_upper, v23_upper);
                            let out3 = #shuffle_fn::<#i4>(v01_upper, v23_upper);

                            #combine_code
                    }
                }
            }
            OpSig::StoreInterleaved {
                block_size,
                block_count,
            } => {
                assert_eq!(block_count, 4, "only count of 4 is currently supported");
                let elems_per_vec = block_size as usize / vec_ty.scalar_bits;

                let (lower_indices, upper_indices, shuffle_fn) = match vec_ty.scalar_bits {
                    8 => (
                        quote! { 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23 },
                        quote! { 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31 },
                        quote! { u8x16_shuffle },
                    ),
                    16 => (
                        quote! { 0, 8, 1, 9, 2, 10, 3, 11 },
                        quote! { 4, 12, 5, 13, 6, 14, 7, 15 },
                        quote! { u16x8_shuffle },
                    ),
                    32 => (
                        quote! { 0, 4, 1, 5 },
                        quote! { 2, 6, 3, 7 },
                        quote! { u32x4_shuffle },
                    ),
                    _ => panic!("unsupported scalar_bits"),
                };

                let block_ty = vec_ty.block_ty();
                let block_ty_2x =
                    VecType::new(block_ty.scalar, block_ty.scalar_bits, block_ty.len * 2);
                let block_ty_4x =
                    VecType::new(block_ty.scalar, block_ty.scalar_bits, block_ty.len * 4);

                let split_method = generic_op_name("split", &block_ty_2x);
                let split_method_2x = generic_op_name("split", &block_ty_4x);

                let split_code = quote! {
                    let (lower, upper) = self.#split_method_2x(a);
                    let (v0_vec, v1_vec) = self.#split_method(lower);
                    let (v2_vec, v3_vec) = self.#split_method(upper);

                    let v0: v128 = v0_vec.into();
                    let v1: v128 = v1_vec.into();
                    let v2: v128 = v2_vec.into();
                    let v3: v128 = v3_vec.into();
                };

                quote! {
                    #method_sig {
                        #split_code

                        // InterleaveLowerLanes(v0, v2) and InterleaveLowerLanes(v1, v3)
                        let v02_lower = #shuffle_fn::<#lower_indices>(v0, v2);
                        let v13_lower = #shuffle_fn::<#lower_indices>(v1, v3);

                        // InterleaveUpperLanes(v0, v2) and InterleaveUpperLanes(v1, v3)
                        let v02_upper = #shuffle_fn::<#upper_indices>(v0, v2);
                        let v13_upper = #shuffle_fn::<#upper_indices>(v1, v3);

                        // Interleave lower and upper to get final result
                        let out0 = #shuffle_fn::<#lower_indices>(v02_lower, v13_lower);
                        let out1 = #shuffle_fn::<#upper_indices>(v02_lower, v13_lower);
                        let out2 = #shuffle_fn::<#lower_indices>(v02_upper, v13_upper);
                        let out3 = #shuffle_fn::<#upper_indices>(v02_upper, v13_upper);

                        unsafe {
                            v128_store(dest[0 * #elems_per_vec..].as_mut_ptr() as *mut v128, out0);
                            v128_store(dest[1 * #elems_per_vec..].as_mut_ptr() as *mut v128, out1);
                            v128_store(dest[2 * #elems_per_vec..].as_mut_ptr() as *mut v128, out2);
                            v128_store(dest[3 * #elems_per_vec..].as_mut_ptr() as *mut v128, out3);
                        }
                    }
                }
            }
            OpSig::FromArray { kind } => {
                generic_from_array(method_sig, vec_ty, kind, self.max_block_size(), |_| {
                    v128_intrinsic("load")
                })
            }
            OpSig::AsArray { kind } => {
                generic_as_array(method_sig, vec_ty, kind, self.max_block_size(), |_| {
                    Ident::new("v128", Span::call_site())
                })
            }
            OpSig::ToArray => generic_to_array(method_sig, vec_ty, self.max_block_size(), |_| {
                v128_intrinsic("store")
            }),
            OpSig::FromBytes => generic_from_bytes(method_sig, vec_ty),
            OpSig::ToBytes => generic_to_bytes(method_sig, vec_ty),
        }
    }
}
