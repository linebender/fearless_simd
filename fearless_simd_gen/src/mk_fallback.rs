// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::fallback;
use crate::generic::{
    generic_mask_from_bitmask, generic_mask_set, generic_mask_to_bitmask, generic_op_name,
    integer_lane_mask_rotate, integer_lane_mask_splat_arg,
};
use crate::level::Level;
use crate::ops::{NarrowingMode, Op, OpSig, relaxed_narrow_method};
use crate::types::{ScalarType, VecType};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

#[derive(Clone, Copy)]
pub(crate) struct Fallback;

pub(crate) fn scalar_mul_add_precise_f32_body() -> TokenStream {
    quote! {
        // Every finite f32 product is exactly representable as f64. Recover the exact error
        // of the widened addition with TwoSum, then turn the rounded f64 sum into a
        // round-to-odd value before narrowing. This is the p=24, k=29 specialization of
        // Boldo and Melquiond's Theorem 3, which proves that this final narrowing is
        // equivalent to rounding the exact product-plus-add once to f32:
        // https://guillaume.melquiond.fr/doc/08-tc.pdf
        // The theorem's exponent-range condition also holds: every finite exact result is
        // a multiple of 2^-298 with magnitude below 2^256, well inside binary64's normal range.
        let product = (a as f64) * (b as f64);
        let c = c as f64;
        let mut sum = product + c;

        if sum.is_finite() {
            // Knuth's unconditional TwoSum; each arithmetic operation must round to binary64.
            let virtual_sum = sum - product;
            let residual =
                (product - (sum - virtual_sum)) + (c - virtual_sum);
            let sum_bits = sum.to_bits();

            if residual != 0.0 && sum_bits & 1 == 0 {
                let corrected_bits = if sum.is_sign_negative()
                    == residual.is_sign_negative()
                {
                    sum_bits.wrapping_add(1)
                } else {
                    sum_bits.wrapping_sub(1)
                };
                sum = f64::from_bits(corrected_bits);
            }
        }

        sum as f32
    }
}

pub(crate) fn scalar_mul_add_precise_f32_helper() -> TokenStream {
    let body = scalar_mul_add_precise_f32_body();
    quote! {
        #[inline(always)]
        fn scalar_mul_add_precise_f32(a: f32, b: f32, c: f32) -> f32 {
            #body
        }
    }
}

pub(crate) fn float_ext_prelude() -> TokenStream {
    quote! {
        #[cfg(all(feature = "libm", not(feature = "std")))]
        #[allow(dead_code, reason = "Generated backends use different subsets of these helpers")]
        trait FloatExt {
            fn floor(self) -> Self;
            fn ceil(self) -> Self;
            fn round_ties_even(self) -> Self;
            fn fract(self) -> Self;
            fn sqrt(self) -> Self;
            fn trunc(self) -> Self;
            fn mul_add(self, a: Self, b: Self) -> Self;
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
            #[inline(always)]
            fn mul_add(self, a: f32, b: f32) -> f32 {
                libm::fmaf(self, a, b)
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
            #[inline(always)]
            fn mul_add(self, a: f64, b: f64) -> f64 {
                libm::fma(self, a, b)
            }
        }
    }
}

fn count_bits_method(op: Op, vec_ty: &VecType) -> TokenStream {
    let count = match op.method {
        "count_ones" => quote! { count_ones },
        "count_zeros" => quote! { count_zeros },
        _ => unreachable!("count_bits_method only implements bit-counting operations"),
    };
    let method_sig = op.simd_trait_method_sig(vec_ty);
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let items = make_list(
        (0..vec_ty.len)
            .map(|idx| {
                let value = lane(quote! { a }, vec_ty, idx);
                match (vec_ty.scalar, vec_ty.scalar_bits) {
                    (ScalarType::Unsigned, 32) => quote! { #value.#count() },
                    (ScalarType::Int, 32) => quote! { #value.#count().cast_signed() },
                    (_, 64) => quote! { #scalar::from(#value.#count()) },
                    (_, 8 | 16) => {
                        quote! { #scalar::try_from(#value.#count()).unwrap() }
                    }
                    _ => unreachable!(),
                }
            })
            .collect::<Vec<_>>(),
    );

    quote! {
        #method_sig {
            #items.simd_into(self)
        }
    }
}

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

    fn enabled_target_features(&self) -> Option<&'static str> {
        None
    }

    fn arch_ty(&self, vec_ty: &VecType) -> TokenStream {
        let scalar_rust = vec_ty.scalar.rust(vec_ty.scalar_bits);
        let len = vec_ty.len;
        quote!([#scalar_rust; #len])
    }

    fn token_doc(&self) -> &'static str {
        r#"A token for scalar fallback SIMD, representing the "fallback" level."#
    }

    fn make_module_prelude(&self) -> TokenStream {
        let float_ext = float_ext_prelude();
        let scalar_mul_add_precise_f32 = scalar_mul_add_precise_f32_helper();

        quote! {
            use core::ops::*;

            #scalar_mul_add_precise_f32
            #float_ext
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
            /// Create a scalar fallback token.
            #[inline]
            pub const fn new() -> Self {
                Self { _private: () }
            }
        }
    }

    fn make_method(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let Op { sig, method, .. } = op;
        let method_sig = op.simd_trait_method_sig(vec_ty);

        match sig {
            OpSig::Splat => {
                let num_elements = vec_ty.len;
                let normalize_mask = integer_lane_mask_splat_arg(vec_ty);
                quote! {
                    #method_sig {
                        #normalize_mask
                        [val; #num_elements].simd_into(self)
                    }
                }
            }
            OpSig::Unary => {
                if method == "reverse" {
                    let items = make_list(
                        (0..vec_ty.len)
                            .rev()
                            .map(|idx| lane(quote! { a }, vec_ty, idx))
                            .collect::<Vec<_>>(),
                    );
                    return quote! {
                        #method_sig {
                            #items.simd_into(self)
                        }
                    };
                }

                if matches!(method, "count_ones" | "count_zeros") {
                    return count_bits_method(op, vec_ty);
                }

                if method == "approximate_recip" {
                    return quote! {
                        #method_sig {
                            1.0 / a
                        }
                    };
                }

                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            let args = [lane(quote! { a }, vec_ty, idx)];
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
            OpSig::Reduce { lane_op } => {
                if lane_op == "add" {
                    fallback_reduce_sum(method_sig, vec_ty)
                } else {
                    fallback_reduce_min_max(method_sig, vec_ty, lane_op)
                }
            }
            OpSig::RotateElements { .. } => integer_lane_mask_rotate(op, vec_ty),
            OpSig::Widen { target_ty } => {
                let scalar = target_ty.scalar.rust(target_ty.scalar_bits);
                let half_len = vec_ty.len / 2;
                let low = make_list(
                    (0..half_len)
                        .map(|idx| {
                            let a = lane(quote! { a }, vec_ty, idx);
                            quote! { #a as #scalar }
                        })
                        .collect::<Vec<_>>(),
                );
                let high = make_list(
                    (half_len..vec_ty.len)
                        .map(|idx| {
                            let a = lane(quote! { a }, vec_ty, idx);
                            quote! { #a as #scalar }
                        })
                        .collect::<Vec<_>>(),
                );
                quote! {
                    #method_sig {
                        (#low.simd_into(self), #high.simd_into(self))
                    }
                }
            }
            OpSig::Narrow { target_ty, mode } => {
                if mode == NarrowingMode::Relaxed {
                    return relaxed_narrow_method(op, vec_ty, target_ty, "narrow");
                }

                let scalar = target_ty.scalar.rust(target_ty.scalar_bits);
                let src_scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
                let convert = |value: TokenStream| {
                    if vec_ty.scalar == ScalarType::Float || mode == NarrowingMode::Wrap {
                        quote! { #value as #scalar }
                    } else {
                        quote! {
                            #value.clamp(#scalar::MIN as #src_scalar, #scalar::MAX as #src_scalar) as #scalar
                        }
                    }
                };
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| convert(lane(quote! { a }, vec_ty, idx)))
                        .chain((0..vec_ty.len).map(|idx| convert(lane(quote! { b }, vec_ty, idx))))
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
                            let b_lane = lane(quote! { b }, vec_ty, idx);
                            let b_lane = if matches!(method, "shlv" | "shrv") {
                                quote! { #b_lane as u32 }
                            } else {
                                b_lane
                            };
                            let b = if fallback::translate_op(
                                method,
                                vec_ty.scalar == ScalarType::Float,
                            )
                            .map(rhs_reference)
                            .unwrap_or(true)
                            {
                                quote! { &#b_lane }
                            } else {
                                b_lane
                            };

                            let args = [lane(quote! { a }, vec_ty, idx), quote! { #b }];
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
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            let args = [lane(quote! { a }, vec_ty, idx), quote! { shift }];
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
                } else if method == "mul_sub_precise" {
                    let mul_add_precise = generic_op_name("mul_add_precise", vec_ty);
                    quote! {
                        #method_sig {
                            self.#mul_add_precise(a, b, -c)
                        }
                    }
                } else if method == "mul_add_precise"
                    && vec_ty.scalar == ScalarType::Float
                    && vec_ty.scalar_bits == 32
                {
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                let a = lane(quote! { a }, vec_ty, idx);
                                let b = lane(quote! { b }, vec_ty, idx);
                                let c = lane(quote! { c }, vec_ty, idx);
                                quote! { scalar_mul_add_precise_f32(#a, #b, #c) }
                            })
                            .collect::<Vec<_>>(),
                    );
                    quote! {
                        #method_sig {
                            #items.simd_into(self)
                        }
                    }
                } else {
                    let items = make_list(
                        (0..vec_ty.len)
                            .map(|idx| {
                                let args = [
                                    lane(quote! { a }, vec_ty, idx),
                                    lane(quote! { b }, vec_ty, idx),
                                    lane(quote! { c }, vec_ty, idx),
                                ];
                                fallback::expr(method, vec_ty, &args)
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
            OpSig::Compare => {
                let mask_type = vec_ty.cast(ScalarType::Mask);
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx: usize| {
                            let a = lane(quote! { a }, vec_ty, idx);
                            let b = lane(quote! { b }, vec_ty, idx);
                            let args = [quote! { &#a }, quote! { &#b }];
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
                let mask_type = vec_ty.mask_ty();
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            let a = lane(quote! { a }, &mask_type, idx);
                            let b = lane(quote! { b }, vec_ty, idx);
                            let c = lane(quote! { c }, vec_ty, idx);
                            quote! { if #a != 0 { #b } else { #c } }
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
                            let a = lane(quote! { a }, vec_ty, idx);
                            let b = lane(quote! { b }, vec_ty, idx);
                            quote! { #a, #b }
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
                        .map(|idx| lane(quote! { a }, vec_ty, idx))
                        .chain(indices.map(|idx| lane(quote! { b }, vec_ty, idx)))
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #unzip.simd_into(self)
                    }
                }
            }
            OpSig::Slide { .. } => {
                let n = vec_ty.len;
                quote! {
                    #method_sig {
                        let mut dest = [Default::default(); #n];
                        dest[..#n - SHIFT].copy_from_slice(&a.val.0[SHIFT..]);
                        dest[#n - SHIFT..].copy_from_slice(&b.val.0[..SHIFT]);
                        dest.simd_into(self)
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
                let bytes_rust = bytes_ty.rust();
                let byte_count = bytes_ty.len;
                let items = make_list(
                    (0..byte_count)
                        .map(|idx| {
                            quote! {
                                {
                                    let index = indices[#idx] as usize;
                                    bytes[index % #byte_count]
                                }
                            }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        let bytes = Bytes::to_bytes(a);
                        let result: #bytes_rust<Self> = #items.simd_into(self);
                        Bytes::from_bytes(result)
                    }
                }
            }
            OpSig::SwizzleDyn => {
                let bytes_ty = vec_ty.bytes_ty();
                let bytes_rust = bytes_ty.rust();
                let byte_count = bytes_ty.len;

                quote! {
                    #method_sig {
                        let bytes = Bytes::to_bytes(a);
                        let mut output = [0u8; #byte_count];
                        for lane in 0..#byte_count {
                            let index = indices[lane] as usize;
                            output[lane] = bytes[index % #byte_count];
                        }
                        let result: #bytes_rust<Self> = output.simd_into(self);
                        Bytes::from_bytes(result)
                    }
                }
            }
            OpSig::SwizzleDynPrecise => {
                let bytes_ty = vec_ty.bytes_ty();
                let bytes_rust = bytes_ty.rust();
                let byte_count = bytes_ty.len;
                // This formulation lowers into one cmov per element on SSE2/SSE4.2
                // and autovectorizes on RISC-V.
                quote! {
                    #method_sig {
                        let bytes = Bytes::to_bytes(a);
                        let mut output = [0u8; #byte_count];
                        for lane in 0..#byte_count {
                            // Keep the load unconditionally in bounds so LLVM can always execute it,
                            // and select zero afterwards. This avoids a branch that could be mispredicted.
                            let index = indices[lane] as usize;
                            let value = bytes[index % #byte_count];
                            output[lane] = if index < #byte_count { value } else { 0 };
                        }
                        let result: #bytes_rust<Self> = output.simd_into(self);
                        Bytes::from_bytes(result)
                    }
                }
            }
            OpSig::Cvt {
                target_ty,
                scalar_bits,
                precise: _,
            } => {
                let to_ty = vec_ty.reinterpret(target_ty, scalar_bits);
                let scalar = to_ty.scalar.rust(scalar_bits);
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            let a = lane(quote! { a }, vec_ty, idx);
                            quote! { #a as #scalar }
                        })
                        .collect::<Vec<_>>(),
                );
                quote! {
                    #method_sig {
                        #items.simd_into(self)
                    }
                }
            }
            OpSig::MaskReduce {
                quantifier,
                condition,
            } => {
                let check = if condition {
                    quote! { != }
                } else {
                    quote! { == }
                };

                let expr = match quantifier {
                    crate::ops::Quantifier::Any => {
                        let lanes = (0..vec_ty.len).map(|idx| lane(quote! { a }, vec_ty, idx));
                        quote! { #(#lanes #check 0)||* }
                    }
                    crate::ops::Quantifier::All => {
                        let lanes = (0..vec_ty.len).map(|idx| lane(quote! { a }, vec_ty, idx));
                        quote! { #(#lanes #check 0)&&* }
                    }
                };

                quote! {
                    #method_sig {
                        #expr
                    }
                }
            }
            OpSig::MaskFromBitmask => generic_mask_from_bitmask(method_sig, vec_ty),
            OpSig::MaskToBitmask => generic_mask_to_bitmask(method_sig, vec_ty),
            OpSig::MaskSet => generic_mask_set(method_sig, vec_ty),
            OpSig::LoadInterleaved {
                block_size,
                block_count,
            } => {
                let block_count = block_count as usize;
                let elems_per_vec = block_size as usize / vec_ty.scalar_bits;
                let vectors = (0..block_count).map(|channel| {
                    let items = make_list(
                        (0..elems_per_vec)
                            .map(|lane| {
                                let idx = lane * block_count + channel;
                                quote! { src[#idx] }
                            })
                            .collect(),
                    );
                    quote! { #items.simd_into(self) }
                });

                quote! {
                    #method_sig {
                        [#(#vectors),*]
                    }
                }
            }
            OpSig::StoreInterleaved {
                block_size,
                block_count,
            } => {
                let block_count = block_count as usize;
                let elems_per_vec = block_size as usize / vec_ty.scalar_bits;
                let items = make_list(
                    (0..elems_per_vec)
                        .flat_map(|lane| {
                            (0..block_count).map(move |channel| {
                                quote! { vectors[#channel][#lane] }
                            })
                        })
                        .collect(),
                );

                quote! {
                    #method_sig {
                        *dest = #items;
                    }
                }
            }
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

    fn make_type_impl(&self) -> TokenStream {
        TokenStream::new()
    }
}

fn lane(value: TokenStream, vec_ty: &VecType, idx: usize) -> TokenStream {
    if vec_ty.scalar == ScalarType::Mask {
        quote! { #value.val.0[#idx] }
    } else {
        quote! { #value[#idx] }
    }
}

/// Build an adjacent balanced min/max reduction one horizontal level at a time.
fn fallback_reduce_min_max(
    method_sig: TokenStream,
    vec_ty: &VecType,
    lane_op: &str,
) -> TokenStream {
    assert_eq!(
        vec_ty.n_bits(),
        128,
        "wide reductions must use the generic 128-bit-grained implementation"
    );

    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let mut statements = Vec::new();
    let mut previous = quote! { a };
    let mut previous_len = vec_ty.len;

    while previous_len > 1 {
        let next_len = previous_len / 2;
        let results = (0..next_len).map(|index| {
            let left_index = index * 2;
            let right_index = left_index + 1;
            let args = [
                quote! { #previous[#left_index] },
                quote! { #previous[#right_index] },
            ];
            fallback::expr(lane_op, vec_ty, &args)
        });
        statements.push(quote! {
            let reduced: [#scalar; #next_len] = [#(#results),*];
        });
        previous = quote! { reduced };
        previous_len = next_len;
    }

    quote! {
        #method_sig {
            #(#statements)*
            #previous[0]
        }
    }
}

/// Build an adjacent balanced reduction one horizontal level at a time.
fn fallback_reduce_sum(method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
    // The structure directly mirrors the SIMD constructions for two reasons:
    // 1. We promise the same output across all platforms, which means
    //    we have to perform additions in the same SIMD-friendly order,
    //    because floating-point addition is not associative.
    // 2. Expressing individual stages as arrays allows for autovectorization
    //    on platforms we don't have explicit intrinsics for.
    assert_eq!(
        vec_ty.n_bits(),
        128,
        "wide reductions must use the generic 128-bit-grained implementation"
    );

    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let mut statements = Vec::new();
    let mut previous = quote! { a };
    let mut previous_len = vec_ty.len;
    let mut level = 0;

    while previous_len > 1 {
        let name = format_ident!("sum_level_{level}");
        let next_len = previous_len / 2;
        let additions = (0..next_len).map(|index| {
            let left_index = index * 2;
            let right_index = left_index + 1;
            if vec_ty.scalar == ScalarType::Float {
                quote! { #previous[#left_index] + #previous[#right_index] }
            } else {
                quote! { #previous[#left_index].wrapping_add(#previous[#right_index]) }
            }
        });
        statements.push(quote! {
            let #name: [#scalar; #next_len] = [#(#additions),*];
        });
        previous = quote! { #name };
        previous_len = next_len;
        level += 1;
    }

    quote! {
        #method_sig {
            #(#statements)*
            #previous[0]
        }
    }
}

/// Whether the second argument of the function needs to be passed by reference.
fn rhs_reference(method: &str) -> bool {
    !matches!(
        method,
        "copysign"
            | "min"
            | "max"
            | "wrapping_sub"
            | "wrapping_mul"
            | "wrapping_add"
            | "saturating_add"
            | "saturating_sub"
            | "wrapping_shl"
            | "wrapping_shr"
    )
}

fn make_list(items: Vec<TokenStream>) -> TokenStream {
    quote!([#( #items, )*])
}
