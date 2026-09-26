// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{ToTokens, quote};

use crate::{
    level::Level,
    ops::{ElementDirection, Op, OpSig, SlideGranularity},
    types::{ScalarType, VecType},
};

pub(crate) fn generic_op_name(op: &str, ty: &VecType) -> Ident {
    Ident::new(&format!("{op}_{}", ty.rust_name()), Span::call_site())
}

pub(crate) fn fallback_method(op: Op, vec_ty: &VecType) -> TokenStream {
    crate::mk_fallback::Fallback.make_method(op, vec_ty)
}

/// Implement `count_zeros` in terms of `count_ones`, matching the scalar and
/// portable-SIMD formulations.
pub(crate) fn count_zeros_method(op: Op, vec_ty: &VecType) -> TokenStream {
    assert_eq!(
        op.method, "count_zeros",
        "count_zeros_method only implements count_zeros"
    );
    let method_sig = op.simd_trait_method_sig(vec_ty);
    let not = generic_op_name("not", vec_ty);
    let count_ones = generic_op_name("count_ones", vec_ty);
    quote! {
        #method_sig {
            self.#count_ones(self.#not(a))
        }
    }
}

/// Implement a typed byte swizzle by forwarding to the corresponding byte-vector operation.
pub(crate) fn byte_swizzle_op(op: &Op, vec_ty: &VecType) -> TokenStream {
    assert!(
        op.sig.should_route_swizzle_through_bytes(vec_ty),
        "what are we even doing here?"
    );

    let method_sig = op.simd_trait_method_sig(vec_ty);
    let byte_method = generic_op_name(op.method, &vec_ty.bytes_ty());
    match op.sig {
        OpSig::SwizzleDynWithinBlocks | OpSig::SwizzleDyn | OpSig::SwizzleDynPrecise => {
            quote! {
                #method_sig {
                    Bytes::from_bytes(self.#byte_method(Bytes::to_bytes(a), indices))
                }
            }
        }
        OpSig::ConcatSwizzleDyn | OpSig::ConcatSwizzleDynPrecise => {
            quote! {
                #method_sig {
                    Bytes::from_bytes(self.#byte_method(
                        Bytes::to_bytes(a),
                        Bytes::to_bytes(b),
                        indices,
                    ))
                }
            }
        }
        _ => unreachable!("non-swizzle operation routed through bytes"),
    }
}

/// Implement a greater-than comparison by reversing the corresponding less-than comparison.
pub(crate) fn reversed_compare_op(op: &Op, vec_ty: &VecType) -> Option<TokenStream> {
    let reversed_method = generic_op_name(op.reversed_compare_method()?, vec_ty);
    let method_sig = op.simd_trait_method_sig(vec_ty);
    Some(quote! {
        #method_sig {
            self.#reversed_method(b, a)
        }
    })
}

/// Implement a fixed lane reversal as a byte swizzle with compile-time-known indices.
///
/// Backends use this only at widths they handle natively. Emulated wider vectors recurse through
/// [`generic_op`], which also reverses the order of their halves.
pub(crate) fn reverse_method(op: Op, vec_ty: &VecType) -> TokenStream {
    assert_eq!(op.method, "reverse");
    assert!(matches!(op.sig, OpSig::Unary));
    assert_ne!(
        vec_ty.scalar,
        ScalarType::Mask,
        "reverse_method requires byte-swizzlable non-mask vectors"
    );

    let method_sig = op.simd_trait_method_sig(vec_ty);
    let bytes_ty = vec_ty.bytes_ty();
    let bytes = bytes_ty.rust();
    let swizzle = generic_op_name("swizzle_dyn", vec_ty);
    let element_bytes = vec_ty.scalar_bits / 8;
    let indices = (0..bytes_ty.len).map(|output_byte| {
        let output_element = output_byte / element_bytes;
        let byte_in_element = output_byte % element_bytes;
        let input_byte = (vec_ty.len - 1 - output_element) * element_bytes + byte_in_element;
        Literal::u8_unsuffixed(u8::try_from(input_byte).unwrap())
    });

    quote! {
        #method_sig {
            let indices: #bytes<Self> = [#(#indices),*].simd_into(self);
            self.#swizzle(a, indices)
        }
    }
}

/// Reverse a native vector-backed mask by reusing the corresponding signed-vector operation.
///
/// Compact predicate masks such as AVX-512 masks must use a representation-specific lowering
/// instead.
pub(crate) fn reverse_vector_mask_method(op: Op, vec_ty: &VecType) -> TokenStream {
    assert_eq!(op.method, "reverse");
    assert!(matches!(op.sig, OpSig::Unary));
    assert_eq!(
        vec_ty.scalar,
        ScalarType::Mask,
        "reverse_vector_mask_method only implements masks"
    );

    let method_sig = op.simd_trait_method_sig(vec_ty);
    let int_ty = vec_ty.cast(ScalarType::Int);
    let int = int_ty.rust();
    let mask = vec_ty.rust();
    let reverse_int = generic_op_name("reverse", &int_ty);

    quote! {
        #method_sig {
            let lanes = #int {
                val: crate::transmute::checked_transmute_copy(&a.val),
                simd: self,
            };
            let reversed = self.#reverse_int(lanes);
            #mask {
                val: crate::transmute::checked_transmute_copy(&reversed.val),
                simd: self,
            }
        }
    }
}

pub(crate) fn recursive_swizzle_dyn_precise_body<T: ToTokens + ?Sized>(
    vec_ty: &VecType,
    token: &T,
) -> TokenStream {
    // We can take advantage of the "precise" property (out-of-range is 0)
    // to assemble a double-vector-width arbitrary shuffle.
    // The trick is to split the input into two, then run for each half of output indices J:
    // ```
    // from_low  = swizzle_H(low_table,  J);
    // from_high = swizzle_H(high_table, J.wrapping_sub(H));
    // result    = from_low | from_high;
    // ```
    // Since each element is out-of-range for at least one half,
    // the final combine is very cheap: a bitwise or.
    let bytes_ty = vec_ty.bytes_ty();
    let half_bytes_ty = VecType::new(ScalarType::Unsigned, 8, bytes_ty.len / 2);
    let split_bytes = generic_op_name("split", &bytes_ty);
    let combine_half_bytes = generic_op_name("combine", &half_bytes_ty);
    let swizzle_half = generic_op_name("swizzle_dyn_precise", &half_bytes_ty);
    let splat_half = generic_op_name("splat", &half_bytes_ty);
    let sub_half = generic_op_name("sub", &half_bytes_ty);
    let or_half = generic_op_name("or", &half_bytes_ty);
    let half_len = Literal::u8_unsuffixed(u8::try_from(bytes_ty.len / 2).unwrap());

    quote! {
        let bytes = Bytes::to_bytes(a);
        let (table_low, table_high) = #token.#split_bytes(bytes);
        let (indices_low, indices_high) = #token.#split_bytes(indices);
        let high_table_offset = #token.#splat_half(#half_len);

        let output_low_from_low = #token.#swizzle_half(table_low, indices_low);
        let output_low_from_high = #token.#swizzle_half(
            table_high,
            #token.#sub_half(indices_low, high_table_offset),
        );
        let output_low = #token.#or_half(output_low_from_low, output_low_from_high);

        let output_high_from_low = #token.#swizzle_half(table_low, indices_high);
        let output_high_from_high = #token.#swizzle_half(
            table_high,
            #token.#sub_half(indices_high, high_table_offset),
        );
        let output_high = #token.#or_half(output_high_from_low, output_high_from_high);

        let result_bytes = #token.#combine_half_bytes(output_low, output_high);
    }
}

/// Implement a precise swizzle from a concatenated pair of vectors by combining two zeroing
/// whole-vector swizzles with bitwise OR.
pub(crate) fn concat_swizzle_dyn_precise_body<T: ToTokens + ?Sized>(
    vec_ty: &VecType,
    token: &T,
) -> TokenStream {
    let bytes_ty = vec_ty.bytes_ty();
    let swizzle = generic_op_name("swizzle_dyn_precise", &bytes_ty);
    let splat = generic_op_name("splat", &bytes_ty);
    let sub = generic_op_name("sub", &bytes_ty);
    let or = generic_op_name("or", &bytes_ty);
    let second_table_offset = Literal::u8_unsuffixed(u8::try_from(bytes_ty.len).unwrap());

    quote! {
        let first_table = Bytes::to_bytes(a);
        let second_table = Bytes::to_bytes(b);
        let second_table_offset = #token.#splat(#second_table_offset);
        let from_first = #token.#swizzle(first_table, indices);
        let from_second = #token.#swizzle(
            second_table,
            #token.#sub(indices, second_table_offset),
        );
        let result_bytes = #token.#or(from_first, from_second);
    }
}

/// For backends that store masks as all-zero/all-one integer lanes, convert the public
/// `bool` mask splat argument into the backend's lane representation.
pub(crate) fn integer_lane_mask_splat_arg(vec_ty: &VecType) -> TokenStream {
    if vec_ty.scalar != ScalarType::Mask {
        return TokenStream::new();
    }

    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    quote! {
        let val: #scalar = if val { !0 } else { 0 };
    }
}

/// Rotate an integer-lane mask by bitcasting its storage to the corresponding
/// signed vector and forwarding to `SimdBase`'s element rotation.
///
/// This is only valid for backends whose masks use full integer lanes. Compact
/// predicate masks, such as AVX-512 masks, need their own implementation.
pub(crate) fn integer_lane_mask_rotate(op: Op, vec_ty: &VecType) -> TokenStream {
    assert_eq!(
        vec_ty.scalar,
        ScalarType::Mask,
        "mask element rotation only operates on masks"
    );
    let direction = match op.sig {
        OpSig::RotateElements { direction } => direction,
        _ => panic!("integer_lane_mask_rotate only implements mask element rotation"),
    };
    let method_sig = op.simd_trait_method_sig(vec_ty);
    let rotate = match direction {
        ElementDirection::Left => quote! { rotate_elements_left },
        ElementDirection::Right => quote! { rotate_elements_right },
    };
    let int_ty = vec_ty.cast(ScalarType::Int).rust();
    let mask_ty = vec_ty.rust();

    quote! {
        #method_sig {
            let int = #int_ty {
                val: crate::transmute::checked_transmute_copy(&a.val),
                simd: self,
            };
            let rotated = int.#rotate::<OFFSET>();
            #mask_ty {
                val: crate::transmute::checked_transmute_copy(&rotated.val),
                simd: self,
            }
        }
    }
}

/// Generic operation implementations.
///
/// Most operations are implemented using split/combine, while some forward to
/// another operation with compatible semantics.
pub(crate) fn generic_op(op: &Op, ty: &VecType) -> TokenStream {
    let split = generic_op_name("split", ty);
    let half = VecType::new(ty.scalar, ty.scalar_bits, ty.len / 2);
    let combine = generic_op_name("combine", &half);
    let do_half = generic_op_name(op.method, &half);
    let method_sig = op.simd_trait_method_sig(ty);
    match op.sig {
        OpSig::Splat => {
            quote! {
                #method_sig {
                    let half = self.#do_half(val);
                    self.#combine(half, half)
                }
            }
        }
        OpSig::Unary => {
            let halves = if op.method == "reverse" {
                quote! { self.#do_half(a1), self.#do_half(a0) }
            } else {
                quote! { self.#do_half(a0), self.#do_half(a1) }
            };
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(#halves)
                }
            }
        }
        OpSig::Reduce { lane_op } => {
            let combine_halves = generic_op_name(lane_op, &half);
            // Combine corresponding lanes before reducing so vectors wider than 128 bits
            // retain a fixed-depth tree while needing only one horizontal 128-bit leaf.
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#do_half(self.#combine_halves(a0, a1))
                }
            }
        }
        OpSig::RotateElements { .. } => {
            panic!("mask element rotation must operate on the full mask")
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
        OpSig::SwizzleDynWithinBlocks => {
            let bytes_ty = ty.bytes_ty();
            let split_indices = generic_op_name("split", &bytes_ty);
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (indices0, indices1) = self.#split_indices(indices);
                    self.#combine(
                        self.#do_half(a0, indices0),
                        self.#do_half(a1, indices1),
                    )
                }
            }
        }
        OpSig::SwizzleDyn
        | OpSig::SwizzleDynPrecise
        | OpSig::ConcatSwizzleDyn
        | OpSig::ConcatSwizzleDynPrecise => {
            panic!("whole-vector swizzles cannot be done via split/combine");
        }
        OpSig::Compress { merge: false } => {
            let merge_method = generic_op_name("compress_merge", ty);
            let ty = ty.rust();
            quote! {
                #method_sig {
                    self.#merge_method(values, mask, #ty::splat(self, 0))
                }
            }
        }
        OpSig::Compress { merge: true } => {
            let len = Literal::usize_unsuffixed(ty.len);
            let to_bitmask = generic_op_name("to_bitmask", &ty.mask_ty());
            quote! {
                #method_sig {
                    // Branchless: every lane is written at the cursor, which only advances on
                    // selected lanes. The one slot past the selected lanes may be clobbered by an
                    // unselected lane, so it is restored from `merge` afterwards.
                    let mask = self.#to_bitmask(mask);
                    let mut merge_padded = [0u8; #len + 1];
                    merge_padded[..#len].copy_from_slice(&*merge);
                    let mut compacted = merge_padded;
                    let mut output_lane = 0;
                    for input_lane in 0..#len {
                        compacted[output_lane] = values[input_lane];
                        output_lane += ((mask >> input_lane) & 1) as usize;
                    }
                    compacted[output_lane] = merge_padded[output_lane];
                    let mut result = merge;
                    result.copy_from_slice(&compacted[..#len]);
                    result
                }
            }
        }
        OpSig::Expand { merge: false } => {
            let merge_method = generic_op_name("expand_merge", ty);
            let ty = ty.rust();
            quote! {
                #method_sig {
                    self.#merge_method(values, mask, #ty::splat(self, 0))
                }
            }
        }
        OpSig::Expand { merge: true } => {
            let len = Literal::usize_unsuffixed(ty.len);
            let to_bitmask = generic_op_name("to_bitmask", &ty.mask_ty());
            quote! {
                #method_sig {
                    // Branchless: the input cursor never passes the output lane, so the read is
                    // always in bounds and selected with a lane mask instead of a branch.
                    let mask = self.#to_bitmask(mask);
                    let mut result = merge;
                    let mut input_lane = 0;
                    for output_lane in 0..#len {
                        let bit = ((mask >> output_lane) & 1) as u8;
                        let keep = 0u8.wrapping_sub(bit);
                        result[output_lane] =
                            (values[input_lane] & keep) | (result[output_lane] & !keep);
                        input_lane += usize::from(bit);
                    }
                    result
                }
            }
        }
        OpSig::LoadExpand { merge: false } => {
            let merge_method = generic_op_name("load_expand_merge", ty);
            let ty = ty.rust();
            quote! {
                #method_sig {
                    self.#merge_method(source, mask, #ty::splat(self, 0))
                }
            }
        }
        OpSig::LoadExpand { merge: true } => {
            let len = Literal::usize_unsuffixed(ty.len);
            let to_bitmask = generic_op_name("to_bitmask", &ty.mask_ty());
            quote! {
                #method_sig {
                    // Branchless like expand: the source cursor never passes the output lane.
                    let mask = self.#to_bitmask(mask);
                    let mut result = merge;
                    let mut source_index = 0;
                    for output_lane in 0..#len {
                        let bit = ((mask >> output_lane) & 1) as u8;
                        let keep = 0u8.wrapping_sub(bit);
                        result[output_lane] =
                            (source[source_index] & keep) | (result[output_lane] & !keep);
                        source_index += usize::from(bit);
                    }
                    result
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
            let combine_mask = generic_op_name("combine", &half_mask);
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    self.#combine_mask(self.#do_half(a0, b0), self.#do_half(a1, b1))
                }
            }
        }
        OpSig::Select => {
            let mask_ty = ty.cast(ScalarType::Mask);
            let split_mask = generic_op_name("split", &mask_ty);
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

            let zip_low_half = generic_op_name("zip_low", &half);
            let zip_high_half = generic_op_name("zip_high", &half);

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
            ..
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
        OpSig::Widen { target_ty } => {
            let mut target_half = target_ty;
            target_half.len /= 2;
            let combine_target = generic_op_name("combine", &target_half);
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (a00, a01) = self.#do_half(a0);
                    let (a10, a11) = self.#do_half(a1);
                    (
                        self.#combine_target(a00, a01),
                        self.#combine_target(a10, a11),
                    )
                }
            }
        }
        OpSig::Narrow { target_ty, .. } => {
            let mut target_half = target_ty;
            target_half.len /= 2;
            let combine_target = generic_op_name("combine", &target_half);
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    self.#combine_target(self.#do_half(a0, a1), self.#do_half(b0, b1))
                }
            }
        }
        OpSig::MaskReduce {
            quantifier,
            condition,
        } => {
            // Combining the halves element-wise first means only one horizontal reduction
            // (the expensive part) is needed, and avoids the branch a short-circuiting
            // `||`/`&&` would allow.
            let combine_halves = generic_op_name(quantifier.mask_combine_op(condition), &half);
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#do_half(self.#combine_halves(a0, a1))
                }
            }
        }
        OpSig::MaskFromBitmask => {
            let half_len = half.len;
            quote! {
                #method_sig {
                    let lo = self.#do_half(bits);
                    let hi = self.#do_half(bits >> #half_len);
                    self.#combine(lo, hi)
                }
            }
        }
        OpSig::MaskToBitmask => {
            let half_len = half.len;
            quote! {
                #method_sig {
                    let (lo, hi) = self.#split(a);
                    let lo = self.#do_half(lo);
                    let hi = self.#do_half(hi);
                    lo | (hi << #half_len)
                }
            }
        }
        OpSig::MaskSet => {
            panic!("Mask set must operate on the full mask vector")
        }
        OpSig::LoadInterleaved { .. } | OpSig::StoreInterleaved { .. } => {
            panic!("Interleaved memory operations must operate on full 128-bit vectors")
        }
        OpSig::Split { .. } | OpSig::Combine { .. } => {
            panic!("These operations require more information about the target platform");
        }
        OpSig::Interleave => {
            // interleave(a, b) = (zip_low(a, b), zip_high(a, b))
            // For wider vectors, we split each input, interleave the halves separately,
            // then combine the low parts and high parts.
            let zip_low_half = generic_op_name("zip_low", &half);
            let zip_high_half = generic_op_name("zip_high", &half);
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);

                    let lo_lo = self.#zip_low_half(a0, b0);
                    let lo_hi = self.#zip_high_half(a0, b0);

                    let hi_lo = self.#zip_low_half(a1, b1);
                    let hi_hi = self.#zip_high_half(a1, b1);

                    (self.#combine(lo_lo, lo_hi), self.#combine(hi_lo, hi_hi))
                }
            }
        }
        OpSig::Deinterleave => {
            // deinterleave(a, b) = (unzip_low(a, b), unzip_high(a, b))
            // For wider vectors, we split each input, deinterleave the halves separately,
            // then combine the even parts and odd parts.
            let unzip_low_half = generic_op_name("unzip_low", &half);
            let unzip_high_half = generic_op_name("unzip_high", &half);
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);

                    let lo_even = self.#unzip_low_half(a0, a1);
                    let lo_odd = self.#unzip_high_half(a0, a1);

                    let hi_even = self.#unzip_low_half(b0, b1);
                    let hi_odd = self.#unzip_high_half(b0, b1);

                    (self.#combine(lo_even, hi_even), self.#combine(lo_odd, hi_odd))
                }
            }
        }
        OpSig::Slide { granularity, .. } => {
            match (granularity, ty.n_bits()) {
                (SlideGranularity::WithinBlocks, 128) => {
                    // If this operation is done on a 128-bit vector type, the "within blocks" method is identical to the
                    // non-within-blocks one, so just defer to that.
                    let non_blockwise = generic_op_name("slide", ty);
                    quote! {
                        #method_sig {
                            self.#non_blockwise::<SHIFT>(a, b)
                        }
                    }
                }
                (SlideGranularity::WithinBlocks, _) => {
                    quote! {
                        #method_sig {
                            let (a0, a1) = self.#split(a);
                            let (b0, b1) = self.#split(b);
                            self.#combine(self.#do_half::<SHIFT>(a0, b0), self.#do_half::<SHIFT>(a1, b1))
                        }
                    }
                }
                _ => {
                    panic!("Item-wise shifts across blocks cannot be done via split/combine");
                }
            }
        }
    }
}

pub(crate) type CompactMergeSwizzle =
    fn(&VecType, &TokenStream, &TokenStream, &TokenStream) -> TokenStream;

/// Backend capabilities used by [`composed_compact_op`].
pub(crate) struct CompactOptions {
    /// Compact wide vectors a native 128-bit block at a time and splice them through memory.
    pub(crate) splice_wide_vectors: bool,
    /// Use the target's scalar population-count instruction instead of the byte-count table.
    pub(crate) hardware_popcount: bool,
    /// Optional backend operation that combines an out-of-range-zeroing shuffle with its merge
    /// operand. The arguments are `(vector type, values, control, merge)`.
    pub(crate) merge_swizzle: Option<CompactMergeSwizzle>,
}

/// Implement byte compression and expansion in terms of a backend's optimized whole-vector
/// swizzle, compact mask conversion, and selection operations.
pub(crate) fn composed_compact_op(op: Op, ty: &VecType, options: CompactOptions) -> TokenStream {
    assert_eq!(
        ty.scalar,
        ScalarType::Unsigned,
        "compact operations require unsigned vectors"
    );
    assert_eq!(ty.scalar_bits, 8, "compact operations require byte lanes");

    let method_sig = op.simd_trait_method_sig(ty);
    let vec = ty.rust();
    let len = Literal::usize_unsuffixed(ty.len);
    let swizzle = generic_op_name("swizzle_dyn_precise", ty);
    let to_bitmask = generic_op_name("to_bitmask", &ty.mask_ty());
    let from_bitmask = generic_op_name("from_bitmask", &ty.mask_ty());
    let select = generic_op_name("select", ty);
    let count_8 = |mask: TokenStream| {
        if options.hardware_popcount {
            quote! { #mask.count_ones() as usize }
        } else {
            quote! { crate::support::COMPACT_8_COUNTS[#mask] as usize }
        }
    };

    // Backends whose byte shuffle is confined to 128-bit blocks can avoid an expensive emulated
    // whole-vector shuffle by compacting each native block and splicing through a stack buffer.
    // This is the same broad strategy used by Highway on targets without native byte
    // compress/expand instructions.
    if options.splice_wide_vectors && ty.len > 16 {
        assert!(
            options.merge_swizzle.is_none(),
            "block-spliced compact operations do not support a merging shuffle"
        );
        let block_vec = VecType {
            scalar: ScalarType::Unsigned,
            scalar_bits: 8,
            len: 16,
        };
        let block_swizzle = generic_op_name("swizzle_dyn_precise", &block_vec);

        return match op.sig {
            OpSig::Compress { merge } => {
                let low_count = count_8(quote! { low_mask });
                let high_count = count_8(quote! { high_mask });
                let finish = if merge {
                    quote! {
                        let compressed = #vec::simd_from(self, output);
                        let prefix_bits = if output_lane == 64 {
                            u64::MAX
                        } else {
                            (1u64 << output_lane) - 1
                        };
                        let prefix_mask = self.#from_bitmask(prefix_bits);
                        self.#select(prefix_mask, compressed, merge)
                    }
                } else {
                    quote! { #vec::simd_from(self, output) }
                };
                quote! {
                    #method_sig {
                        let mask_bits = self.#to_bitmask(mask);
                        let mut output = [0u8; #len];
                        let mut output_lane = 0;
                        for block in 0..#len / 16 {
                            let block_bits = ((mask_bits >> (block * 16)) & 0xffff) as usize;
                            let low_mask = block_bits & 0xff;
                            let high_mask = block_bits >> 8;
                            let low_count = #low_count;
                            let low = crate::support::COMPRESS_8_CONTROLS[low_mask];
                            let high = crate::support::COMPRESS_8_CONTROLS[high_mask];
                            let high = ((high & 0x7f7f_7f7f_7f7f_7f7f)
                                + 0x0808_0808_0808_0808)
                                | (high & 0x8080_8080_8080_8080);
                            let mut control = [u8::MAX; 16];
                            control[..8].copy_from_slice(&low.to_le_bytes());
                            control[low_count..low_count + 8]
                                .copy_from_slice(&high.to_le_bytes());
                            let input = u8x16::from_slice(
                                self,
                                &values.as_array()[block * 16..block * 16 + 16],
                            );
                            let control = u8x16::simd_from(self, control);
                            let compacted: [u8; 16] =
                                self.#block_swizzle(input, control).into();
                            let write_len = core::cmp::min(16, #len - output_lane);
                            output[output_lane..output_lane + write_len]
                                .copy_from_slice(&compacted[..write_len]);
                            output_lane += low_count + #high_count;
                        }
                        #finish
                    }
                }
            }
            OpSig::Expand { merge } => {
                let low_count = count_8(quote! { low_mask });
                let high_count = count_8(quote! { high_mask });
                let finish = if merge {
                    quote! {
                        let expanded = #vec::simd_from(self, output);
                        self.#select(mask, expanded, merge)
                    }
                } else {
                    quote! { #vec::simd_from(self, output) }
                };
                quote! {
                    #method_sig {
                        let mask_bits = self.#to_bitmask(mask);
                        let values: [u8; #len] = values.into();
                        let mut output = [0u8; #len];
                        let mut input_lane = 0;
                        for block in 0..#len / 16 {
                            let block_bits = ((mask_bits >> (block * 16)) & 0xffff) as usize;
                            let low_mask = block_bits & 0xff;
                            let high_mask = block_bits >> 8;
                            let low_count = #low_count;
                            let low = crate::support::EXPAND_8_CONTROLS[low_mask];
                            let high = crate::support::EXPAND_8_CONTROLS[high_mask];
                            let high_base = low_count as u64 * 0x0101_0101_0101_0101;
                            let high = ((high & 0x7f7f_7f7f_7f7f_7f7f) + high_base)
                                | (high & 0x8080_8080_8080_8080);
                            let mut control = [0u8; 16];
                            control[..8].copy_from_slice(&low.to_le_bytes());
                            control[8..].copy_from_slice(&high.to_le_bytes());
                            let mut input = [0u8; 16];
                            let read_len = core::cmp::min(16, #len - input_lane);
                            input[..read_len]
                                .copy_from_slice(&values[input_lane..input_lane + read_len]);
                            let input = u8x16::simd_from(self, input);
                            let control = u8x16::simd_from(self, control);
                            let expanded: [u8; 16] =
                                self.#block_swizzle(input, control).into();
                            output[block * 16..block * 16 + 16].copy_from_slice(&expanded);
                            input_lane += low_count + #high_count;
                        }
                        #finish
                    }
                }
            }
            OpSig::LoadExpand { merge } => {
                let expand = generic_op_name(if merge { "expand_merge" } else { "expand" }, ty);
                let call = if merge {
                    quote! { self.#expand(values, mask, merge) }
                } else {
                    quote! { self.#expand(values, mask) }
                };
                quote! {
                    #method_sig {
                        let values = #vec::simd_from(self, *source);
                        #call
                    }
                }
            }
            _ => unreachable!("composed_compact_op only implements compact byte operations"),
        };
    }

    match op.sig {
        OpSig::Compress { merge } => {
            let block_count = count_8(quote! { block_mask });
            let finish = if merge {
                if let Some(merge_swizzle) = options.merge_swizzle {
                    merge_swizzle(
                        ty,
                        &quote! { values },
                        &quote! { control },
                        &quote! { merge },
                    )
                } else {
                    quote! {
                        let compressed = self.#swizzle(values, control);
                        let prefix_bits = if output_lane == 64 {
                            u64::MAX
                        } else {
                            (1u64 << output_lane) - 1
                        };
                        let prefix_mask = self.#from_bitmask(prefix_bits);
                        self.#select(prefix_mask, compressed, merge)
                    }
                }
            } else {
                quote! { self.#swizzle(values, control) }
            };
            quote! {
                #method_sig {
                    let mask_bits = self.#to_bitmask(mask);
                    let mut control = [u8::MAX; #len];
                    let mut output_lane = 0;
                    for block in 0..#len / 8 {
                        let block_mask = ((mask_bits >> (block * 8)) & 0xff) as usize;
                        let packed = crate::support::COMPRESS_8_CONTROLS[block_mask];
                        let base = (block * 8) as u64 * 0x0101_0101_0101_0101;
                        let adjusted = ((packed & 0x7f7f_7f7f_7f7f_7f7f) + base)
                            | (packed & 0x8080_8080_8080_8080);
                        let adjusted = adjusted.to_le_bytes();
                        let write_len = core::cmp::min(8, #len - output_lane);
                        control[output_lane..output_lane + write_len]
                            .copy_from_slice(&adjusted[..write_len]);
                        output_lane += #block_count;
                    }
                    let control = #vec::simd_from(self, control);
                    #finish
                }
            }
        }
        OpSig::Expand { merge } => {
            let block_count = count_8(quote! { block_mask });
            let finish = if merge {
                if let Some(merge_swizzle) = options.merge_swizzle {
                    merge_swizzle(
                        ty,
                        &quote! { values },
                        &quote! { control },
                        &quote! { merge },
                    )
                } else {
                    quote! {
                        let expanded = self.#swizzle(values, control);
                        self.#select(mask, expanded, merge)
                    }
                }
            } else {
                quote! { self.#swizzle(values, control) }
            };
            quote! {
                #method_sig {
                    let mask_bits = self.#to_bitmask(mask);
                    let mut control = [u8::MAX; #len];
                    let mut input_lane = 0;
                    for block in 0..#len / 8 {
                        let block_mask = ((mask_bits >> (block * 8)) & 0xff) as usize;
                        let packed = crate::support::EXPAND_8_CONTROLS[block_mask];
                        let base = input_lane as u64 * 0x0101_0101_0101_0101;
                        let adjusted = ((packed & 0x7f7f_7f7f_7f7f_7f7f) + base)
                            | (packed & 0x8080_8080_8080_8080);
                        let output_lane = block * 8;
                        control[output_lane..output_lane + 8]
                            .copy_from_slice(&adjusted.to_le_bytes());
                        input_lane += #block_count;
                    }
                    let control = #vec::simd_from(self, control);
                    #finish
                }
            }
        }
        OpSig::LoadExpand { merge } => {
            let expand = generic_op_name(if merge { "expand_merge" } else { "expand" }, ty);
            let call = if merge {
                quote! { self.#expand(values, mask, merge) }
            } else {
                quote! { self.#expand(values, mask) }
            };
            quote! {
                #method_sig {
                    let values = #vec::simd_from(self, *source);
                    #call
                }
            }
        }
        _ => unreachable!("composed_compact_op only implements compact byte operations"),
    }
}

pub(crate) fn unrolled_array(len: usize, item: impl FnMut(usize) -> TokenStream) -> TokenStream {
    let items = (0..len).map(item).collect::<Vec<_>>();
    quote! { [#(#items),*] }
}

pub(crate) fn generic_block_split(
    method_sig: TokenStream,
    half_ty: &VecType,
    max_block_size: usize,
) -> TokenStream {
    let split_arch_ty = half_ty.aligned_wrapper();
    let half_rust = half_ty.rust();
    let expr = match (half_ty.n_bits(), max_block_size) {
        (256, 128) => quote! {
            (
                #half_rust { val: #split_arch_ty([a.val.0[0], a.val.0[1]]), simd: self },
                #half_rust { val: #split_arch_ty([a.val.0[2], a.val.0[3]]), simd: self },
            )
        },
        (128, 128) | (256, 256) => quote! {
            (
                #half_rust { val: #split_arch_ty(a.val.0[0]), simd: self },
                #half_rust { val: #split_arch_ty(a.val.0[1]), simd: self },
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

pub(crate) fn generic_block_combine(
    method_sig: TokenStream,
    combined_ty: &VecType,
    max_block_size: usize,
) -> TokenStream {
    let combined_arch_ty = combined_ty.aligned_wrapper();
    let combined_rust = combined_ty.rust();
    let expr = match (combined_ty.n_bits(), max_block_size) {
        (512, 128) => quote! {
            #combined_rust {val: #combined_arch_ty([a.val.0[0], a.val.0[1], b.val.0[0], b.val.0[1]]), simd: self }
        },
        (256, 128) | (512, 256) => quote! {
            #combined_rust {val: #combined_arch_ty([a.val.0, b.val.0]), simd: self }
        },
        _ => unimplemented!(),
    };
    quote! {
        #method_sig {
            #expr
        }
    }
}

pub(crate) fn generic_mask_from_bitmask(method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let len = vec_ty.len;
    let lanes = unrolled_array(len, |idx| {
        let bit = if idx == 0 {
            quote! { bits & 1 }
        } else {
            quote! { (bits >> #idx) & 1 }
        };
        quote! { if #bit != 0 { !0 } else { 0 } }
    });

    quote! {
        #method_sig {
            let lanes: [#scalar; #len] = #lanes;
            lanes.simd_into(self)
        }
    }
}

pub(crate) fn generic_mask_to_bitmask(method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let len = vec_ty.len;

    quote! {
        #method_sig {
            let lanes: [#scalar; #len] = a.into();
            let mut bits = 0u64;
            let mut i = 0;
            while i < #len {
                if lanes[i] != 0 {
                    bits |= 1u64 << i;
                }
                i += 1;
            }
            bits
        }
    }
}

pub(crate) fn generic_mask_set(method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
    let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let len = vec_ty.len;

    quote! {
        #method_sig {
            assert!(
                index < #len,
                "mask lane index {index} is out of bounds for {} lanes",
                #len
            );
            let mut lanes: [#scalar; #len] = (*a).into();
            lanes[index] = if value { !0 } else { 0 };
            *a = lanes.simd_into(self);
        }
    }
}
