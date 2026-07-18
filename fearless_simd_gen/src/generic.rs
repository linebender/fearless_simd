// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{ToTokens, quote};

use crate::{
    level::Level,
    ops::{ElementDirection, Op, OpSig, RefKind, SlideGranularity},
    types::{ScalarType, VecType},
};

pub(crate) fn generic_op_name(op: &str, ty: &VecType) -> Ident {
    Ident::new(&format!("{op}_{}", ty.rust_name()), Span::call_site())
}

pub(crate) fn fallback_method(op: Op, vec_ty: &VecType) -> TokenStream {
    crate::mk_fallback::Fallback.make_method(op, vec_ty)
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

/// Implementation based on split/combine
///
/// Only suitable for lane-wise and block-wise operations
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
        }
        | OpSig::Reinterpret {
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
            let combine = generic_op_name("combine", &target_ty);
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::MaskReduce { quantifier, .. } => {
            let combine_op = quantifier.bool_op();
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#do_half(a0) #combine_op self.#do_half(a1)
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
        OpSig::LoadInterleaved {
            block_size,
            block_count,
        } => {
            let split_len = (block_size * block_count) as usize / (ty.scalar_bits * 2);
            let ty_rust = ty.rust();
            quote! {
                #method_sig {
                    let (chunks, _) = src.as_chunks::<#split_len>();
                    let lo = self.#do_half(&chunks[0]);
                    let hi = self.#do_half(&chunks[1]);
                    #ty_rust {
                        val: crate::transmute::checked_transmute_copy(&[lo.val, hi.val]),
                        simd: self,
                    }
                }
            }
        }
        OpSig::StoreInterleaved { .. } => {
            panic!("The generic fallback is not implemented for this operation")
        }
        OpSig::Split { .. }
        | OpSig::Combine { .. }
        | OpSig::AsArray { .. }
        | OpSig::FromArray { .. }
        | OpSig::StoreArray => {
            panic!("These operations require more information about the target platform");
        }
        OpSig::FromBytes => generic_from_bytes(method_sig, ty),
        OpSig::ToBytes => generic_to_bytes(method_sig, ty),
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
        OpSig::ElementRotate { direction } => {
            let slide = generic_op_name("slide", ty);
            let len = Literal::usize_unsuffixed(ty.len);
            match direction {
                ElementDirection::Left => {
                    let arms = modulo_offset_arms(ty, |offset| {
                        let offset = Literal::usize_unsuffixed(offset);
                        quote! { self.#slide::<#offset>(a, a) }
                    });
                    quote! {
                        #method_sig {
                            match OFFSET % #len {
                                #(#arms,)*
                                _ => unreachable!(),
                            }
                        }
                    }
                }
                ElementDirection::Right => {
                    let arms = modulo_offset_arms(ty, |offset| {
                        let shift = if offset == 0 { ty.len } else { ty.len - offset };
                        let shift = Literal::usize_unsuffixed(shift);
                        quote! { self.#slide::<#shift>(a, a) }
                    });
                    quote! {
                        #method_sig {
                            match OFFSET % #len {
                                #(#arms,)*
                                _ => unreachable!(),
                            }
                        }
                    }
                }
            }
        }
        OpSig::ElementShift { direction } => {
            let splat = generic_op_name("splat", ty);
            let slide = generic_op_name("slide", ty);
            match direction {
                ElementDirection::Left => {
                    let arms =
                        offset_arms(ty, |shift| quote! { self.#slide::<#shift>(a, padding) });
                    let all_padding = Literal::usize_unsuffixed(ty.len);
                    quote! {
                        #method_sig {
                            let padding = self.#splat(padding);
                            match OFFSET {
                                #(#arms,)*
                                _ => self.#slide::<#all_padding>(a, padding),
                            }
                        }
                    }
                }
                ElementDirection::Right => {
                    let arms =
                        right_offset_arms(ty, |shift| quote! { self.#slide::<#shift>(padding, a) });
                    quote! {
                        #method_sig {
                            let padding = self.#splat(padding);
                            match OFFSET {
                                #(#arms,)*
                                _ => self.#slide::<0>(padding, a),
                            }
                        }
                    }
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

fn modulo_offset_arms(
    ty: &VecType,
    mut body: impl FnMut(usize) -> TokenStream,
) -> Vec<TokenStream> {
    (0..ty.len)
        .map(|offset| {
            let offset_lit = Literal::usize_unsuffixed(offset);
            let body = body(offset);
            quote! { #offset_lit => #body }
        })
        .collect()
}

fn offset_arms(ty: &VecType, mut body: impl FnMut(Literal) -> TokenStream) -> Vec<TokenStream> {
    (0..=ty.len)
        .map(|offset| {
            let offset_lit = Literal::usize_unsuffixed(offset);
            let body = body(offset_lit.clone());
            quote! { #offset_lit => #body }
        })
        .collect()
}

fn right_offset_arms(
    ty: &VecType,
    mut body: impl FnMut(Literal) -> TokenStream,
) -> Vec<TokenStream> {
    (0..=ty.len)
        .map(|offset| {
            let offset_lit = Literal::usize_unsuffixed(offset);
            let shift = Literal::usize_unsuffixed(ty.len - offset);
            let body = body(shift);
            quote! { #offset_lit => #body }
        })
        .collect()
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

pub(crate) fn generic_from_array(
    method_sig: TokenStream,
    vec_ty: &VecType,
    kind: RefKind,
) -> TokenStream {
    let inner_ref = if kind == RefKind::Value {
        quote! { &val }
    } else {
        quote! { val }
    };
    // There are architecture-specific "load" intrinsics, but they can actually be *worse* for performance. If they
    // lower to LLVM intrinsics, they will likely not be optimized until much later in the pipeline (if at all),
    // resulting in substantially worse codegen. See https://github.com/linebender/fearless_simd/pull/185.
    let expr = quote! {
        crate::transmute::checked_transmute_copy(#inner_ref)
    };
    let vec_rust = vec_ty.rust();

    quote! {
        #method_sig {
            #vec_rust { val: #expr, simd: self }
        }
    }
}

pub(crate) fn generic_as_array<T: ToTokens>(
    method_sig: TokenStream,
    vec_ty: &VecType,
    kind: RefKind,
    max_block_size: usize,
    arch_ty: impl Fn(&VecType) -> T,
) -> TokenStream {
    let rust_scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let num_scalars = vec_ty.len;

    let native_ty =
        vec_ty.wrapped_native_ty(|vec_ty| arch_ty(vec_ty).into_token_stream(), max_block_size);

    match kind {
        RefKind::Value => quote! {
            #method_sig {
                crate::transmute::checked_transmute_copy::<#native_ty, [#rust_scalar; #num_scalars]>(&a.val.0)
            }
        },
        RefKind::Ref => quote! {
            #method_sig {
                crate::transmute::checked_cast_ref::<#native_ty, [#rust_scalar; #num_scalars]>(&a.val.0)
            }
        },
        RefKind::Mut => quote! {
            #method_sig {
                crate::transmute::checked_cast_mut::<#native_ty, [#rust_scalar; #num_scalars]>(&mut a.val.0)
            }
        },
    }
}

pub(crate) fn generic_store_array(method_sig: TokenStream, _vec_ty: &VecType) -> TokenStream {
    quote! {
        #method_sig {
            crate::transmute::checked_transmute_store(a.val.0, dest);
        }
    }
}

pub(crate) fn generic_to_bytes(method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
    let bytes_ty = vec_ty.reinterpret(ScalarType::Unsigned, 8).rust();
    quote! {
        #method_sig {
            #bytes_ty { val: crate::transmute::checked_transmute_copy(&a.val), simd: self }
        }
    }
}

pub(crate) fn generic_from_bytes(method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
    let ty = vec_ty.rust();
    quote! {
        #method_sig {
            #ty { val: crate::transmute::checked_transmute_copy(&a.val), simd: self }
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
    let as_array = generic_op_name("as_array", vec_ty);
    let len = vec_ty.len;

    quote! {
        #method_sig {
            let lanes = self.#as_array(a);
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
    let from_array = generic_op_name("load_array", vec_ty);
    let as_array = generic_op_name("as_array", vec_ty);
    let len = vec_ty.len;

    quote! {
        #method_sig {
            assert!(
                index < #len,
                "mask lane index {index} is out of bounds for {} lanes",
                #len
            );
            let mut lanes = self.#as_array(*a);
            lanes[index] = if value { !0 } else { 0 };
            *a = self.#from_array(lanes);
        }
    }
}
