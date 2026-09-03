// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::x86::{
    self, cast_ident, coarse_type, extend_intrinsic, float_compare_method, intrinsic_ident,
    op_suffix, pack_intrinsic, set1_intrinsic, simple_intrinsic, simple_sign_unaware_intrinsic,
    unpack_intrinsic,
};
use crate::generic::{
    count_zeros_method, fallback_method, generic_block_combine, generic_block_split,
    generic_mask_from_bitmask, generic_mask_set, generic_op_name, integer_lane_mask_rotate,
    integer_lane_mask_splat_arg, recursive_swizzle_dyn_precise_body, reverse_method,
    reverse_vector_mask_method,
};
use crate::level::Level;
use crate::ops::{
    ElementDirection, NarrowingMode, Op, OpSig, Quantifier, SlideGranularity, relaxed_narrow_method,
};
use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{ToTokens as _, format_ident, quote};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum X86 {
    Sse2,
    Sse4_2,
    Avx2,
    Avx512,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Precision {
    Approx,
    Precise,
}

pub(crate) const SSE2_FEATURES: &str = "fxsr,sse,sse2";
pub(crate) const SSE4_2_FEATURES: &str = "fxsr,sse4.2,cmpxchg16b,popcnt";
pub(crate) const AVX2_FEATURES: &str =
    "avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,fxsr,lzcnt,movbe,popcnt,xsave";
pub(crate) const AVX512_FEATURES: &str = "adx,aes,avx512bitalg,avx512bw,avx512cd,avx512dq,avx512f,avx512ifma,avx512vbmi,avx512vbmi2,avx512vl,avx512vnni,avx512vpopcntdq,bmi1,bmi2,cmpxchg16b,fma,fxsr,gfni,lzcnt,movbe,pclmulqdq,popcnt,rdrand,rdseed,sha,vaes,vpclmulqdq,xsave,xsavec,xsaveopt,xsaves";

fn sse2_scalar_mul_add_precise_f32_helper() -> TokenStream {
    let body = crate::mk_fallback::scalar_mul_add_precise_f32_body();
    quote! {
        crate::kernel!(
            #[inline(always)]
            fn scalar_mul_add_precise_f32(_token: Sse2, a: f32, b: f32, c: f32) -> f32 {
                #body
            }
        );
    }
}

impl Level for X86 {
    fn name(&self) -> &'static str {
        match self {
            Self::Sse2 => "Sse2",
            Self::Sse4_2 => "Sse4_2",
            Self::Avx2 => "Avx2",
            Self::Avx512 => "Avx512",
        }
    }

    fn native_width(&self) -> usize {
        match self {
            Self::Sse2 => 128,
            Self::Sse4_2 => 128,
            Self::Avx2 => 256,
            Self::Avx512 => 512,
        }
    }

    fn max_block_size(&self) -> usize {
        self.native_width()
    }

    fn enabled_target_features(&self) -> Option<&'static str> {
        Some(match self {
            Self::Sse2 => SSE2_FEATURES,
            Self::Sse4_2 => SSE4_2_FEATURES,
            Self::Avx2 => AVX2_FEATURES,
            Self::Avx512 => AVX512_FEATURES,
        })
    }

    fn arch_ty(&self, vec_ty: &VecType) -> TokenStream {
        // AVX-512 masks are compact predicate registers, not vector registers.
        if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Mask {
            let bits = avx512_mask_register_bits(vec_ty);
            let name = format!("__mmask{bits}");
            return Ident::new(&name, Span::call_site()).into_token_stream();
        }

        let suffix = match (vec_ty.scalar, vec_ty.scalar_bits) {
            (ScalarType::Float, 32) => "",
            (ScalarType::Float, 64) => "d",
            (ScalarType::Float, _) => unimplemented!(),
            (ScalarType::Unsigned | ScalarType::Int | ScalarType::Mask, _) => "i",
        };
        let name = format!("__m{}{}", vec_ty.scalar_bits * vec_ty.len, suffix);
        Ident::new(&name, Span::call_site()).into_token_stream()
    }

    fn arch_storage_ty(&self, vec_ty: &VecType) -> TokenStream {
        if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Mask {
            self.arch_ty(vec_ty)
        } else {
            vec_ty.aligned_wrapper_ty(|vec_ty| self.arch_ty(vec_ty), self.max_block_size())
        }
    }

    fn custom_mask_array_conversion(&self, vec_ty: &VecType) -> Option<TokenStream> {
        if *self != Self::Avx512 || vec_ty.scalar != ScalarType::Mask {
            return None;
        }

        Some(self.avx512_mask_array_conversion(vec_ty))
    }

    fn token_doc(&self) -> &'static str {
        match self {
            Self::Sse2 => {
                "A token for SSE2 intrinsics on `x86` and `x86_64`, representing the x86-64 baseline."
            }
            Self::Sse4_2 => {
                "A token for SSE4.2 intrinsics on `x86` and `x86_64`, representing the x86-64-v2 level."
            }
            Self::Avx2 => {
                "A token for AVX2 intrinsics on `x86` and `x86_64`, representing the x86-64-v3 level."
            }
            Self::Avx512 => {
                "A token for AVX-512 intrinsics on `x86` and `x86_64`, representing an Ice Lake feature level."
            }
        }
    }

    fn make_module_prelude(&self) -> TokenStream {
        let float_ext = if matches!(self, Self::Sse2 | Self::Sse4_2) {
            crate::mk_fallback::float_ext_prelude()
        } else {
            TokenStream::new()
        };
        let scalar_mul_add_precise_f32 = if *self == Self::Sse2 {
            sse2_scalar_mul_add_precise_f32_helper()
        } else {
            TokenStream::new()
        };

        quote! {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::*;
            use core::ops::*;
            #scalar_mul_add_precise_f32
            #float_ext
        }
    }

    fn make_module_attrs(&self) -> TokenStream {
        if *self != Self::Avx512 {
            return TokenStream::new();
        }

        quote! {
            #![allow(
                clippy::identity_op,
                reason = "AVX-512 mask code is generated uniformly for all __mmask widths"
            )]
            #![allow(
                clippy::useless_conversion,
                reason = "AVX-512 mask code is generated uniformly for all __mmask widths"
            )]
        }
    }

    fn make_module_footer(&self) -> TokenStream {
        let alignr_helpers = self.dyn_alignr_helpers();
        let slide_helpers = match self {
            Self::Sse2 | Self::Sse4_2 => Self::sse_slide_helpers(self.token()),
            Self::Avx2 => Self::avx2_slide_helpers(),
            Self::Avx512 => TokenStream::new(),
        };

        quote! {
            #alignr_helpers
            #slide_helpers
        }
    }

    fn make_level_body(&self) -> TokenStream {
        let level_tok = self.token();

        quote! {
            Level::#level_tok(self)
        }
    }

    fn should_impl_arch_type_conversion(&self, ty: &VecType) -> bool {
        if *self == Self::Sse4_2 {
            return false; // already covered by SSE2 which has the same vector widths
        }

        let n_bits = ty.n_bits();
        // AVX-512 masks are not 512-bit vectors and need special handling
        if *self == Self::Avx512 && ty.scalar == ScalarType::Mask {
            return n_bits <= self.max_block_size();
        }

        n_bits <= self.max_block_size() && n_bits >= self.native_width()
    }

    fn should_use_bitmask_arch_type_conversion(&self, ty: &VecType) -> bool {
        *self == Self::Avx512 && ty.scalar == ScalarType::Mask
    }

    fn custom_arch_type_conversion(&self, ty: &VecType) -> Option<TokenStream> {
        if *self == Self::Avx512 || ty.scalar != ScalarType::Mask {
            return None;
        }

        let simd = ty.rust();
        let arch = self.arch_ty(ty);
        let lane_ty = ScalarType::Int.rust(ty.scalar_bits);
        let len = ty.len;

        Some(quote! {
            impl<S: Simd> SimdFrom<#arch, S> for #simd<S> {
                #[inline(always)]
                fn simd_from(simd: S, arch: #arch) -> Self {
                    let lanes: [#lane_ty; #len] =
                        crate::transmute::checked_transmute_copy(&arch);
                    lanes.simd_into(simd)
                }
            }
            impl<S: Simd> From<#simd<S>> for #arch {
                #[inline(always)]
                fn from(value: #simd<S>) -> Self {
                    let lanes: [#lane_ty; #len] = value.into();
                    crate::transmute::checked_transmute_copy(&lanes)
                }
            }
        })
    }

    fn make_impl_body(&self) -> TokenStream {
        let features = self
            .enabled_target_features()
            .expect("x86 SIMD levels always enable target features");
        let description_doc = match self {
            Self::Sse2 => {
                "Create a SIMD token proving that SSE2 is available.\n\n\
                 This is the baseline on x86-64 and i686 targets. On i586 it needs runtime detection."
            }
            Self::Sse4_2 => {
                "Create a SIMD token proving that the x86-64-v2 features are available."
            }
            Self::Avx2 => "Create a SIMD token proving that the x86-64-v3 features are available.",
            Self::Avx512 => {
                "Create a SIMD token proving that the Ice Lake AVX-512 features are available."
            }
        };
        let required_features = features.replace(',', "`, `");
        let safety_doc = format!(
            "When invoking this function through an `unsafe` block, the caller must ensure \
             that the current CPU supports `{required_features}`."
        );

        quote! {
            #[doc = #description_doc]
            ///
            /// Most users should safely obtain the token from [`Level::new`] instead of calling this function.
            ///
            /// This function can be called without an `unsafe` block from a function
            /// with all the required target features enabled via the `#[target_feature]` annotation.
            ///
            /// # Safety
            ///
            #[doc = #safety_doc]
            #[inline]
            #[target_feature(enable = #features)]
            pub const fn assume_supported() -> Self {
                Self { _private: () }
            }
        }
    }

    fn should_use_generic_op(&self, op: &Op, vec_ty: &VecType) -> bool {
        if *self == Self::Avx512
            && matches!(
                op.sig,
                OpSig::Slide {
                    granularity: SlideGranularity::WithinBlocks,
                    ..
                }
            )
            && vec_ty.scalar == ScalarType::Mask
            && vec_ty.n_bits() > 128
        {
            return true;
        }

        let should_use_generic = op.sig.should_use_generic_op(vec_ty, self.native_width());
        if !should_use_generic {
            return false;
        }

        match op.sig {
            OpSig::MaskFromBitmask => !self.has_specialized_mask_from_bitmask(vec_ty),
            OpSig::MaskToBitmask => !self.has_specialized_mask_to_bitmask(vec_ty),
            _ => true,
        }
    }

    fn make_method(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let Op { sig, method, .. } = op;
        let method_sig = op.simd_trait_method_sig(vec_ty);

        match sig {
            OpSig::Splat => self.handle_splat(op, vec_ty),
            OpSig::Compare => self.handle_compare(op, method, vec_ty),
            OpSig::Unary => self.handle_unary(op, method_sig, method, vec_ty),
            OpSig::Reduce { lane_op } => {
                if lane_op == "add" {
                    self.handle_reduce_sum(op, vec_ty)
                } else {
                    self.handle_reduce_min_max(op, vec_ty, lane_op)
                }
            }
            OpSig::RotateElements { direction } => {
                self.handle_mask_rotate_elements(op, vec_ty, direction)
            }
            OpSig::Widen { target_ty } => self.handle_widen(op, vec_ty, target_ty),
            OpSig::Narrow { target_ty, mode } => self.handle_narrow(op, vec_ty, target_ty, mode),
            OpSig::Binary => self.handle_binary(op, method, vec_ty),
            OpSig::Shift => self.handle_shift(op, method, vec_ty),
            OpSig::Ternary => self.handle_ternary(op, method_sig, method, vec_ty),
            OpSig::Select => self.handle_select(op, vec_ty),
            OpSig::Combine { combined_ty } => self.handle_combine(op, vec_ty, &combined_ty),
            OpSig::Split { half_ty } => self.handle_split(op, vec_ty, &half_ty),
            OpSig::Zip { select_low } => self.handle_zip(op, vec_ty, select_low),
            OpSig::Unzip { select_even } => self.handle_unzip(op, vec_ty, select_even),
            OpSig::Slide { granularity } => self.handle_slide(method_sig, vec_ty, granularity),
            OpSig::SwizzleDynWithinBlocks => self.handle_swizzle_dyn_within_blocks(op, vec_ty),
            OpSig::SwizzleDyn => self.handle_swizzle_dyn(op, vec_ty),
            OpSig::SwizzleDynPrecise => self.handle_swizzle_dyn_precise(op, vec_ty),
            OpSig::Cvt {
                target_ty,
                scalar_bits,
                precise,
            } => self.handle_cvt(op, vec_ty, target_ty, scalar_bits, precise),
            OpSig::MaskReduce {
                quantifier,
                condition,
            } => self.handle_mask_reduce(op, vec_ty, quantifier, condition),
            OpSig::MaskFromBitmask => self.handle_mask_from_bitmask(op, vec_ty),
            OpSig::MaskToBitmask => self.handle_mask_to_bitmask(op, vec_ty),
            OpSig::MaskSet if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Mask => {
                self.handle_avx512_mask_set(method_sig, vec_ty)
            }
            OpSig::MaskSet => generic_mask_set(method_sig, vec_ty),
            OpSig::LoadInterleaved {
                block_size,
                block_count,
            } => self.handle_load_interleaved(op, vec_ty, block_size, block_count),
            OpSig::StoreInterleaved {
                block_size,
                block_count,
            } => self.handle_store_interleaved(op, vec_ty, block_size, block_count),
            OpSig::Interleave => self.handle_interleave(op, vec_ty),
            OpSig::Deinterleave => self.handle_deinterleave(op, vec_ty),
        }
    }
}

fn mask_from_bitmask_bytes(vec_ty: &VecType) -> TokenStream {
    let lane_count = vec_ty.len;
    let bit_mask_128 = mask_bit_pattern_128();

    if lane_count <= 8 {
        return quote! {
            {
                let bit_bytes = _mm_set1_epi8(bits as i8);
                let bit_mask = #bit_mask_128;
                _mm_cmpeq_epi8(_mm_and_si128(bit_bytes, bit_mask), bit_mask)
            }
        };
    }

    if lane_count <= 16 {
        let shuffle = mask_byte_shuffle_128(lane_count);
        return quote! {
            {
                let bit_bytes = _mm_cvtsi32_si128(bits as i32);
                let bit_bytes = _mm_shuffle_epi8(bit_bytes, #shuffle);
                let bit_mask = #bit_mask_128;
                _mm_cmpeq_epi8(_mm_and_si128(bit_bytes, bit_mask), bit_mask)
            }
        };
    }

    assert_eq!(
        (vec_ty.n_bits(), vec_ty.scalar_bits, lane_count),
        (256, 8, 32),
        "only 32-lane masks need a 256-bit inverse movemask"
    );

    let shuffle = mask_byte_shuffle_256();
    let bit_mask = mask_bit_pattern_256();
    quote! {
        {
            let bit_bytes = _mm256_broadcastsi128_si256(_mm_cvtsi32_si128(bits as i32));
            let bit_bytes = _mm256_shuffle_epi8(bit_bytes, #shuffle);
            let bit_mask = #bit_mask;
            _mm256_cmpeq_epi8(_mm256_and_si256(bit_bytes, bit_mask), bit_mask)
        }
    }
}

fn mask_from_bitmask_lanes(vec_ty: &VecType) -> TokenStream {
    let lane_count = vec_ty.len;
    let scalar_bits = vec_ty.scalar_bits;

    match (vec_ty.n_bits(), scalar_bits) {
        (128, 16) => {
            let lanes = (0..lane_count).map(|i| {
                let bit = 1_u16 << i;
                signed_literal(bit.into(), 16)
            });
            quote! {
                {
                    let bit_lanes = _mm_set1_epi16(bits as i16);
                    let bit_mask = _mm_setr_epi16(#(#lanes),*);
                    _mm_cmpeq_epi16(_mm_and_si128(bit_lanes, bit_mask), bit_mask)
                }
            }
        }
        (256, 16) => {
            let lanes = (0..lane_count).map(|i| {
                let bit = 1_u16 << i;
                signed_literal(bit.into(), 16)
            });
            quote! {
                {
                    let bit_lanes = _mm256_set1_epi16(bits as i16);
                    let bit_mask = _mm256_setr_epi16(#(#lanes),*);
                    _mm256_cmpeq_epi16(_mm256_and_si256(bit_lanes, bit_mask), bit_mask)
                }
            }
        }
        (128, 32) => {
            let lanes = (0..lane_count).map(|i| {
                let bit = 1_u32 << i;
                signed_literal(bit.into(), 32)
            });
            quote! {
                {
                    let bit_lanes = _mm_set1_epi32(bits as i32);
                    let bit_mask = _mm_setr_epi32(#(#lanes),*);
                    _mm_cmpeq_epi32(_mm_and_si128(bit_lanes, bit_mask), bit_mask)
                }
            }
        }
        (256, 32) => {
            let lanes = (0..lane_count).map(|i| {
                let bit = 1_u32 << i;
                signed_literal(bit.into(), 32)
            });
            quote! {
                {
                    let bit_lanes = _mm256_set1_epi32(bits as i32);
                    let bit_mask = _mm256_setr_epi32(#(#lanes),*);
                    _mm256_cmpeq_epi32(_mm256_and_si256(bit_lanes, bit_mask), bit_mask)
                }
            }
        }
        (128, 64) => {
            assert_eq!(lane_count, 2, "128-bit 64-bit masks must have two lanes");
            quote! {
                {
                    let bit_lanes = _mm_set1_epi64x(bits.cast_signed());
                    let bit_mask = _mm_set_epi64x(2, 1);
                    _mm_cmpeq_epi64(_mm_and_si128(bit_lanes, bit_mask), bit_mask)
                }
            }
        }
        (256, 64) => {
            assert_eq!(lane_count, 4, "256-bit 64-bit masks must have four lanes");
            quote! {
                {
                    let bit_lanes = _mm256_set1_epi64x(bits.cast_signed());
                    let bit_mask = _mm256_set_epi64x(8, 4, 2, 1);
                    _mm256_cmpeq_epi64(_mm256_and_si256(bit_lanes, bit_mask), bit_mask)
                }
            }
        }
        _ => unimplemented!(),
    }
}

fn mask_from_bitmask_wide_avx2(vec_ty: &VecType, simd: &Ident) -> TokenStream {
    assert_eq!(
        vec_ty.n_bits(),
        512,
        "only 512-bit masks use direct wide AVX2 bitmask lowering"
    );
    assert!(
        matches!(vec_ty.scalar_bits, 32 | 64),
        "only 32-bit and 64-bit AVX2 masks use direct wide lowering"
    );

    let ty = vec_ty.rust();
    let lanes_per_chunk = 256 / vec_ty.scalar_bits;
    let chunks = (0..2).map(|chunk| {
        let chunk_start = chunk * lanes_per_chunk;
        match vec_ty.scalar_bits {
            32 => {
                let lanes = (0..lanes_per_chunk).map(|i| {
                    let bit = 1_u32 << (chunk_start + i);
                    signed_literal(bit.into(), 32)
                });
                quote! {
                    {
                        let bit_mask = _mm256_setr_epi32(#(#lanes),*);
                        _mm256_cmpeq_epi32(_mm256_and_si256(bit_lanes, bit_mask), bit_mask)
                    }
                }
            }
            64 => {
                let lanes = (0..lanes_per_chunk).rev().map(|i| {
                    let bit = 1_u64 << (chunk_start + i);
                    signed_literal(bit, 64)
                });
                quote! {
                    {
                        let bit_mask = _mm256_set_epi64x(#(#lanes),*);
                        _mm256_cmpeq_epi64(_mm256_and_si256(bit_lanes, bit_mask), bit_mask)
                    }
                }
            }
            _ => unreachable!(),
        }
    });
    let set1 = match vec_ty.scalar_bits {
        32 => quote! { _mm256_set1_epi32(bits as i32) },
        64 => quote! { _mm256_set1_epi64x(bits.cast_signed()) },
        _ => unreachable!(),
    };

    quote! {
        {
            let bit_lanes = #set1;
            #ty {
                val: crate::support::Aligned512([#(#chunks),*]),
                simd: #simd,
            }
        }
    }
}

fn mask_from_bitmask_wide_bytes(
    native_width: usize,
    vec_ty: &VecType,
    simd: &Ident,
) -> TokenStream {
    assert_eq!(
        vec_ty.n_bits(),
        512,
        "only 512-bit masks use direct wide byte-mask lowering"
    );
    assert_eq!(
        vec_ty.scalar_bits, 8,
        "only mask8x64 uses direct wide byte-mask lowering"
    );

    let ty = vec_ty.rust();
    match native_width {
        128 => {
            let bit_mask = mask_bit_pattern_128();
            let chunks = (0..4).map(|chunk| {
                let shuffle = mask_byte_shuffle_128_offset(16, chunk * 2);
                quote! {
                    {
                        let bit_bytes = _mm_shuffle_epi8(bit_bytes, #shuffle);
                        _mm_cmpeq_epi8(_mm_and_si128(bit_bytes, bit_mask), bit_mask)
                    }
                }
            });

            quote! {
                {
                    let bit_bytes = _mm_set1_epi64x(bits.cast_signed());
                    let bit_mask = #bit_mask;
                    #ty {
                        val: crate::support::Aligned512([#(#chunks),*]),
                        simd: #simd,
                    }
                }
            }
        }
        256 => {
            let bit_mask = mask_bit_pattern_256();
            let chunks = (0..2).map(|chunk| {
                let shuffle = mask_byte_shuffle_256_offset(chunk * 4);
                quote! {
                    {
                        let bit_bytes = _mm256_shuffle_epi8(bit_bytes, #shuffle);
                        _mm256_cmpeq_epi8(_mm256_and_si256(bit_bytes, bit_mask), bit_mask)
                    }
                }
            });

            quote! {
                {
                    let bit_bytes = _mm256_set1_epi64x(bits.cast_signed());
                    let bit_mask = #bit_mask;
                    #ty {
                        val: crate::support::Aligned512([#(#chunks),*]),
                        simd: #simd,
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}

fn mask_to_bitmask_words(native_width: usize, vec_ty: &VecType) -> TokenStream {
    assert_eq!(
        vec_ty.scalar_bits, 16,
        "only 16-bit masks use word packing to produce bitmasks"
    );

    match (native_width, vec_ty.n_bits()) {
        (128 | 256, 128) => quote! {
            {
                let packed = _mm_packs_epi16(a.into(), a.into());
                _mm_movemask_epi8(packed) as u8 as u64
            }
        },
        (128, 256) => quote! {
            {
                let packed = _mm_packs_epi16(a.val.0[0], a.val.0[1]);
                _mm_movemask_epi8(packed) as u32 as u64
            }
        },
        (128, 512) => quote! {
            {
                let lo = _mm_packs_epi16(a.val.0[0], a.val.0[1]);
                let hi = _mm_packs_epi16(a.val.0[2], a.val.0[3]);
                let lo = _mm_movemask_epi8(lo) as u32 as u64;
                let hi = _mm_movemask_epi8(hi) as u32 as u64;
                lo | (hi << 16usize)
            }
        },
        (256, 256) => quote! {
            {
                let halves: [__m128i; 2usize] = crate::transmute::checked_transmute_copy(&a.val.0);
                let packed = _mm_packs_epi16(halves[0], halves[1]);
                _mm_movemask_epi8(packed) as u32 as u64
            }
        },
        (256, 512) => quote! {
            {
                let lo = _mm256_movemask_epi8(a.val.0[0]) as u32;
                let hi = _mm256_movemask_epi8(a.val.0[1]) as u32;
                let lo = _pext_u32(lo, 0x5555_5555u32) as u64;
                let hi = _pext_u32(hi, 0x5555_5555u32) as u64;
                lo | (hi << 16usize)
            }
        },
        _ => unimplemented!(),
    }
}

fn mask_bit_pattern_128() -> TokenStream {
    let lanes = (0..16).map(|i| {
        let bit = 1_u8 << (i % 8);
        signed_literal(bit.into(), 8)
    });
    quote! { _mm_setr_epi8(#(#lanes),*) }
}

fn mask_bit_pattern_256() -> TokenStream {
    let lanes = (0..32).map(|i| {
        let bit = 1_u8 << (i % 8);
        signed_literal(bit.into(), 8)
    });
    quote! { _mm256_setr_epi8(#(#lanes),*) }
}

fn mask_byte_shuffle_128_offset(lane_count: usize, byte_offset: usize) -> TokenStream {
    let lanes = (0..16).map(|i| {
        let byte = u8::try_from(byte_offset + i.min(lane_count - 1) / 8)
            .expect("SSE byte shuffle index must fit in u8");
        signed_literal(byte.into(), 8)
    });
    quote! { _mm_setr_epi8(#(#lanes),*) }
}

fn mask_byte_shuffle_128(lane_count: usize) -> TokenStream {
    mask_byte_shuffle_128_offset(lane_count, 0)
}

fn mask_byte_shuffle_256_offset(byte_offset: usize) -> TokenStream {
    let lanes = (0..32).map(|i| {
        let byte =
            u8::try_from(byte_offset + i / 8).expect("AVX2 byte shuffle index must fit in u8");
        signed_literal(byte.into(), 8)
    });
    quote! { _mm256_setr_epi8(#(#lanes),*) }
}

fn mask_byte_shuffle_256() -> TokenStream {
    mask_byte_shuffle_256_offset(0)
}

fn signed_literal(value: u64, bits: u32) -> TokenStream {
    assert!(
        bits <= 64,
        "signed literal width must fit in a primitive integer"
    );
    let shift = 64 - bits;
    let value = (value << shift).cast_signed() >> shift;
    if value < 0 {
        let magnitude = Literal::u64_unsuffixed(value.unsigned_abs());
        quote! { -#magnitude }
    } else {
        let value = Literal::u64_unsuffixed(value as u64);
        quote! { #value }
    }
}

/// Invert an SSE2 mask expression.
///
/// SSE2 has no vector NOT intrinsic, so we XOR the mask with an all-ones vector.
fn sse2_not_mask_expr(mask: TokenStream) -> TokenStream {
    quote! {
        {
            let all_ones = _mm_set1_epi8(-1);
            _mm_xor_si128(#mask, all_ones)
        }
    }
}

/// Build a signed or unsigned integer `>` comparison using SSE2 operations.
///
/// SSE2 only provides signed greater-than comparisons for these integer lane
/// widths. Unsigned comparisons are lowered by flipping the sign bit in both
/// operands, which preserves unsigned ordering when interpreted as signed.
fn sse2_cmpgt_expr(vec_ty: &VecType, lhs: TokenStream, rhs: TokenStream) -> TokenStream {
    let gt = simple_sign_unaware_intrinsic("cmpgt", vec_ty);
    if vec_ty.scalar != ScalarType::Unsigned {
        return quote! { #gt(#lhs, #rhs) };
    }

    let set = set1_intrinsic(vec_ty);
    let xor = intrinsic_ident("xor", coarse_type(vec_ty), vec_ty.n_bits());
    let sign = match vec_ty.scalar_bits {
        8 => quote! { 0x80u8 },
        16 => quote! { 0x8000u16 },
        32 => quote! { 0x80000000u32 },
        _ => unimplemented!(),
    };

    quote! {
        {
            let sign_bit = #set(#sign.cast_signed());
            let lhs_signed = #xor(#lhs, sign_bit);
            let rhs_signed = #xor(#rhs, sign_bit);
            #gt(lhs_signed, rhs_signed)
        }
    }
}

/// Build an integer comparison expression that stays within SSE2.
///
/// Later x86 levels have richer comparisons, but SSE2 needs comparisons like
/// `<`, `<=`, and `>=` expressed in terms of signed `>` plus operand swapping
/// or mask inversion. Equality for 64-bit lanes is also synthesized from
/// 32-bit equality because SSE2 has no `_mm_cmpeq_epi64`.
fn sse2_int_compare_expr(method: &str, vec_ty: &VecType) -> TokenStream {
    match method {
        "simd_eq" if vec_ty.scalar_bits == 64 => quote! {
            {
                let eq32 = _mm_cmpeq_epi32(a.into(), b.into());
                let swapped = _mm_shuffle_epi32::<0b10_11_00_01>(eq32);
                _mm_and_si128(eq32, swapped)
            }
        },
        "simd_eq" => {
            let eq = simple_sign_unaware_intrinsic("cmpeq", vec_ty);
            quote! { #eq(a.into(), b.into()) }
        }
        "simd_lt" => sse2_cmpgt_expr(vec_ty, quote! { b.into() }, quote! { a.into() }),
        "simd_gt" => sse2_cmpgt_expr(vec_ty, quote! { a.into() }, quote! { b.into() }),
        "simd_le" => {
            let gt = sse2_cmpgt_expr(vec_ty, quote! { a.into() }, quote! { b.into() });
            sse2_not_mask_expr(gt)
        }
        "simd_ge" => {
            let gt = sse2_cmpgt_expr(vec_ty, quote! { b.into() }, quote! { a.into() });
            sse2_not_mask_expr(gt)
        }
        _ => unreachable!(),
    }
}

/// Select between two vectors using an SSE2 mask.
///
/// SSE2 predates `blendv`, but these masks are represented as all-zero or
/// all-one bits, so selection can be implemented as `(mask & true) |
/// (!mask & false)`.
fn sse2_select_expr(
    vec_ty: &VecType,
    mask: TokenStream,
    if_true: TokenStream,
    if_false: TokenStream,
) -> TokenStream {
    let and = intrinsic_ident("and", coarse_type(vec_ty), vec_ty.n_bits());
    let andnot = intrinsic_ident("andnot", coarse_type(vec_ty), vec_ty.n_bits());
    let or = intrinsic_ident("or", coarse_type(vec_ty), vec_ty.n_bits());

    quote! {
        #or(#and(#mask, #if_true), #andnot(#mask, #if_false))
    }
}

/// Build an integer min/max expression that is available on SSE2.
///
/// SSE2 has native min/max only for unsigned 8-bit and signed 16-bit lanes.
/// Other 8/16/32-bit integer min/max operations are synthesized with an SSE2
/// comparison and the bitwise select helper above.
fn sse2_min_max_expr(method: &str, vec_ty: &VecType) -> TokenStream {
    sse2_min_max_native_expr(method, vec_ty, quote! { a.into() }, quote! { b.into() })
}

/// Build an SSE2 integer min/max expression from operands that already evaluate
/// to native `__m128i` values.
fn sse2_min_max_native_expr(
    method: &str,
    vec_ty: &VecType,
    a: TokenStream,
    b: TokenStream,
) -> TokenStream {
    match (method, vec_ty.scalar, vec_ty.scalar_bits) {
        ("min", ScalarType::Unsigned, 8) | ("max", ScalarType::Unsigned, 8) => {
            let intrinsic = simple_intrinsic(method, vec_ty);
            quote! { #intrinsic(#a, #b) }
        }
        ("min", ScalarType::Int, 16) | ("max", ScalarType::Int, 16) => {
            let intrinsic = simple_intrinsic(method, vec_ty);
            quote! { #intrinsic(#a, #b) }
        }
        ("min" | "max", ScalarType::Int | ScalarType::Unsigned, 8 | 16 | 32) => {
            let gt = sse2_cmpgt_expr(vec_ty, quote! { a }, quote! { b });
            let select = if method == "max" {
                sse2_select_expr(vec_ty, quote! { gt }, quote! { a }, quote! { b })
            } else {
                sse2_select_expr(vec_ty, quote! { gt }, quote! { b }, quote! { a })
            };
            quote! {
                {
                    let a = #a;
                    let b = #b;
                    let gt = #gt;
                    #select
                }
            }
        }
        _ => unimplemented!(),
    }
}

/// Build the native expression used to combine one stage of a min/max reduction.
///
/// Keeping this in terms of the same helpers as the vertical operations ensures
/// that reductions inherit their NaN behavior. In particular, precise floating-
/// point reductions retain the explicit SSE2 NaN correction, use RANGE on
/// AVX-512, and use the regular x86 precise expression elsewhere.
fn x86_reduce_min_max_expr(level: X86, lane_op: &str, vec_ty: &VecType) -> TokenStream {
    if level == X86::Avx512
        && vec_ty.scalar == ScalarType::Float
        && matches!(lane_op, "min_precise" | "max_precise")
    {
        let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, true);
        let range = intrinsic_ident("range", suffix, vec_ty.n_bits());
        let imm = if lane_op == "max_precise" {
            0b0101
        } else {
            0b0100
        };
        return quote! { #range::<#imm>(reduced, shifted) };
    }

    if level == X86::Sse2
        && vec_ty.scalar == ScalarType::Float
        && matches!(lane_op, "min_precise" | "max_precise")
    {
        let intrinsic = simple_intrinsic(
            if lane_op == "max_precise" {
                "max"
            } else {
                "min"
            },
            vec_ty,
        );
        let cmpunord = float_compare_method("unord", vec_ty);
        let select = sse2_select_expr(
            vec_ty,
            quote! { shifted_is_nan },
            quote! { reduced },
            quote! { intermediate },
        );
        return quote! {
            let intermediate = #intrinsic(reduced, shifted);
            let shifted_is_nan = #cmpunord(shifted, shifted);
            #select
        };
    }

    if level == X86::Sse2 && matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned) {
        return sse2_min_max_native_expr(lane_op, vec_ty, quote! { reduced }, quote! { shifted });
    }

    x86::expr(lane_op, vec_ty, &[quote! { reduced }, quote! { shifted }])
}

fn avx512_mask_register_bits(vec_ty: &VecType) -> usize {
    match vec_ty.len {
        0..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        33..=64 => 64,
        _ => unreachable!("SIMD masks never have more than 64 lanes"),
    }
}

fn avx512_mask_lane_bits(vec_ty: &VecType) -> TokenStream {
    if vec_ty.len == 64 {
        quote! { u64::MAX }
    } else {
        let bits = (1_u64 << vec_ty.len) - 1;
        quote! { #bits }
    }
}

fn avx512_mask_value_with_simd(
    vec_ty: &VecType,
    bits: TokenStream,
    simd: TokenStream,
) -> TokenStream {
    let ty = vec_ty.rust();
    let bits = if avx512_mask_register_bits(vec_ty) == 64 {
        bits
    } else {
        quote! { (#bits) as _ }
    };
    quote! {
        #ty {
            val: #bits,
            simd: #simd,
        }
    }
}

fn avx512_mask_value(vec_ty: &VecType, bits: TokenStream) -> TokenStream {
    avx512_mask_value_with_simd(vec_ty, bits, quote! { self })
}

fn avx512_mask_register_value_with_simd(
    vec_ty: &VecType,
    bits: TokenStream,
    simd: TokenStream,
) -> TokenStream {
    let ty = vec_ty.rust();
    quote! {
        #ty {
            val: #bits,
            simd: #simd,
        }
    }
}

fn avx512_mask_bits_expr(expr: TokenStream) -> TokenStream {
    quote! { u64::from((#expr).val) }
}

fn avx512_compare_op(method: &str) -> &'static str {
    match method {
        "simd_eq" => "cmpeq",
        "simd_lt" => "cmplt",
        "simd_le" => "cmple",
        "simd_ge" => "cmpge",
        "simd_gt" => "cmpgt",
        _ => unreachable!(),
    }
}

fn avx512_float_compare_predicate(method: &str) -> i32 {
    // source for the values:
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmp_ps_mask
    match method {
        "simd_eq" => 0x00,
        "simd_lt" => 0x11,
        "simd_le" => 0x12,
        "simd_ge" => 0x1D,
        "simd_gt" => 0x1E,
        "ord" => 0x07,
        "unord" => 0x03,
        _ => unreachable!(),
    }
}

fn avx512_mask_compare_expr(method: &str, vec_ty: &VecType) -> TokenStream {
    let lane_mask = avx512_mask_lane_bits(vec_ty);
    match method {
        "simd_eq" => quote! { !u64::from(a.val ^ b.val) & #lane_mask },
        _ => unreachable!("masks only support equality comparison"),
    }
}

fn avx512_permutex2var_intrinsic(vec_ty: &VecType) -> Ident {
    let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
    intrinsic_ident("permutex2var", suffix, vec_ty.n_bits())
}

fn avx512_should_use_unzip_permutex2var(vec_ty: &VecType) -> bool {
    vec_ty.scalar != ScalarType::Mask
        && (vec_ty.n_bits() >= 256
            || (vec_ty.n_bits() == 128
                && matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned)))
}

fn avx512_permutexvar_intrinsic(vec_ty: &VecType) -> Ident {
    let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
    intrinsic_ident("permutexvar", suffix, vec_ty.n_bits())
}

fn avx512_mask_blend_intrinsic(vec_ty: &VecType) -> Ident {
    let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
    intrinsic_ident("mask_blend", suffix, vec_ty.n_bits())
}

fn avx512_index_vector(vec_ty: &VecType, indices: impl IntoIterator<Item = usize>) -> TokenStream {
    let indices: Vec<usize> = indices.into_iter().collect();
    let n_bits = vec_ty.n_bits();
    let scalar_bits = vec_ty.scalar_bits;
    match (n_bits, scalar_bits) {
        (128, 8) => {
            let lanes = indices.into_iter().map(|i| signed_literal(i as u64, 8));
            quote! { _mm_setr_epi8(#(#lanes),*) }
        }
        (256, 8) => {
            let lanes = indices.into_iter().map(|i| signed_literal(i as u64, 8));
            quote! { _mm256_setr_epi8(#(#lanes),*) }
        }
        (512, 8) => {
            let lanes = indices
                .into_iter()
                .rev()
                .map(|i| signed_literal(i as u64, 8));
            quote! { _mm512_set_epi8(#(#lanes),*) }
        }
        (128, 16) => {
            let lanes = indices.into_iter().map(|i| signed_literal(i as u64, 16));
            quote! { _mm_setr_epi16(#(#lanes),*) }
        }
        (256, 16) => {
            let lanes = indices.into_iter().map(|i| signed_literal(i as u64, 16));
            quote! { _mm256_setr_epi16(#(#lanes),*) }
        }
        (512, 16) => {
            let lanes = indices
                .into_iter()
                .rev()
                .map(|i| signed_literal(i as u64, 16));
            quote! { _mm512_set_epi16(#(#lanes),*) }
        }
        (128, 32) => {
            let lanes = indices.into_iter().map(|i| signed_literal(i as u64, 32));
            quote! { _mm_setr_epi32(#(#lanes),*) }
        }
        (256, 32) => {
            let lanes = indices.into_iter().map(|i| signed_literal(i as u64, 32));
            quote! { _mm256_setr_epi32(#(#lanes),*) }
        }
        (512, 32) => {
            let lanes = indices.into_iter().map(|i| signed_literal(i as u64, 32));
            quote! { _mm512_setr_epi32(#(#lanes),*) }
        }
        (128, 64) => {
            let mut lanes = indices
                .into_iter()
                .map(|i| signed_literal(i as u64, 64))
                .collect::<Vec<_>>();
            lanes.reverse();
            quote! { _mm_set_epi64x(#(#lanes),*) }
        }
        (256, 64) => {
            let lanes = indices.into_iter().map(|i| signed_literal(i as u64, 64));
            quote! { _mm256_setr_epi64x(#(#lanes),*) }
        }
        (512, 64) => {
            let lanes = indices.into_iter().map(|i| signed_literal(i as u64, 64));
            quote! { _mm512_setr_epi64(#(#lanes),*) }
        }
        _ => unreachable!(),
    }
}

/// A lane-local `pshufb` mask that groups the even elements before the odd elements.
fn narrow_unzip_shuffle_mask(vec_ty: &VecType) -> TokenStream {
    let lane_mask = match vec_ty.scalar_bits {
        8 => quote! { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 },
        16 => quote! { 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 },
        _ => unreachable!(),
    };

    match vec_ty.n_bits() {
        128 => quote! { _mm_setr_epi8(#lane_mask) },
        256 => quote! { _mm256_setr_epi8(#lane_mask, #lane_mask) },
        _ => unreachable!(),
    }
}

fn interleaved_load_indices(len: usize, block_count: usize) -> Vec<usize> {
    let stream_len = len / block_count;
    (0..block_count)
        .flat_map(|stream| (0..stream_len).map(move |i| i * block_count + stream))
        .collect()
}

fn interleaved_store_indices(len: usize, block_count: usize) -> Vec<usize> {
    let stream_len = len / block_count;
    (0..stream_len)
        .flat_map(|i| (0..block_count).map(move |stream| stream * stream_len + i))
        .collect()
}

impl X86 {
    fn handle_count_ones(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        match *self {
            Self::Avx512 => {
                let suffix = format!("epi{}", vec_ty.scalar_bits);
                let popcnt = intrinsic_ident("popcnt", &suffix, vec_ty.n_bits());
                self.kernel_method(op, vec_ty, |token| {
                    quote! { #popcnt(a.into()).simd_into(#token) }
                })
            }
            Self::Sse4_2 | Self::Avx2 => {
                let bits = vec_ty.n_bits();
                let set1_epi8 = intrinsic_ident("set1", "epi8", bits);
                let set1_epi16 = intrinsic_ident("set1", "epi16", bits);
                let setzero = intrinsic_ident("setzero", coarse_type(vec_ty), bits);
                let and = intrinsic_ident("and", coarse_type(vec_ty), bits);
                let shift = intrinsic_ident("srli", "epi16", bits);
                let shuffle = intrinsic_ident("shuffle", "epi8", bits);
                let add = intrinsic_ident("add", "epi8", bits);
                let maddubs = intrinsic_ident("maddubs", "epi16", bits);
                let madd = intrinsic_ident("madd", "epi16", bits);
                let sad = intrinsic_ident("sad", "epu8", bits);
                let lookup = match bits {
                    128 => quote! {
                        _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4)
                    },
                    256 => quote! {
                        _mm256_setr_epi8(
                            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                        )
                    },
                    _ => unreachable!(),
                };
                self.kernel_method(op, vec_ty, |token| {
                    let combine = match vec_ty.scalar_bits {
                        8 => quote! { byte_counts },
                        16 => quote! { #maddubs(byte_counts, #set1_epi8(1)) },
                        32 => quote! {
                            #madd(#maddubs(byte_counts, #set1_epi8(1)), #set1_epi16(1))
                        },
                        64 => quote! { #sad(byte_counts, #setzero()) },
                        _ => unreachable!(),
                    };
                    quote! {
                        let value = a.into();
                        let nibble_mask = #set1_epi8(0x0f);
                        let lookup = #lookup;
                        let low = #and(value, nibble_mask);
                        let high = #and(#shift::<4>(value), nibble_mask);
                        let byte_counts = #add(#shuffle(lookup, low), #shuffle(lookup, high));
                        #combine.simd_into(#token)
                    }
                })
            }
            Self::Sse2 => fallback_method(op, vec_ty),
        }
    }

    fn handle_count_zeros(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        match *self {
            Self::Sse2 => fallback_method(op, vec_ty), // slightly faster than going through the generic code
            Self::Sse4_2 | Self::Avx2 | Self::Avx512 => count_zeros_method(op, vec_ty),
        }
    }

    pub(crate) fn handle_reduce_min_max(
        &self,
        op: Op,
        vec_ty: &VecType,
        lane_op: &str,
    ) -> TokenStream {
        assert_eq!(
            vec_ty.n_bits(),
            128,
            "wide reductions must use the generic 128-bit-grained implementation"
        );
        assert!(
            matches!(lane_op, "min" | "max" | "min_precise" | "max_precise"),
            "unexpected min/max reduction lane operation"
        );
        assert!(
            vec_ty.scalar != ScalarType::Mask,
            "min/max reductions only operate on numeric vectors"
        );

        // x86 has no packed i64/u64 min/max before AVX-512, and a scalar pair
        // is also the shortest fixed-depth reduction when AVX-512VL is present.
        if matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned)
            && vec_ty.scalar_bits == 64
        {
            assert!(
                matches!(lane_op, "min" | "max"),
                "precise integer reductions are forwarded to ordinary reductions"
            );
            let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
            let scalar_op = Ident::new(lane_op, Span::call_site());
            return self.kernel_method(op, vec_ty, |_| {
                quote! {
                    let lanes: [#scalar; 2] = a.into();
                    lanes[0].#scalar_op(lanes[1])
                }
            });
        }

        let mut stages = Vec::new();
        let mut shift_bytes = 8;
        let scalar_bytes = vec_ty.scalar_bits / 8;
        while shift_bytes >= scalar_bytes {
            let shift = Literal::i32_unsuffixed(i32::try_from(shift_bytes).unwrap());
            let shifted = match (vec_ty.scalar, vec_ty.scalar_bits) {
                (ScalarType::Float, 32) => quote! {
                    _mm_castsi128_ps(_mm_srli_si128::<#shift>(_mm_castps_si128(reduced)))
                },
                (ScalarType::Float, 64) => quote! {
                    _mm_castsi128_pd(_mm_srli_si128::<#shift>(_mm_castpd_si128(reduced)))
                },
                (ScalarType::Int | ScalarType::Unsigned, _) => {
                    quote! { _mm_srli_si128::<#shift>(reduced) }
                }
                _ => unreachable!("min/max reductions only operate on numeric vectors"),
            };
            let combine = x86_reduce_min_max_expr(*self, lane_op, vec_ty);
            stages.push(quote! {
                let shifted = #shifted;
                let reduced = { #combine };
            });
            shift_bytes /= 2;
        }

        let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
        let len = vec_ty.len;
        let arch_ty = self.arch_ty(vec_ty);
        let result = match (vec_ty.scalar, vec_ty.scalar_bits) {
            (ScalarType::Float, 32) => quote! { _mm_cvtss_f32(reduced) },
            (ScalarType::Float, 64) => quote! { _mm_cvtsd_f64(reduced) },
            (ScalarType::Int | ScalarType::Unsigned, _) => quote! {
                {
                    let lanes: [#scalar; #len] =
                        crate::transmute::checked_transmute_copy(&reduced);
                    lanes[0]
                }
            },
            _ => unreachable!("min/max reductions only operate on numeric vectors"),
        };

        self.kernel_method(op, vec_ty, |_| {
            quote! {
                let reduced: #arch_ty = a.into();
                #(#stages)*
                #result
            }
        })
    }

    pub(crate) fn handle_reduce_sum(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        assert_eq!(
            vec_ty.n_bits(),
            128,
            "wide reductions must use the generic 128-bit-grained implementation"
        );

        match (vec_ty.scalar, vec_ty.scalar_bits) {
            (ScalarType::Float, 32) => self.kernel_method(op, vec_ty, |_| {
                quote! {
                    // _mm_hadd_ps is slower than shuffle followed by add
                    let a: __m128 = a.into();
                    let adjacent = _mm_add_ps(a, _mm_shuffle_ps::<0b10_11_00_01>(a, a));
                    _mm_cvtss_f32(_mm_add_ss(adjacent, _mm_movehl_ps(adjacent, adjacent)))
                }
            }),
            (ScalarType::Float, 64) => self.kernel_method(op, vec_ty, |_| {
                quote! {
                    let a: __m128d = a.into();
                    _mm_cvtsd_f64(_mm_add_sd(a, _mm_unpackhi_pd(a, a)))
                }
            }),
            (ScalarType::Int | ScalarType::Unsigned, _) => {
                let add = simple_sign_unaware_intrinsic("add", vec_ty);
                let scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
                let len = vec_ty.len;
                let mut stages = Vec::new();
                let mut shift_bytes = 8;
                let scalar_bytes = vec_ty.scalar_bits / 8;
                while shift_bytes >= scalar_bytes {
                    let shift = Literal::i32_unsuffixed(i32::try_from(shift_bytes).unwrap());
                    stages.push(quote! {
                        let sum = #add(sum, _mm_srli_si128::<#shift>(sum));
                    });
                    shift_bytes /= 2;
                }
                self.kernel_method(op, vec_ty, |_| {
                    quote! {
                        let sum: __m128i = a.into();
                        #(#stages)*
                        let lanes: [#scalar; #len] =
                            crate::transmute::checked_transmute_copy(&sum);
                        lanes[0]
                    }
                })
            }
            _ => unreachable!("reduce_sum only operates on numeric vectors"),
        }
    }

    pub(crate) fn handle_splat(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Mask {
            let lane_mask = avx512_mask_lane_bits(vec_ty);
            let result = avx512_mask_value(
                vec_ty,
                quote! {
                    if val { #lane_mask } else { 0 }
                },
            );
            let method_sig = op.simd_trait_method_sig(vec_ty);
            return quote! {
                #method_sig {
                    #result
                }
            };
        }
        let intrinsic = set1_intrinsic(vec_ty);
        let cast = match vec_ty.scalar {
            ScalarType::Unsigned => quote!(.cast_signed()),
            _ => quote!(),
        };
        let normalize_mask = integer_lane_mask_splat_arg(vec_ty);
        self.kernel_method(op, vec_ty, |token| {
            quote! {
                #normalize_mask
                #intrinsic(val #cast).simd_into(#token)
            }
        })
    }

    fn has_specialized_mask_from_bitmask(&self, vec_ty: &VecType) -> bool {
        if *self == Self::Avx512 {
            return true;
        }
        self.has_wide_byte_mask_from_bitmask(vec_ty) || self.has_wide_avx2_mask_from_bitmask(vec_ty)
    }

    fn has_wide_byte_mask_from_bitmask(&self, vec_ty: &VecType) -> bool {
        // 512-bit byte masks can be constructed directly from one broadcast, avoiding the
        // shift-and-rebroadcast shape from generic split/combine.
        *self != Self::Sse2
            && vec_ty.scalar == ScalarType::Mask
            && vec_ty.n_bits() == 512
            && vec_ty.scalar_bits == 8
    }

    fn has_wide_avx2_mask_from_bitmask(&self, vec_ty: &VecType) -> bool {
        // AVX2 can construct these 512-bit masks directly from one broadcast, avoiding the
        // split/combine shape that shifts and broadcasts each half separately.
        *self == Self::Avx2
            && vec_ty.scalar == ScalarType::Mask
            && vec_ty.n_bits() == 512
            && matches!(vec_ty.scalar_bits, 32 | 64)
    }

    fn has_specialized_mask_to_bitmask(&self, vec_ty: &VecType) -> bool {
        if *self == Self::Avx512 {
            return true;
        }
        vec_ty.scalar == ScalarType::Mask && vec_ty.scalar_bits == 16
    }

    fn avx512_mask_array_conversion(&self, vec_ty: &VecType) -> TokenStream {
        assert!(
            *self == Self::Avx512,
            "compact mask array conversions are only generated for AVX-512"
        );
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "compact mask array conversions are only generated for mask types"
        );

        let storage = self.arch_storage_ty(vec_ty);
        let storage_assoc = vec_ty.rust();
        let scalar = ScalarType::Int.rust(vec_ty.scalar_bits);
        let len = vec_ty.len;
        let from_array = format_ident!("{}_from_array", vec_ty.rust_name());
        let to_array = format_ident!("{}_to_array", vec_ty.rust_name());
        let movepi_mask = intrinsic_ident(
            &format!("movepi{}", vec_ty.scalar_bits),
            "mask",
            vec_ty.n_bits(),
        );
        let movm = intrinsic_ident(
            "movm",
            op_suffix(vec_ty.scalar, vec_ty.scalar_bits, true),
            vec_ty.n_bits(),
        );

        // Mask arrays are specified as either 0 or -1 per lane, so the sign bit is the
        // truth value. Other lane values have unspecified results.
        quote! {
            #[inline(always)]
            fn #from_array(self, val: [#scalar; #len]) -> Self::#storage_assoc {
                crate::kernel!(
                    #[inline(always)]
                    fn kernel(_token: Avx512, val: [#scalar; #len]) -> #storage {
                        let lanes = crate::transmute::checked_transmute_copy(&val);
                        #movepi_mask(lanes)
                    }
                );
                kernel(self, val)
            }

            #[inline(always)]
            fn #to_array(self, val: Self::#storage_assoc) -> [#scalar; #len] {
                crate::kernel!(
                    #[inline(always)]
                    fn kernel(_token: Avx512, val: #storage) -> [#scalar; #len] {
                        let lanes = #movm(val);
                        crate::transmute::checked_transmute_copy(&lanes)
                    }
                );
                kernel(self, val)
            }
        }
    }

    pub(crate) fn handle_avx512_mask_set(
        &self,
        method_sig: TokenStream,
        vec_ty: &VecType,
    ) -> TokenStream {
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "AVX-512 mask set only operates on mask types"
        );
        let len = vec_ty.len;
        let bits = avx512_mask_bits_expr(quote! { a });
        let result = avx512_mask_value(vec_ty, quote! { bits });

        quote! {
            #method_sig {
                assert!(
                    index < #len,
                    "mask lane index {index} is out of bounds for {} lanes",
                    #len
                );
                let bit = 1u64 << index;
                let bits = #bits;
                let bits = if value { bits | bit } else { bits & !bit };
                *a = #result;
            }
        }
    }

    pub(crate) fn handle_mask_rotate_elements(
        &self,
        op: Op,
        vec_ty: &VecType,
        direction: ElementDirection,
    ) -> TokenStream {
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "mask element rotation only operates on masks"
        );

        if *self != Self::Avx512 {
            return integer_lane_mask_rotate(op, vec_ty);
        }

        let method_sig = op.simd_trait_method_sig(vec_ty);
        let len = Literal::usize_unsuffixed(vec_ty.len);

        if vec_ty.len == avx512_mask_register_bits(vec_ty) {
            let rotate = match direction {
                // Lane zero is the low bit, so rotating elements left rotates bits right.
                ElementDirection::Left => quote! { rotate_right },
                ElementDirection::Right => quote! { rotate_left },
            };
            let result =
                avx512_mask_value(vec_ty, quote! { a.val.#rotate((OFFSET % #len) as u32) });
            return quote! {
                #method_sig {
                    #result
                }
            };
        }

        // Two- and four-lane masks both occupy an __mmask8, so rotate within
        // the logical lane width rather than the storage type's eight bits.
        let lane_mask = avx512_mask_lane_bits(vec_ty);
        let input = avx512_mask_bits_expr(quote! { a });
        let rotated = match direction {
            ElementDirection::Left => {
                quote! { ((bits >> offset) | (bits << (#len - offset))) & #lane_mask }
            }
            ElementDirection::Right => {
                quote! { ((bits << offset) | (bits >> (#len - offset))) & #lane_mask }
            }
        };
        let result = avx512_mask_value(vec_ty, quote! { bits });

        quote! {
            #method_sig {
                let bits = #input & #lane_mask;
                let offset = OFFSET % #len;
                let bits = if offset == 0 { bits } else { #rotated };
                #result
            }
        }
    }

    pub(crate) fn handle_mask_from_bitmask(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "mask bitmask conversion only operates on masks"
        );

        if *self == Self::Avx512 {
            let lane_mask = avx512_mask_lane_bits(vec_ty);
            let result = avx512_mask_value(vec_ty, quote! { bits & #lane_mask });
            let method_sig = op.simd_trait_method_sig(vec_ty);
            return quote! {
                #method_sig {
                    #result
                }
            };
        }

        if *self == Self::Sse2 && matches!(vec_ty.scalar_bits, 8 | 64) {
            return generic_mask_from_bitmask(op.simd_trait_method_sig(vec_ty), vec_ty);
        }

        if self.has_wide_byte_mask_from_bitmask(vec_ty) {
            return self.kernel_method(op, vec_ty, |token| {
                mask_from_bitmask_wide_bytes(self.native_width(), vec_ty, token)
            });
        }

        if self.has_wide_avx2_mask_from_bitmask(vec_ty) {
            return self.kernel_method(op, vec_ty, |token| {
                mask_from_bitmask_wide_avx2(vec_ty, token)
            });
        }

        self.kernel_method(op, vec_ty, |token| match vec_ty.scalar_bits {
            8 => {
                let bytes = mask_from_bitmask_bytes(vec_ty);
                quote! {
                    #bytes.simd_into(#token)
                }
            }
            16 | 32 | 64 => {
                let lanes = mask_from_bitmask_lanes(vec_ty);
                quote! {
                    #lanes.simd_into(#token)
                }
            }
            _ => unreachable!(),
        })
    }

    pub(crate) fn handle_mask_to_bitmask(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "mask bitmask conversion only operates on masks"
        );

        if *self == Self::Avx512 {
            let lane_mask = avx512_mask_lane_bits(vec_ty);
            let bits = avx512_mask_bits_expr(quote! { a });
            let method_sig = op.simd_trait_method_sig(vec_ty);
            return quote! {
                #method_sig {
                    #bits & #lane_mask
                }
            };
        }

        match vec_ty.scalar_bits {
            8 => {
                let bits_ty = vec_ty.reinterpret(ScalarType::Int, 8);
                let movemask = simple_intrinsic("movemask", &bits_ty);
                self.kernel_method(op, vec_ty, |_| {
                    quote! { #movemask(a.into()) as u32 as u64 }
                })
            }
            16 => {
                let bits = mask_to_bitmask_words(self.native_width(), vec_ty);
                self.kernel_method(op, vec_ty, |_| bits)
            }
            32 | 64 => {
                let float_ty = vec_ty.cast(ScalarType::Float);
                let movemask = simple_intrinsic("movemask", &float_ty);
                let cast = cast_ident(
                    ScalarType::Mask,
                    ScalarType::Float,
                    vec_ty.scalar_bits,
                    vec_ty.scalar_bits,
                    vec_ty.n_bits(),
                );
                self.kernel_method(op, vec_ty, |_| {
                    quote! { #movemask(#cast(a.into())) as u32 as u64 }
                })
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn handle_compare(&self, op: Op, method: &str, vec_ty: &VecType) -> TokenStream {
        if *self == Self::Avx512 {
            if vec_ty.scalar == ScalarType::Mask {
                let method_sig = op.simd_trait_method_sig(vec_ty);
                let expr = avx512_mask_compare_expr(method, vec_ty);
                let result = avx512_mask_value(vec_ty, expr);
                return quote! {
                    #method_sig {
                        #result
                    }
                };
            }

            return self.kernel_method(op, vec_ty, |token| {
                let mask_ty = vec_ty.mask_ty();
                if vec_ty.scalar == ScalarType::Float {
                    let predicate = avx512_float_compare_predicate(method);
                    let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
                    let intrinsic =
                        intrinsic_ident("cmp", &format!("{suffix}_mask"), vec_ty.n_bits());
                    avx512_mask_register_value_with_simd(
                        &mask_ty,
                        quote! { #intrinsic::<#predicate>(a.into(), b.into()) },
                        quote! { #token },
                    )
                } else {
                    let cmp = avx512_compare_op(method);
                    let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, true);
                    let intrinsic =
                        intrinsic_ident(cmp, &format!("{suffix}_mask"), vec_ty.n_bits());
                    avx512_mask_register_value_with_simd(
                        &mask_ty,
                        quote! { #intrinsic(a.into(), b.into()) },
                        quote! { #token },
                    )
                }
            });
        }

        if *self == Self::Sse2 && vec_ty.scalar != ScalarType::Float {
            if vec_ty.scalar_bits == 64 && method != "simd_eq" {
                return fallback_method(op, vec_ty);
            }

            let expr = sse2_int_compare_expr(method, vec_ty);
            return self.kernel_method(op, vec_ty, |token| {
                quote! { #expr.simd_into(#token) }
            });
        }

        if vec_ty.scalar_bits == 64
            && matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned)
            && method != "simd_eq"
        {
            return fallback_method(op, vec_ty);
        }

        let args = [quote! { a.into() }, quote! { b.into() }];

        let expr = if vec_ty.scalar != ScalarType::Float {
            match method {
                "simd_le" | "simd_ge" => {
                    let max_min = match method {
                        "simd_le" => "min",
                        "simd_ge" => "max",
                        _ => unreachable!(),
                    };

                    let eq_intrinsic = simple_sign_unaware_intrinsic("cmpeq", vec_ty);

                    let max_min_expr = x86::expr(max_min, vec_ty, &args);
                    quote! { #eq_intrinsic(#max_min_expr, a.into()) }
                }
                // Below AVX-512 we only have signed GT/LT, not unsigned. We have to emulate it.
                "simd_lt" | "simd_gt" => sse2_int_compare_expr(method, vec_ty),
                "simd_eq" => x86::expr(method, vec_ty, &args),
                _ => unreachable!(),
            }
        } else {
            let compare_op = float_compare_method(method, vec_ty);
            let ident = cast_ident(
                ScalarType::Float,
                ScalarType::Mask,
                vec_ty.scalar_bits,
                vec_ty.scalar_bits,
                vec_ty.n_bits(),
            );
            quote! { #ident(#compare_op(a.into(), b.into())) }
        };

        self.kernel_method(op, vec_ty, |token| {
            quote! { #expr.simd_into(#token) }
        })
    }

    pub(crate) fn handle_unary(
        &self,
        op: Op,
        method_sig: TokenStream,
        method: &str,
        vec_ty: &VecType,
    ) -> TokenStream {
        if method == "reverse" {
            if vec_ty.scalar == ScalarType::Mask {
                if *self == Self::Avx512 {
                    let shift = avx512_mask_register_bits(vec_ty) - vec_ty.len;
                    let result =
                        avx512_mask_value(vec_ty, quote! { a.val.reverse_bits() >> #shift });
                    return quote! {
                        #method_sig {
                            #result
                        }
                    };
                }
                return reverse_vector_mask_method(op, vec_ty);
            }
            return if *self == Self::Sse2 {
                fallback_method(op, vec_ty)
            } else {
                reverse_method(op, vec_ty)
            };
        }

        if method == "count_zeros" {
            return self.handle_count_zeros(op, vec_ty);
        }

        if method == "count_ones" {
            return self.handle_count_ones(op, vec_ty);
        }

        if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Mask {
            let body = match method {
                "not" => {
                    let lane_mask = avx512_mask_lane_bits(vec_ty);
                    let bits = avx512_mask_bits_expr(quote! { a });
                    let result = avx512_mask_value(vec_ty, quote! { (!#bits) & #lane_mask });
                    quote! { #result }
                }
                _ => unreachable!(),
            };
            return quote! {
                #method_sig {
                    #body
                }
            };
        }

        if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Float {
            match method {
                "floor" | "ceil" | "round_ties_even" | "trunc" if vec_ty.n_bits() == 512 => {
                    let intrinsic = intrinsic_ident(
                        "roundscale",
                        op_suffix(vec_ty.scalar, vec_ty.scalar_bits, true),
                        vec_ty.n_bits(),
                    );
                    let rounding_mode = match method {
                        "floor" => quote! { _MM_FROUND_TO_NEG_INF },
                        "ceil" => quote! { _MM_FROUND_TO_POS_INF },
                        "round_ties_even" => quote! { _MM_FROUND_TO_NEAREST_INT },
                        "trunc" => quote! { _MM_FROUND_TO_ZERO },
                        _ => unreachable!(),
                    };
                    return self.kernel_method(op, vec_ty, |token| {
                        quote! {
                            #intrinsic::<{ #rounding_mode | _MM_FROUND_NO_EXC }>(a.into()).simd_into(#token)
                        }
                    });
                }
                "approximate_recip" => {
                    let intrinsic = intrinsic_ident(
                        "rcp14",
                        op_suffix(vec_ty.scalar, vec_ty.scalar_bits, true),
                        vec_ty.n_bits(),
                    );
                    return self.kernel_method(op, vec_ty, |token| {
                        quote! {
                            #intrinsic(a.into()).simd_into(#token)
                        }
                    });
                }
                _ => {}
            }
        }

        if *self == Self::Sse2
            && vec_ty.scalar == ScalarType::Float
            && matches!(method, "floor" | "ceil" | "round_ties_even" | "trunc")
        {
            return fallback_method(op, vec_ty);
        }

        match method {
            "fract" => {
                let trunc_op = generic_op_name("trunc", vec_ty);
                quote! {
                    #method_sig {
                        a - self.#trunc_op(a)
                    }
                }
            }
            "approximate_recip" if vec_ty.scalar_bits == 64 => {
                quote! {
                    #method_sig {
                        1.0 / a
                    }
                }
            }
            "not" if vec_ty.scalar == ScalarType::Mask => {
                let xor_op = generic_op_name("xor", vec_ty);
                let splat_op = generic_op_name("splat", vec_ty);
                quote! {
                    #method_sig {
                        self.#xor_op(a, self.#splat_op(true))
                    }
                }
            }
            "not" => {
                quote! {
                    #method_sig {
                        a ^ !0
                    }
                }
            }
            _ => {
                let args = [quote! { a.into() }];
                let expr = x86::expr(method, vec_ty, &args);
                self.kernel_method(op, vec_ty, |token| {
                    quote! { #expr.simd_into(#token) }
                })
            }
        }
    }

    fn handle_widen(&self, op: Op, vec_ty: &VecType, target_ty: VecType) -> TokenStream {
        self.kernel_method(op, vec_ty, |token| {
            match (*self, vec_ty.scalar, vec_ty.scalar_bits, vec_ty.n_bits()) {
                (_, ScalarType::Float, 32, 128) => quote! {
                    let raw = a.into();
                    (
                        _mm_cvtps_pd(raw).simd_into(#token),
                        _mm_cvtps_pd(_mm_movehl_ps(raw, raw)).simd_into(#token),
                    )
                },
                (_, ScalarType::Float, 32, 256) => quote! {
                    let raw = a.into();
                    (
                        _mm256_cvtps_pd(_mm256_castps256_ps128(raw)).simd_into(#token),
                        _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(raw)).simd_into(#token),
                    )
                },
                (_, ScalarType::Float, 32, 512) => quote! {
                    let raw = a.into();
                    (
                        _mm512_cvtps_pd(_mm512_castps512_ps256(raw)).simd_into(#token),
                        _mm512_cvtps_pd(_mm512_extractf32x8_ps::<1>(raw)).simd_into(#token),
                    )
                },
                (Self::Sse2, ScalarType::Unsigned, 8 | 16 | 32, 128) => {
                    let unpack_low =
                        unpack_intrinsic(vec_ty.scalar, vec_ty.scalar_bits, true, vec_ty.n_bits());
                    let unpack_high =
                        unpack_intrinsic(vec_ty.scalar, vec_ty.scalar_bits, false, vec_ty.n_bits());
                    quote! {
                        let raw = a.into();
                        let sign = _mm_setzero_si128();
                        (
                            #unpack_low(raw, sign).simd_into(#token),
                            #unpack_high(raw, sign).simd_into(#token),
                        )
                    }
                }
                (Self::Sse2, ScalarType::Int, scalar_bits @ (8 | 16 | 32), 128) => {
                    let unpack_low =
                        unpack_intrinsic(vec_ty.scalar, vec_ty.scalar_bits, true, vec_ty.n_bits());
                    let unpack_high =
                        unpack_intrinsic(vec_ty.scalar, vec_ty.scalar_bits, false, vec_ty.n_bits());
                    let sign = match scalar_bits {
                        8 => quote! { _mm_cmpgt_epi8(_mm_setzero_si128(), raw) },
                        16 => quote! { _mm_srai_epi16::<15>(raw) },
                        32 => quote! { _mm_srai_epi32::<31>(raw) },
                        _ => unreachable!(),
                    };
                    quote! {
                        let raw = a.into();
                        let sign = #sign;
                        (
                            #unpack_low(raw, sign).simd_into(#token),
                            #unpack_high(raw, sign).simd_into(#token),
                        )
                    }
                }
                (
                    Self::Sse4_2 | Self::Avx2 | Self::Avx512,
                    ScalarType::Unsigned | ScalarType::Int,
                    8 | 16 | 32,
                    128,
                ) => {
                    let extend = extend_intrinsic(
                        vec_ty.scalar,
                        vec_ty.scalar_bits,
                        target_ty.scalar_bits,
                        target_ty.n_bits(),
                    );
                    quote! {
                        let raw = a.into();
                        (
                            #extend(raw).simd_into(#token),
                            #extend(_mm_srli_si128::<8>(raw)).simd_into(#token),
                        )
                    }
                }
                (
                    Self::Avx2 | Self::Avx512,
                    ScalarType::Unsigned | ScalarType::Int,
                    8 | 16 | 32,
                    256,
                ) => {
                    let extend = extend_intrinsic(
                        vec_ty.scalar,
                        vec_ty.scalar_bits,
                        target_ty.scalar_bits,
                        target_ty.n_bits(),
                    );
                    quote! {
                        let raw = a.into();
                        (
                            #extend(_mm256_castsi256_si128(raw)).simd_into(#token),
                            #extend(_mm256_extracti128_si256::<1>(raw)).simd_into(#token),
                        )
                    }
                }
                (Self::Avx512, ScalarType::Unsigned | ScalarType::Int, 8 | 16 | 32, 512) => {
                    let extend = extend_intrinsic(
                        vec_ty.scalar,
                        vec_ty.scalar_bits,
                        target_ty.scalar_bits,
                        target_ty.n_bits(),
                    );
                    quote! {
                        let raw = a.into();
                        (
                            #extend(_mm512_castsi512_si256(raw)).simd_into(#token),
                            #extend(_mm512_extracti64x4_epi64::<1>(raw)).simd_into(#token),
                        )
                    }
                }
                _ => unreachable!(),
            }
        })
    }

    fn handle_narrow(
        &self,
        op: Op,
        vec_ty: &VecType,
        target_ty: VecType,
        mode: NarrowingMode,
    ) -> TokenStream {
        use NarrowingMode::{Relaxed, Saturate, Wrap};

        if mode == Relaxed {
            // SSE4.2 and AVX2 only have saturating narrowing instructions for i32 and i16
            let implementation = if *self != Self::Avx512
                && vec_ty.scalar == ScalarType::Int
                && matches!(vec_ty.scalar_bits, 16 | 32)
            {
                "saturating_narrow"
            } else {
                // for everything else (SSE2, AVX-512, unsigned values) truncation is cheaper
                "narrow"
            };
            return relaxed_narrow_method(op, vec_ty, target_ty, implementation);
        }

        // Restore sequential lane order after AVX2 pack instructions interleave results within
        // their two 128-bit lanes.
        let reorder_avx2 = |packed: TokenStream, token: &Ident| {
            quote! {
                _mm256_permute4x64_epi64::<0xd8>(#packed).simd_into(#token)
            }
        };

        // Select the low or high 32-bit half of each 64-bit lane and compact those halves from
        // both inputs, using the instruction sequence appropriate for the vector width.
        let compact_dwords = |high: bool, a: TokenStream, b: TokenStream| {
            let shuffle = if high {
                quote! { 0xdd }
            } else {
                quote! { 0x88 }
            };
            match vec_ty.n_bits() {
                128 => quote! {
                    _mm_castps_si128(_mm_shuffle_ps::<#shuffle>(
                        _mm_castsi128_ps(#a),
                        _mm_castsi128_ps(#b),
                    ))
                },
                256 => quote! {
                    _mm256_permute4x64_epi64::<0xd8>(
                        _mm256_castps_si256(_mm256_shuffle_ps::<#shuffle>(
                            _mm256_castsi256_ps(#a),
                            _mm256_castsi256_ps(#b),
                        )),
                    )
                },
                _ => unreachable!(),
            }
        };

        match (
            *self,
            mode,
            vec_ty.scalar,
            vec_ty.scalar_bits,
            vec_ty.n_bits(),
        ) {
            (_, Saturate, ScalarType::Float, 64, 128 | 256 | 512) => {
                // Saturating conversion for floats doesn't make sense, floats always follow IEEE rounding.
                // Pass through to the regular conversion.
                let method_sig = op.simd_trait_method_sig(vec_ty);
                let narrow = generic_op_name("narrow", vec_ty);
                quote! {
                    #method_sig {
                        self.#narrow(a, b)
                    }
                }
            }
            (_, Wrap, ScalarType::Float, 64, 128) => self.kernel_method(op, vec_ty, |token| {
                quote! {
                    let low = _mm_cvtpd_ps(a.into());
                    let high = _mm_cvtpd_ps(b.into());
                    _mm_movelh_ps(low, high).simd_into(#token)
                }
            }),
            // This function is amenable to split/combine lowering,
            // so this generator is only invoked for e.g. 256-bit vectors on AVX2 and higher
            (_, Wrap, ScalarType::Float, 64, 256) => self.kernel_method(op, vec_ty, |token| {
                quote! {
                    let low = _mm256_cvtpd_ps(a.into());
                    let high = _mm256_cvtpd_ps(b.into());
                    _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(low), high)
                        .simd_into(#token)
                }
            }),
            (_, Wrap, ScalarType::Float, 64, 512) => self.kernel_method(op, vec_ty, |token| {
                quote! {
                    let low = _mm512_cvtpd_ps(a.into());
                    let high = _mm512_cvtpd_ps(b.into());
                    _mm512_insertf32x8::<1>(_mm512_castps256_ps512(low), high)
                        .simd_into(#token)
                }
            }),
            (
                Self::Avx512,
                mode,
                scalar @ (ScalarType::Int | ScalarType::Unsigned),
                scalar_bits @ (16 | 32 | 64),
                n_bits @ (128 | 256 | 512),
            ) => {
                // AVX-512 has hardware instructions for all integer narrowing operations.
                let conversion = match (mode, scalar) {
                    (Wrap, _) => format!("cvtepi{scalar_bits}"),
                    (Saturate, ScalarType::Int) => format!("cvtsepi{}", vec_ty.scalar_bits),
                    (Saturate, ScalarType::Unsigned) => {
                        format!("cvtusepi{}", vec_ty.scalar_bits)
                    }
                    _ => unreachable!(),
                };
                let suffix = format!("epi{}", target_ty.scalar_bits);
                let narrow = intrinsic_ident(&conversion, &suffix, n_bits);
                let combine = match n_bits {
                    128 => quote! { _mm_unpacklo_epi64(low, high) },
                    256 => quote! {
                        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low), high)
                    },
                    512 => quote! {
                        _mm512_inserti64x4::<1>(_mm512_castsi256_si512(low), high)
                    },
                    _ => unreachable!(),
                };
                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        let low = #narrow(a.into());
                        let high = #narrow(b.into());
                        #combine.simd_into(#token)
                    }
                })
            }
            (
                Self::Avx2,
                mode,
                scalar @ (ScalarType::Int | ScalarType::Unsigned),
                scalar_bits @ (16 | 32),
                256,
            ) => self.kernel_method(op, vec_ty, |token| {
                // AVX2 has native instructions for narrowing 16 and 32-bit integers.
                let signed_saturation = mode == Saturate && scalar == ScalarType::Int;
                let pack = pack_intrinsic(scalar_bits, signed_saturation, 256);
                let packed = if signed_saturation {
                    // AVX2 has a native signed integer narrowing instruction.
                    quote! { #pack(a.into(), b.into()) }
                } else {
                    // Everything else needs to be emulated on top of signed integer narrowing via masking.
                    let set1 = set1_intrinsic(vec_ty);
                    let limit = match scalar_bits {
                        16 => quote! { 0xff },
                        32 => quote! { 0xffff },
                        _ => unreachable!(),
                    };
                    let (prepare, limit_name) = match mode {
                        Saturate => (simple_intrinsic("min", vec_ty), format_ident!("max")),
                        Wrap => (intrinsic_ident("and", "si256", 256), format_ident!("mask")),
                        Relaxed => unreachable!(),
                    };
                    quote! {{
                        let #limit_name = #set1(#limit);
                        #pack(
                            #prepare(a.into(), #limit_name),
                            #prepare(b.into(), #limit_name),
                        )
                    }}
                };
                reorder_avx2(packed, token)
            }),
            (Self::Avx2, Wrap, ScalarType::Int | ScalarType::Unsigned, 64, 256) => self
                .kernel_method(op, vec_ty, |token| {
                    // Non-saturating narrowing is just a shuffle.
                    let low = compact_dwords(false, quote! { a.into() }, quote! { b.into() });
                    quote! {
                        #low.simd_into(#token)
                    }
                }),
            (Self::Avx2, Saturate, ScalarType::Int, 64, 256) => {
                self.kernel_method(op, vec_ty, |token| {
                    // Saturating narrowing needs to be emulated for lack of hardware instructions.
                    let low = compact_dwords(false, quote! { a }, quote! { b });
                    let high = compact_dwords(true, quote! { a }, quote! { b });
                    quote! {
                        let a = a.into();
                        let b = b.into();
                        let low = #low;
                        let high = #high;
                        let low_sign = _mm256_srai_epi32::<31>(low);
                        let fits = _mm256_cmpeq_epi32(high, low_sign);
                        let high_sign = _mm256_srai_epi32::<31>(high);
                        let bound = _mm256_xor_si256(
                            high_sign,
                            _mm256_set1_epi32(i32::MAX),
                        );
                        _mm256_blendv_epi8(bound, low, fits).simd_into(#token)
                    }
                })
            }
            (Self::Avx2, Saturate, ScalarType::Unsigned, 64, 256) => {
                self.kernel_method(op, vec_ty, |token| {
                    let low = compact_dwords(false, quote! { a }, quote! { b });
                    quote! {
                        let zero = _mm256_setzero_si256();
                        let ones = _mm256_cmpeq_epi64(zero, zero);
                        let a = a.into();
                        let b = b.into();
                        let a_fits = _mm256_cmpeq_epi64(
                            _mm256_srli_epi64::<32>(a),
                            zero,
                        );
                        let b_fits = _mm256_cmpeq_epi64(
                            _mm256_srli_epi64::<32>(b),
                            zero,
                        );
                        let a = _mm256_or_si256(a, _mm256_xor_si256(a_fits, ones));
                        let b = _mm256_or_si256(b, _mm256_xor_si256(b_fits, ones));
                        #low.simd_into(#token)
                    }
                })
            }
            (Self::Sse2 | Self::Sse4_2 | Self::Avx2, Saturate, ScalarType::Unsigned, 16, 128) => {
                self.kernel_method(op, vec_ty, |token| {
                    let pack = pack_intrinsic(16, false, vec_ty.n_bits());
                    quote! {
                        let max = _mm_set1_epi16(u8::MAX as i16);
                        let a = a.into();
                        let b = b.into();
                        let a = _mm_sub_epi16(a, _mm_subs_epu16(a, max));
                        let b = _mm_sub_epi16(b, _mm_subs_epu16(b, max));
                        #pack(a, b).simd_into(#token)
                    }
                })
            }
            (Self::Sse4_2 | Self::Avx2, Saturate, ScalarType::Unsigned, 32, 128) => self
                .kernel_method(op, vec_ty, |token| {
                    let pack = pack_intrinsic(32, false, vec_ty.n_bits());
                    quote! {
                        let max = _mm_set1_epi32(u16::MAX as i32);
                        #pack(
                            _mm_min_epu32(a.into(), max),
                            _mm_min_epu32(b.into(), max),
                        ).simd_into(#token)
                    }
                }),
            (Self::Sse2 | Self::Sse4_2 | Self::Avx2, Saturate, ScalarType::Int, 16 | 32, 128) => {
                self.kernel_method(op, vec_ty, |token| {
                    let pack = pack_intrinsic(vec_ty.scalar_bits, true, vec_ty.n_bits());
                    quote! { #pack(a.into(), b.into()).simd_into(#token) }
                })
            }
            (
                Self::Sse4_2 | Self::Avx2,
                Saturate,
                scalar @ (ScalarType::Int | ScalarType::Unsigned),
                64,
                128,
            ) => self.kernel_method(op, vec_ty, |token| {
                let low = compact_dwords(false, quote! { a }, quote! { b });
                let high = compact_dwords(true, quote! { a }, quote! { b });
                let saturate = match scalar {
                    ScalarType::Int => quote! {
                        let low_sign = _mm_srai_epi32::<31>(low);
                        let fits = _mm_cmpeq_epi32(high, low_sign);
                        let high_sign = _mm_srai_epi32::<31>(high);
                        let bound = _mm_xor_si128(high_sign, _mm_set1_epi32(i32::MAX));
                        _mm_blendv_epi8(bound, low, fits)
                    },
                    ScalarType::Unsigned => quote! {
                        let zero = _mm_setzero_si128();
                        let fits = _mm_cmpeq_epi32(high, zero);
                        let ones = _mm_cmpeq_epi32(zero, zero);
                        _mm_or_si128(low, _mm_xor_si128(fits, ones))
                    },
                    _ => unreachable!(),
                };
                quote! {
                    let a = a.into();
                    let b = b.into();
                    let low = #low;
                    let high = #high;
                    #saturate.simd_into(#token)
                }
            }),
            (
                Self::Sse2 | Self::Sse4_2 | Self::Avx2,
                Wrap,
                ScalarType::Int | ScalarType::Unsigned,
                16,
                128,
            ) => self.kernel_method(op, vec_ty, |token| {
                let set1 = set1_intrinsic(vec_ty);
                let pack = pack_intrinsic(16, false, vec_ty.n_bits());
                quote! {
                    let mask = #set1(0xff);
                    #pack(
                        _mm_and_si128(a.into(), mask),
                        _mm_and_si128(b.into(), mask),
                    ).simd_into(#token)
                }
            }),
            (Self::Sse4_2 | Self::Avx2, Wrap, ScalarType::Int | ScalarType::Unsigned, 32, 128) => {
                self.kernel_method(op, vec_ty, |token| {
                    let pack = pack_intrinsic(32, false, vec_ty.n_bits());
                    quote! {
                        let mask = _mm_set1_epi32(0xffff);
                        #pack(
                            _mm_and_si128(a.into(), mask),
                            _mm_and_si128(b.into(), mask),
                        ).simd_into(#token)
                    }
                })
            }
            (
                Self::Sse2 | Self::Sse4_2 | Self::Avx2,
                Wrap,
                ScalarType::Int | ScalarType::Unsigned,
                64,
                128,
            ) => self.kernel_method(op, vec_ty, |token| {
                let low = compact_dwords(false, quote! { a.into() }, quote! { b.into() });
                quote! {
                    #low.simd_into(#token)
                }
            }),
            // SSE2 autovectorizes acceptably and we don't care to spend the complexity budget optimizing it
            (Self::Sse2, Saturate, ScalarType::Unsigned, 32, 128)
            | (Self::Sse2, Saturate, ScalarType::Int | ScalarType::Unsigned, 64, 128)
            | (Self::Sse2, Wrap, ScalarType::Int | ScalarType::Unsigned, 32, 128) => {
                fallback_method(op, vec_ty)
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn handle_binary(&self, op: Op, method: &str, vec_ty: &VecType) -> TokenStream {
        let method_sig = op.simd_trait_method_sig(vec_ty);

        if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Mask {
            let lane_mask = avx512_mask_lane_bits(vec_ty);
            let a_bits = avx512_mask_bits_expr(quote! { a });
            let b_bits = avx512_mask_bits_expr(quote! { b });
            let expr = match method {
                "and" => quote! { (#a_bits & #b_bits) & #lane_mask },
                "or" => quote! { (#a_bits | #b_bits) & #lane_mask },
                "xor" => quote! { (#a_bits ^ #b_bits) & #lane_mask },
                _ => unreachable!(),
            };
            let result = avx512_mask_value(vec_ty, expr);
            return quote! {
                #method_sig {
                    #result
                }
            };
        }

        if *self == Self::Avx512
            && vec_ty.scalar == ScalarType::Float
            && matches!(method, "min_precise" | "max_precise")
        {
            let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, true);
            let range = intrinsic_ident("range", suffix, vec_ty.n_bits());
            let imm = if method == "max_precise" {
                0b0101
            } else {
                0b0100
            };
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    #range::<#imm>(a.into(), b.into()).simd_into(#token)
                }
            });
        }

        if *self == Self::Sse2
            && vec_ty.scalar == ScalarType::Float
            && matches!(method, "min_precise" | "max_precise")
        {
            let intrinsic = simple_intrinsic(
                if method == "max_precise" {
                    "max"
                } else {
                    "min"
                },
                vec_ty,
            );
            let cmpunord = float_compare_method("unord", vec_ty);
            return self.kernel_method(op, vec_ty, |token| {
                let expr = sse2_select_expr(
                    vec_ty,
                    quote! { b_is_nan },
                    quote! { a },
                    quote! { intermediate },
                );
                quote! {
                    let a = a.into();
                    let b = b.into();
                    let intermediate = #intrinsic(a, b);
                    let b_is_nan = #cmpunord(b, b);
                    #expr.simd_into(#token)
                }
            });
        }

        if *self != Self::Avx512
            && vec_ty.scalar_bits == 64
            && matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned)
            && matches!(method, "mul" | "min" | "max")
        {
            return fallback_method(op, vec_ty);
        }

        if *self == Self::Sse2
            && matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned)
            && matches!(method, "min" | "max")
        {
            let expr = sse2_min_max_expr(method, vec_ty);
            return self.kernel_method(op, vec_ty, |token| {
                quote! { #expr.simd_into(#token) }
            });
        }

        if *self == Self::Sse2
            && matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned)
            && method == "mul"
            && vec_ty.scalar_bits == 32
        {
            return fallback_method(op, vec_ty);
        }

        match method {
            "shrv"
                if *self == Self::Avx2
                    && vec_ty.scalar == ScalarType::Int
                    && vec_ty.scalar_bits == 64 =>
            {
                // AVX2 has no packed arithmetic right shift for 64-bit lanes. For shift counts in
                // 0..=63, biasing the logical result reconstructs the sign extension:
                //     arithmetic = ((value >> count) ^ (MIN >> count)) - (MIN >> count)
                let set1 = set1_intrinsic(vec_ty);
                let srlv = intrinsic_ident("srlv", "epi64", vec_ty.n_bits());
                let xor = intrinsic_ident("xor", coarse_type(vec_ty), vec_ty.n_bits());
                let sub = intrinsic_ident("sub", "epi64", vec_ty.n_bits());
                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        let value = a.into();
                        let counts = b.into();
                        let bias = #set1(i64::MIN);
                        let shifted_bias = #srlv(bias, counts);
                        let shifted = #srlv(value, counts);
                        #sub(#xor(shifted, shifted_bias), shifted_bias).simd_into(#token)
                    }
                })
            }
            "shrv"
                if *self != Self::Avx512
                    && vec_ty.scalar == ScalarType::Int
                    && vec_ty.scalar_bits == 64 =>
            {
                fallback_method(op, vec_ty)
            }
            "shlv" | "shrv"
                if *self == Self::Avx512
                    && matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned)
                    && matches!(vec_ty.scalar_bits, 8 | 16) =>
            {
                self.kernel_method(op, vec_ty, |token| {
                    self.handle_avx512_narrow_variable_shift(method, vec_ty, token)
                })
            }
            "shlv" | "shrv"
                if !(matches!(self, Self::Avx2 | Self::Avx512) && vec_ty.scalar_bits >= 32) =>
            {
                // x86 only has lane-wise variable shifts for wider lanes starting at AVX2.
                fallback_method(op, vec_ty)
            }
            _ => self.kernel_method(op, vec_ty, |token| match method {
                "mul" if vec_ty.scalar_bits == 8 => {
                    // https://stackoverflow.com/questions/8193601/sse-multiplication-16-x-uint8-t
                    let mullo = intrinsic_ident("mullo", "epi16", vec_ty.n_bits());
                    let set1 = intrinsic_ident("set1", "epi16", vec_ty.n_bits());
                    let and = intrinsic_ident("and", coarse_type(vec_ty), vec_ty.n_bits());
                    let or = intrinsic_ident("or", coarse_type(vec_ty), vec_ty.n_bits());
                    let slli = intrinsic_ident("slli", "epi16", vec_ty.n_bits());
                    if *self == Self::Sse2 {
                        let srli = intrinsic_ident("srli", "epi16", vec_ty.n_bits());
                        quote! {
                            let dst_even = #mullo(a.into(), b.into());
                            let dst_odd = #mullo(#srli::<8>(a.into()), #srli::<8>(b.into()));

                            #or(#slli(dst_odd, 8), #and(dst_even, #set1(0xFF))).simd_into(#token)
                        }
                    } else {
                        // LLVM's byte-multiplication lowering uses PMADDUBSW to calculate the
                        // odd byte of every i16 lane. The cleared adjacent byte means there is
                        // only one product per lane, so the saturating add cannot saturate.
                        // LLVM only uses this for 128-bit vectors while we apply it everywhere.
                        // Both llvm-mca and hardware benchmarks show this is beneficial.
                        let andnot =
                            intrinsic_ident("andnot", coarse_type(vec_ty), vec_ty.n_bits());
                        let maddubs = intrinsic_ident("maddubs", "epi16", vec_ty.n_bits());
                        quote! {
                            let a = a.into();
                            let b = b.into();
                            let low_mask = #set1(0xFF);
                            let dst_even = #mullo(a, b);
                            let dst_odd = #maddubs(a, #andnot(low_mask, b));

                            #or(#slli(dst_odd, 8), #and(dst_even, low_mask)).simd_into(#token)
                        }
                    }
                }
                "shlv" | "shrv" => {
                    let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
                    let name = match (method, vec_ty.scalar) {
                        ("shrv", ScalarType::Int) => "srav",
                        ("shrv", _) => "srlv",
                        ("shlv", _) => "sllv",
                        _ => unreachable!(),
                    };
                    let intrinsic = intrinsic_ident(name, suffix, vec_ty.n_bits());
                    quote! {
                        #intrinsic(a.into(), b.into()).simd_into(#token)
                    }
                }
                _ => {
                    let args = [quote! { a.into() }, quote! { b.into() }];
                    let expr = x86::expr(method, vec_ty, &args);
                    quote! {
                        #expr.simd_into(#token)
                    }
                }
            }),
        }
    }

    fn handle_avx512_narrow_variable_shift(
        &self,
        method: &str,
        vec_ty: &VecType,
        token: &Ident,
    ) -> TokenStream {
        assert!(
            *self == Self::Avx512,
            "narrow variable shifts are specialized for AVX-512"
        );
        assert!(
            matches!(vec_ty.scalar_bits, 8 | 16),
            "narrow variable shifts only handle 8-bit and 16-bit lanes"
        );
        let name = match (method, vec_ty.scalar) {
            ("shrv", ScalarType::Int) => "srav",
            ("shrv", _) => "srlv",
            ("shlv", _) => "sllv",
            _ => unreachable!(),
        };
        let shift_intrinsic = intrinsic_ident(name, "epi16", vec_ty.n_bits());

        if vec_ty.scalar_bits == 16 {
            return quote! {
                #shift_intrinsic(a.into(), b.into()).simd_into(#token)
            };
        }

        let ty_bits = vec_ty.n_bits();
        let and = intrinsic_ident("and", coarse_type(vec_ty), ty_bits);
        let set1_epi16 = intrinsic_ident("set1", "epi16", ty_bits);

        if method == "shlv" {
            // AVX-512 has variable shifts for i16 lanes but not i8 lanes. Treat each pair
            // of bytes as an i16 and shift its even and odd bytes along separate paths.
            let andnot = intrinsic_ident("andnot", coarse_type(vec_ty), ty_bits);
            let srli = intrinsic_ident("srli", "epi16", ty_bits);
            let blend = avx512_mask_blend_intrinsic(vec_ty);
            let odd_byte_mask = match ty_bits {
                128 => quote! { 0xaaaa_u16 },
                256 => quote! { 0xaaaa_aaaa_u32 },
                512 => quote! { 0xaaaa_aaaa_aaaa_aaaa_u64 },
                _ => unreachable!(),
            };

            return quote! {
                let val = a.into();
                let counts = b.into();
                let byte_mask = #set1_epi16(0x00ff);
                let lo_counts = #and(counts, byte_mask);
                let hi_counts = #srli::<8>(counts);
                // Left shifts cannot move bits from the odd byte into the even byte, so
                // only the odd-byte path needs its adjacent input byte cleared.
                let lo_shifted = #shift_intrinsic(val, lo_counts);
                let hi_values = #andnot(byte_mask, val);
                let hi_shifted = #shift_intrinsic(hi_values, hi_counts);
                // Mask bits set in odd byte positions select `hi_shifted`.
                #blend(#odd_byte_mask, lo_shifted, hi_shifted).simd_into(#token)
            };
        }

        let unpack_hi = unpack_intrinsic(ScalarType::Int, 8, false, ty_bits);
        let unpack_lo = unpack_intrinsic(ScalarType::Int, 8, true, ty_bits);
        let set0 = intrinsic_ident("setzero", coarse_type(vec_ty), ty_bits);
        let pack = pack_intrinsic(16, false, ty_bits);
        let value_extend = match (method, vec_ty.scalar) {
            ("shlv", _) | (_, ScalarType::Unsigned) => quote! { zero },
            ("shrv", ScalarType::Int) if ty_bits == 512 => {
                quote! { _mm512_movm_epi8(_mm512_cmpgt_epi8_mask(zero, val)) }
            }
            ("shrv", ScalarType::Int) => {
                let cmpgt = intrinsic_ident("cmpgt", "epi8", ty_bits);
                quote! { #cmpgt(zero, val) }
            }
            _ => unreachable!(),
        };

        quote! {
            let val = a.into();
            let counts = b.into();
            let zero = #set0();
            let value_extend = #value_extend;
            let lo_values = #unpack_lo(val, value_extend);
            let hi_values = #unpack_hi(val, value_extend);
            let lo_counts = #unpack_lo(counts, zero);
            let hi_counts = #unpack_hi(counts, zero);
            let byte_mask = #set1_epi16(0x00ff);
            let lo_shifted = #and(#shift_intrinsic(lo_values, lo_counts), byte_mask);
            let hi_shifted = #and(#shift_intrinsic(hi_values, hi_counts), byte_mask);
            #pack(lo_shifted, hi_shifted).simd_into(#token)
        }
    }

    fn shl_8bit(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let ty_bits = vec_ty.n_bits();
        let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits.max(16), false);

        let and = intrinsic_ident("and", coarse_type(vec_ty), ty_bits);
        let set1_epi8 = intrinsic_ident("set1", "epi8", ty_bits);
        let shift_intrinsic = intrinsic_ident("sll", suffix, ty_bits);

        self.kernel_method(op, vec_ty, |token| {
            quote! {
                let val = a.into();
                let shift_count = _mm_cvtsi32_si128(shift.cast_signed());
                let mask_byte = 0xff_u32.wrapping_shr(shift) as i8;
                let byte_mask = #set1_epi8(mask_byte);
                #shift_intrinsic(#and(val, byte_mask), shift_count).simd_into(#token)
            }
        })
    }

    fn shr_8bit(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let ty_bits = vec_ty.n_bits();
        let and = intrinsic_ident("and", coarse_type(vec_ty), ty_bits);
        let set1_epi8 = intrinsic_ident("set1", "epi8", ty_bits);
        let shift_intrinsic = intrinsic_ident("srl", "epi16", ty_bits);
        let shift_fixup = match vec_ty.scalar {
            ScalarType::Unsigned => quote! {
                #and(shifted, byte_mask)
            },
            // Signed integers require special handling to implement sign extension
            ScalarType::Int => {
                let set0 = intrinsic_ident("setzero", coarse_type(vec_ty), ty_bits);
                let sign = if *self == Self::Avx512 && ty_bits == 512 {
                    quote! {
                        // there is no _mm512_cmpgt_epi8, so we need to use _mask variant
                        // and expand it back into a vector so that ternarylogic can use it
                        _mm512_movm_epi8(_mm512_cmpgt_epi8_mask(#set0(), val))
                    }
                } else {
                    let cmpgt = intrinsic_ident("cmpgt", "epi8", ty_bits);
                    quote! { #cmpgt(#set0(), val) }
                };
                if *self == Self::Avx512 {
                    // GFNI formulation was also considered, but it regresses latency:
                    // https://github.com/linebender/fearless_simd/pull/291#issuecomment-5107747999
                    let ternary = intrinsic_ident("ternarylogic", "epi32", ty_bits);
                    quote! {
                        let sign = #sign;
                        // Select shifted bits where the byte mask is set and sign bits elsewhere.
                        #ternary::<0xca>(byte_mask, shifted, sign)
                    }
                } else {
                    let andnot = intrinsic_ident("andnot", coarse_type(vec_ty), ty_bits);
                    let or = intrinsic_ident("or", coarse_type(vec_ty), ty_bits);
                    quote! {
                        let sign = #sign;
                        #or(#and(shifted, byte_mask), #andnot(byte_mask, sign))
                    }
                }
            }
            _ => unreachable!(),
        };

        self.kernel_method(op, vec_ty, |token| {
            quote! {
                let val = a.into();
                let shift_count = _mm_cvtsi32_si128(shift.cast_signed());
                let mask_byte = 0xff_u32.wrapping_shr(shift) as i8;
                let byte_mask = #set1_epi8(mask_byte);
                let shifted = #shift_intrinsic(val, shift_count);
                let result = { #shift_fixup };
                result.simd_into(#token)
            }
        })
    }

    pub(crate) fn handle_shift(&self, op: Op, method: &str, vec_ty: &VecType) -> TokenStream {
        if *self != Self::Avx512
            && method == "shr"
            && vec_ty.scalar == ScalarType::Int
            && vec_ty.scalar_bits == 64
        {
            return fallback_method(op, vec_ty);
        }

        if vec_ty.scalar_bits == 8 {
            if method == "shl" {
                self.shl_8bit(op, vec_ty)
            } else {
                self.shr_8bit(op, vec_ty)
            }
        } else {
            let shift_op = match (method, vec_ty.scalar) {
                ("shr", ScalarType::Unsigned) => "srl",
                ("shr", ScalarType::Int) => "sra",
                ("shl", _) => "sll",
                _ => unreachable!(),
            };
            let ty_bits = vec_ty.n_bits();
            let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits.max(16), false);
            let shift_intrinsic = intrinsic_ident(shift_op, suffix, ty_bits);

            self.kernel_method(
                op,
                vec_ty,
                |token| {
                    quote! { #shift_intrinsic(a.into(), _mm_cvtsi32_si128(shift.cast_signed())).simd_into(#token) }
                },
            )
        }
    }

    fn precise_mul_add_f32x4(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        assert!(
            *self == Self::Sse4_2,
            "precise f32 multiply-add emulation is specific to SSE4.2"
        );
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Float,
            "precise f32 multiply-add requires a floating-point vector"
        );
        assert_eq!(
            vec_ty.scalar_bits, 32,
            "precise f32 multiply-add requires 32-bit lanes"
        );
        assert_eq!(
            vec_ty.n_bits(),
            128,
            "SSE4.2 precise f32 multiply-add requires one native vector"
        );

        // Apply Boldo and Melquiond's Theorem 3: turn f32x4 into two f64x2 vectors,
        // conceptually compute a binary64 round-to-odd value of `a * b + c`, then round it
        // to f32. The fast path below skips the round-to-odd correction only where narrowing
        // the binary64 sum cannot suffer a double-rounding error.
        // Its parameters are p=24, k=29, Ew=149, and Ee=1074, satisfying k >= 2 and
        // Ee >= Ew + 2. The theorem, including gradual-underflow cases, is proved in Coq:
        // https://guillaume.melquiond.fr/doc/08-tc.pdf
        self.kernel_method(op, vec_ty, |token| {
            quote! {
                let a: __m128 = a.into();
                let b: __m128 = b.into();
                let c: __m128 = c.into();

                let a_low = _mm_cvtps_pd(a);
                let a_high = _mm_cvtps_pd(_mm_movehl_ps(a, a));
                let b_low = _mm_cvtps_pd(b);
                let b_high = _mm_cvtps_pd(_mm_movehl_ps(b, b));
                let c_low = _mm_cvtps_pd(c);
                let c_high = _mm_cvtps_pd(_mm_movehl_ps(c, c));

                // Every finite f32 product is exactly representable as f64. Every exact finite
                // product-plus-add is a multiple of 2^-298 with magnitude below 2^256, so none of
                // these operations underflow or overflow in f64. Usually the f64 sum can be
                // narrowed directly. For a normal f32 result, the only possible double-rounding
                // error is when that sum is exactly halfway between two adjacent f32 values. This
                // also recognizes the f32 finite/infinity overflow threshold.
                let product_low = _mm_mul_pd(a_low, b_low);
                let product_high = _mm_mul_pd(a_high, b_high);
                let mut sum_low = _mm_add_pd(product_low, c_low);
                let mut sum_high = _mm_add_pd(product_high, c_high);

                let midpoint_fraction_mask = _mm_set1_epi64x(0x1fff_ffff);
                let midpoint_fraction = _mm_set1_epi64x(0x1000_0000);
                let midpoint_low = _mm_cmpeq_epi64(
                    _mm_and_si128(_mm_castpd_si128(sum_low), midpoint_fraction_mask),
                    midpoint_fraction,
                );
                let midpoint_high = _mm_cmpeq_epi64(
                    _mm_and_si128(_mm_castpd_si128(sum_high), midpoint_fraction_mask),
                    midpoint_fraction,
                );
                // Subnormal f32 values have fewer than 24 significant bits, so their midpoint
                // position within the f64 significand varies with the result's exponent. Use a
                // provisional narrowing to identify every result whose rounding interval touches
                // the subnormal range, including zero and the smallest normal value. Those rare
                // lanes take the general round-to-odd path below.
                let mut low = _mm_cvtpd_ps(sum_low);
                let mut high = _mm_cvtpd_ps(sum_high);
                let provisional = _mm_movelh_ps(low, high);
                let abs_provisional_bits = _mm_and_si128(
                    _mm_castps_si128(provisional),
                    _mm_set1_epi32(0x7fff_ffff),
                );
                let at_most_min_normal = _mm_cmpgt_epi32(
                    _mm_set1_epi32(0x0080_0001),
                    abs_provisional_bits,
                );
                let subnormal_low = _mm_unpacklo_epi32(
                    at_most_min_normal,
                    at_most_min_normal,
                );
                let subnormal_high = _mm_unpackhi_epi32(
                    at_most_min_normal,
                    at_most_min_normal,
                );
                let round_to_odd_low = _mm_or_si128(midpoint_low, subnormal_low);
                let round_to_odd_high = _mm_or_si128(midpoint_high, subnormal_high);
                let any_round_to_odd = _mm_or_si128(round_to_odd_low, round_to_odd_high);

                // Optimization notes:
                // On uniform numeric values over the entire range, this branch only fires once per 685k calls,
                // and on uniform numeric values in [-1, 1) it fires once in 17k calls (empirically).
                // So it is worth putting under an `if` despite the branch misprediction penalty.
                // Outlining this into a #[cold] function regresses performance on both fast and slow paths.
                // TODO: try using std::hint::cold_path() once MSRV is >= 1.95 and see if that does anything
                if _mm_testz_si128(any_round_to_odd, any_round_to_odd) == 0 {
                    // Knuth's unconditional TwoSum establishes
                    // `sum + error == product + c` exactly. If a candidate addition was
                    // inexact and its rounded f64 significand is even, shift it by one ULP toward
                    // the error. This produces the round-to-odd intermediate from Theorem 3.
                    let virtual_low = _mm_sub_pd(sum_low, product_low);
                    let error_low = _mm_add_pd(
                        _mm_sub_pd(product_low, _mm_sub_pd(sum_low, virtual_low)),
                        _mm_sub_pd(c_low, virtual_low),
                    );
                    let virtual_high = _mm_sub_pd(sum_high, product_high);
                    let error_high = _mm_add_pd(
                        _mm_sub_pd(product_high, _mm_sub_pd(sum_high, virtual_high)),
                        _mm_sub_pd(c_high, virtual_high),
                    );

                    let one = _mm_set1_epi64x(1);
                    let zero_si128 = _mm_setzero_si128();
                    let zero = _mm_setzero_pd();
                    let sum_low_bits = _mm_castpd_si128(sum_low);
                    let error_low_bits = _mm_castpd_si128(error_low);
                    let even_low = _mm_cmpeq_epi64(
                        _mm_and_si128(sum_low_bits, one),
                        zero_si128,
                    );
                    let correction_low = _mm_and_si128(
                        _mm_and_si128(round_to_odd_low, even_low),
                        _mm_castpd_si128(_mm_cmpneq_pd(error_low, zero)),
                    );
                    // The XOR is negative as a signed qword exactly when the signs differ.
                    // Its zero-greater-than mask is therefore -1 for a downward correction
                    // and zero otherwise; OR with one turns those into the desired -1/+1.
                    let different_sign_low = _mm_cmpgt_epi64(
                        zero_si128,
                        _mm_xor_si128(sum_low_bits, error_low_bits),
                    );
                    let direction_low = _mm_or_si128(different_sign_low, one);
                    sum_low = _mm_castsi128_pd(_mm_add_epi64(
                        sum_low_bits,
                        _mm_and_si128(direction_low, correction_low),
                    ));

                    let sum_high_bits = _mm_castpd_si128(sum_high);
                    let error_high_bits = _mm_castpd_si128(error_high);
                    let even_high = _mm_cmpeq_epi64(
                        _mm_and_si128(sum_high_bits, one),
                        zero_si128,
                    );
                    let correction_high = _mm_and_si128(
                        _mm_and_si128(round_to_odd_high, even_high),
                        _mm_castpd_si128(_mm_cmpneq_pd(error_high, zero)),
                    );
                    let different_sign_high = _mm_cmpgt_epi64(
                        zero_si128,
                        _mm_xor_si128(sum_high_bits, error_high_bits),
                    );
                    let direction_high = _mm_or_si128(different_sign_high, one);
                    sum_high = _mm_castsi128_pd(_mm_add_epi64(
                        sum_high_bits,
                        _mm_and_si128(direction_high, correction_high),
                    ));

                    low = _mm_cvtpd_ps(sum_low);
                    high = _mm_cvtpd_ps(sum_high);
                }

                _mm_movelh_ps(low, high).simd_into(#token)
            }
        })
    }

    fn precise_mul_add_f64x2(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        assert!(
            *self == Self::Sse4_2,
            "precise f64 multiply-add emulation is specific to SSE4.2"
        );
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Float,
            "precise f64 multiply-add requires a floating-point vector"
        );
        assert_eq!(
            vec_ty.scalar_bits, 64,
            "precise f64 multiply-add requires 64-bit lanes"
        );
        assert_eq!(
            vec_ty.n_bits(),
            128,
            "SSE4.2 precise f64 multiply-add requires one native vector"
        );

        // Graillat and Muller's Algorithm 9 computes an FMA from a Dekker product
        // and their correctly rounded addition of a double-word and an FP number:
        // https://perso.lip6.fr/Stef.Graillat/papers/NM-2025.pdf
        // https://doi.org/10.1007/s00211-025-01487-2
        //
        // Their proof assumes an unbounded exponent range. Restrict the packed path
        // to a conservative range where no nonzero intermediate can underflow or
        // overflow, and use scalar FMA for complete IEEE binary64 range coverage.
        //
        // The vectorized path is taken for values between 2^-400 and 2^400,
        // values outside that range are routed to the scalar fallback.
        self.kernel_method(op, vec_ty, |token| {
            quote! {
                let a_raw: __m128d = a.into();
                let b_raw: __m128d = b.into();
                let c_raw: __m128d = c.into();

                let absolute_value_mask = _mm_set1_epi64x(i64::MAX);
                let a_bits = _mm_and_si128(_mm_castpd_si128(a_raw), absolute_value_mask);
                let b_bits = _mm_and_si128(_mm_castpd_si128(b_raw), absolute_value_mask);
                let c_bits = _mm_and_si128(_mm_castpd_si128(c_raw), absolute_value_mask);

                // 2^-400 and 2^400 have biased exponents 623 and 1423. Inclusive
                // comparisons are expressed using strict signed qword comparisons;
                // absolute-value binary64 bits are always nonnegative as i64 values.
                let lower_bound_minus_one = _mm_set1_epi64x((623_i64 << 52) - 1);
                let upper_bound_plus_one = _mm_set1_epi64x((1423_i64 << 52) + 1);
                let zero_bits = _mm_setzero_si128();
                let all_ones = _mm_set1_epi64x(-1);
                let a_safe = _mm_and_si128(
                    _mm_cmpgt_epi64(a_bits, lower_bound_minus_one),
                    _mm_cmpgt_epi64(upper_bound_plus_one, a_bits),
                );
                let b_safe = _mm_and_si128(
                    _mm_cmpgt_epi64(b_bits, lower_bound_minus_one),
                    _mm_cmpgt_epi64(upper_bound_plus_one, b_bits),
                );
                let c_in_range = _mm_and_si128(
                    _mm_cmpgt_epi64(c_bits, lower_bound_minus_one),
                    _mm_cmpgt_epi64(upper_bound_plus_one, c_bits),
                );
                let c_safe = _mm_or_si128(c_in_range, _mm_cmpeq_epi64(c_bits, zero_bits));
                let all_safe = _mm_and_si128(_mm_and_si128(a_safe, b_safe), c_safe);

                // Scalarize both lanes if either one is outside the proven exponent-safe range.
                // This branch must precede the packed arithmetic so unsafe inactive lanes cannot
                // overflow or underflow inside the error-free transforms.
                if _mm_testc_si128(all_safe, all_ones) == 0 {
                    return [
                        f64::mul_add(a[0], b[0], c[0]),
                        f64::mul_add(a[1], b[1], c[1]),
                    ]
                    .simd_into(#token);
                }

                // Split each normal multiplicand into nonoverlapping high and low parts.
                // Adding half of the discarded range before clearing 27 significand bits
                // gives the high part at most 26 significant bits and leaves the low part
                // at most 27. The addition operates on sign-magnitude binary64 encodings,
                // so it rounds the magnitude in the same direction for either sign. The
                // exponent guard above prevents the integer addition from wrapping.
                let split_rounding_bit = _mm_set1_epi64x(1_i64 << 26);
                let split_high_mask = _mm_set1_epi64x(!((1_i64 << 27) - 1));
                let a_high = _mm_castsi128_pd(_mm_and_si128(
                    _mm_add_epi64(_mm_castpd_si128(a_raw), split_rounding_bit),
                    split_high_mask,
                ));
                let a_low = _mm_sub_pd(a_raw, a_high);
                let b_high = _mm_castsi128_pd(_mm_and_si128(
                    _mm_add_epi64(_mm_castpd_si128(b_raw), split_rounding_bit),
                    split_high_mask,
                ));
                let b_low = _mm_sub_pd(b_raw, b_high);

                // Dekker product: product_high + product_low is exactly a * b.
                let product_high = _mm_mul_pd(a_raw, b_raw);
                let product_error_1 =
                    _mm_sub_pd(_mm_mul_pd(a_high, b_high), product_high);
                let product_error_2 =
                    _mm_add_pd(product_error_1, _mm_mul_pd(a_high, b_low));
                let product_error_3 =
                    _mm_add_pd(product_error_2, _mm_mul_pd(a_low, b_high));
                let product_low =
                    _mm_add_pd(product_error_3, _mm_mul_pd(a_low, b_low));

                // Magnitude-sort each addition so Fast2Sum can replace TwoSum.
                // This trades packed compares and blends for a shorter floating-point
                // dependency chain, as suggested by Graillat and Muller.
                let sum_high = _mm_add_pd(product_high, c_raw);
                let product_high_abs = _mm_and_si128(
                    _mm_castpd_si128(product_high),
                    absolute_value_mask,
                );
                let product_high_larger = _mm_cmpgt_pd(
                    _mm_castsi128_pd(product_high_abs),
                    _mm_castsi128_pd(c_bits),
                );
                let sum_large = _mm_blendv_pd(c_raw, product_high, product_high_larger);
                let sum_small = _mm_blendv_pd(product_high, c_raw, product_high_larger);
                let sum_low = _mm_sub_pd(sum_small, _mm_sub_pd(sum_high, sum_large));

                // Only the rounded sum of product_low and sum_low is needed on the
                // overwhelmingly common path. Its exact residual matters only when
                // v_high has one of the two significand shapes that can require the
                // Graillat-Muller correction, so defer that second Fast2Sum until then.
                let v_high = _mm_add_pd(product_low, sum_low);
                let v_high_bits = _mm_castpd_si128(v_high);
                let lower_fraction_mask = _mm_set1_epi64x(0x0007_ffff_ffff_ffff);
                let special_fraction = _mm_cmpeq_epi64(
                    _mm_and_si128(v_high_bits, lower_fraction_mask),
                    zero_bits,
                );

                // A positive zero has the same fraction shape, but cannot need correction.
                // Testing the shape mask against v_high itself excludes it without forming
                // another packed mask. A negative zero may enter the rare path harmlessly.
                if _mm_testz_si128(special_fraction, v_high_bits) == 0 {
                    // Fast2Sum(product_low, sum_low), after the same magnitude sort.
                    let product_low_abs = _mm_and_si128(
                        _mm_castpd_si128(product_low),
                        absolute_value_mask,
                    );
                    let sum_low_abs = _mm_and_si128(
                        _mm_castpd_si128(sum_low),
                        absolute_value_mask,
                    );
                    let product_low_larger = _mm_cmpgt_pd(
                        _mm_castsi128_pd(product_low_abs),
                        _mm_castsi128_pd(sum_low_abs),
                    );
                    let v_large = _mm_blendv_pd(sum_low, product_low, product_low_larger);
                    let v_small = _mm_blendv_pd(product_low, sum_low, product_low_larger);
                    let v_low = _mm_sub_pd(v_small, _mm_sub_pd(v_high, v_large));

                    // The default sum is correctly rounded except when v_low is nonzero and
                    // |v_high| is 2^k or 3 * 2^k. Under the exponent guard, every nonzero
                    // product or residual is a multiple of at least 2^-904, so v_low != 0
                    // implies that v_high is finite, normal, and nonzero. The two relevant
                    // significand shapes therefore differ only in their top fraction bit.
                    let v_low_nonzero =
                        _mm_castpd_si128(_mm_cmpneq_pd(v_low, _mm_setzero_pd()));
                    let special = _mm_and_si128(v_low_nonzero, special_fraction);
                    let different_sign_bits = _mm_slli_epi64::<63>(_mm_srli_epi64::<63>(
                        _mm_xor_si128(v_high_bits, _mm_castpd_si128(v_low)),
                    ));
                    let factor = _mm_blendv_pd(
                        _mm_set1_pd(9.0 / 8.0),
                        _mm_set1_pd(7.0 / 8.0),
                        _mm_castsi128_pd(different_sign_bits),
                    );
                    let adjusted_v_high = _mm_mul_pd(factor, v_high);
                    let correction = _mm_blendv_pd(
                        v_high,
                        adjusted_v_high,
                        _mm_castsi128_pd(special),
                    );
                    _mm_add_pd(sum_high, correction).simd_into(#token)
                } else {
                    _mm_add_pd(sum_high, v_high).simd_into(#token)
                }
            }
        })
    }

    pub(crate) fn handle_ternary(
        &self,
        op: Op,
        method_sig: TokenStream,
        method: &str,
        vec_ty: &VecType,
    ) -> TokenStream {
        match method {
            "mul_add_precise"
                if *self == Self::Sse2
                    && vec_ty.scalar == ScalarType::Float
                    && vec_ty.scalar_bits == 32 =>
            {
                let calls = (0..vec_ty.len).map(|idx| {
                    quote! {
                        scalar_mul_add_precise_f32(self, a[#idx], b[#idx], c[#idx])
                    }
                });
                quote! {
                    #method_sig {
                        [#(#calls),*].simd_into(self)
                    }
                }
            }
            "mul_add_precise" if *self == Self::Sse4_2 && vec_ty.scalar_bits == 32 => {
                self.precise_mul_add_f32x4(op, vec_ty)
            }
            "mul_add_precise" if *self == Self::Sse4_2 && vec_ty.scalar_bits == 64 => {
                self.precise_mul_add_f64x2(op, vec_ty)
            }
            "mul_add_precise" if matches!(self, Self::Avx2 | Self::Avx512) => {
                let mul_add = generic_op_name("mul_add", vec_ty);
                quote! {
                    #method_sig {
                        self.#mul_add(a, b, c)
                    }
                }
            }
            "mul_add_precise" => fallback_method(op, vec_ty),
            "mul_sub_precise" if matches!(self, Self::Avx2 | Self::Avx512) => {
                let mul_sub = generic_op_name("mul_sub", vec_ty);
                quote! {
                    #method_sig {
                        self.#mul_sub(a, b, c)
                    }
                }
            }
            "mul_sub_precise" => {
                let mul_add_precise = generic_op_name("mul_add_precise", vec_ty);
                quote! {
                    #method_sig {
                        self.#mul_add_precise(a, b, -c)
                    }
                }
            }
            "mul_add" if matches!(self, Self::Avx2 | Self::Avx512) => {
                let intrinsic = simple_intrinsic("fmadd", vec_ty);
                self.kernel_method(
                    op,
                    vec_ty,
                    |token| quote! { #intrinsic(a.into(), b.into(), c.into()).simd_into(#token) },
                )
            }
            "mul_sub" if matches!(self, Self::Avx2 | Self::Avx512) => {
                let intrinsic = simple_intrinsic("fmsub", vec_ty);
                self.kernel_method(
                    op,
                    vec_ty,
                    |token| quote! { #intrinsic(a.into(), b.into(), c.into()).simd_into(#token) },
                )
            }
            "mul_add" => {
                quote! {
                    #method_sig {
                        a * b + c
                    }
                }
            }
            "mul_sub" => {
                quote! {
                    #method_sig {
                        a * b - c
                    }
                }
            }
            _ => {
                let args = [
                    quote! { a.into() },
                    quote! { b.into() },
                    quote! { c.into() },
                ];

                let expr = x86::expr(method, vec_ty, &args);
                self.kernel_method(op, vec_ty, |token| {
                    quote! { #expr.simd_into(#token) }
                })
            }
        }
    }

    pub(crate) fn handle_select(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        if *self == Self::Avx512 {
            let method_sig = op.simd_trait_method_sig(vec_ty);
            if vec_ty.scalar == ScalarType::Mask {
                let lane_mask = avx512_mask_lane_bits(vec_ty);
                let a_bits = avx512_mask_bits_expr(quote! { a });
                let b_bits = avx512_mask_bits_expr(quote! { b });
                let c_bits = avx512_mask_bits_expr(quote! { c });
                let result = avx512_mask_value(
                    vec_ty,
                    quote! { ((#a_bits & #b_bits) | ((!#a_bits) & #c_bits)) & #lane_mask },
                );
                return quote! {
                    #method_sig {
                        #result
                    }
                };
            }

            let blend = avx512_mask_blend_intrinsic(vec_ty);
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    #blend(a.val, c.into(), b.into()).simd_into(#token)
                }
            });
        }

        if *self == Self::Sse2 {
            let mask = match vec_ty.scalar {
                ScalarType::Float => {
                    let cast = cast_ident(
                        ScalarType::Mask,
                        ScalarType::Float,
                        vec_ty.scalar_bits,
                        vec_ty.scalar_bits,
                        vec_ty.n_bits(),
                    );
                    quote! { #cast(a.into()) }
                }
                _ => quote! { a.into() },
            };
            let expr = sse2_select_expr(vec_ty, mask, quote! { b.into() }, quote! { c.into() });
            return self.kernel_method(op, vec_ty, |token| {
                quote! { #expr.simd_into(#token) }
            });
        }

        // Our select ops' argument order is mask, a, b; Intel's intrinsics are b, a, mask
        let args = [
            quote! { c.into() },
            quote! { b.into() },
            match vec_ty.scalar {
                ScalarType::Float => {
                    let ident = cast_ident(
                        ScalarType::Mask,
                        ScalarType::Float,
                        vec_ty.scalar_bits,
                        vec_ty.scalar_bits,
                        vec_ty.n_bits(),
                    );
                    quote! { #ident(a.into()) }
                }
                _ => quote! { a.into() },
            },
        ];
        let expr = x86::expr("select", vec_ty, &args);

        self.kernel_method(op, vec_ty, |token| {
            quote! { #expr.simd_into(#token) }
        })
    }

    pub(crate) fn handle_split(&self, op: Op, vec_ty: &VecType, half_ty: &VecType) -> TokenStream {
        if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Mask {
            let method_sig = op.simd_trait_method_sig(vec_ty);
            let half_rust = half_ty.rust();
            let half_len = half_ty.len;
            let half_mask = avx512_mask_lane_bits(half_ty);
            return quote! {
                #method_sig {
                    let bits = u64::from(a.val);
                    (
                        #half_rust { val: (bits & #half_mask) as _, simd: self },
                        #half_rust { val: ((bits >> #half_len) & #half_mask) as _, simd: self },
                    )
                }
            };
        }

        if *self == Self::Avx512 && half_ty.n_bits() == 256 {
            let (lo, hi) = match vec_ty.scalar {
                ScalarType::Float if vec_ty.scalar_bits == 32 => (
                    quote! { _mm512_castps512_ps256(a.into()) },
                    quote! { _mm512_extractf32x8_ps::<1>(a.into()) },
                ),
                ScalarType::Float if vec_ty.scalar_bits == 64 => (
                    quote! { _mm512_castpd512_pd256(a.into()) },
                    quote! { _mm512_extractf64x4_pd::<1>(a.into()) },
                ),
                _ => (
                    quote! { _mm512_castsi512_si256(a.into()) },
                    quote! { _mm512_extracti64x4_epi64::<1>(a.into()) },
                ),
            };
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    (
                        #lo.simd_into(#token),
                        #hi.simd_into(#token),
                    )
                }
            });
        }

        if matches!(self, Self::Avx2 | Self::Avx512) && half_ty.n_bits() == 128 {
            let extract_op = match vec_ty.scalar {
                ScalarType::Float => "extractf128",
                _ => "extracti128",
            };
            let extract_intrinsic = intrinsic_ident(extract_op, coarse_type(vec_ty), 256);
            self.kernel_method(op, vec_ty, |token| {
                quote! {
                    (
                        #extract_intrinsic::<0>(a.into()).simd_into(#token),
                        #extract_intrinsic::<1>(a.into()).simd_into(#token),
                    )
                }
            })
        } else {
            let method_sig = op.simd_trait_method_sig(vec_ty);
            generic_block_split(method_sig, half_ty, self.max_block_size())
        }
    }

    pub(crate) fn handle_combine(
        &self,
        op: Op,
        vec_ty: &VecType,
        combined_ty: &VecType,
    ) -> TokenStream {
        let method_sig = op.simd_trait_method_sig(vec_ty);
        if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Mask {
            let combined_rust = combined_ty.rust();
            let shift = vec_ty.len;
            let lane_mask = avx512_mask_lane_bits(combined_ty);
            let bits = if avx512_mask_register_bits(combined_ty) == 64 {
                quote! { bits }
            } else {
                quote! { bits as _ }
            };
            return quote! {
                #method_sig {
                    let bits = (u64::from(a.val) | (u64::from(b.val) << #shift)) & #lane_mask;
                    #combined_rust { val: #bits, simd: self }
                }
            };
        }

        if *self == Self::Avx512 && combined_ty.n_bits() == 512 {
            let expr = match vec_ty.scalar {
                ScalarType::Float if vec_ty.scalar_bits == 32 => quote! {
                    _mm512_insertf32x8::<1>(_mm512_castps256_ps512(a.into()), b.into())
                },
                ScalarType::Float if vec_ty.scalar_bits == 64 => quote! {
                    _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(a.into()), b.into())
                },
                _ => quote! {
                    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(a.into()), b.into())
                },
            };
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    #expr.simd_into(#token)
                }
            });
        }

        if matches!(self, Self::Avx2 | Self::Avx512) && combined_ty.n_bits() == 256 {
            let suffix = match (vec_ty.scalar, vec_ty.scalar_bits) {
                (ScalarType::Float, 32) => "m128",
                (ScalarType::Float, 64) => "m128d",
                _ => "m128i",
            };
            let set_intrinsic = intrinsic_ident("setr", suffix, 256);
            self.kernel_method(
                op,
                vec_ty,
                |token| quote! { #set_intrinsic(a.into(), b.into()).simd_into(#token) },
            )
        } else {
            generic_block_combine(method_sig, combined_ty, self.max_block_size())
        }
    }

    pub(crate) fn handle_zip(&self, op: Op, vec_ty: &VecType, select_low: bool) -> TokenStream {
        if *self == Self::Avx512 && vec_ty.scalar != ScalarType::Mask && vec_ty.n_bits() >= 256 {
            let offset = if select_low { 0 } else { vec_ty.len / 2 };
            let indices = (0..vec_ty.len).map(|i| {
                let source_lane = offset + (i / 2);
                if i % 2 == 0 {
                    source_lane
                } else {
                    vec_ty.len + source_lane
                }
            });
            let idx = avx512_index_vector(vec_ty, indices);
            let permute = avx512_permutex2var_intrinsic(vec_ty);
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    #permute(a.into(), #idx, b.into()).simd_into(#token)
                }
            });
        }

        self.kernel_method(op, vec_ty, |token| match vec_ty.n_bits() {
            128 => {
                let op = if select_low { "unpacklo" } else { "unpackhi" };

                let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
                let unpack_intrinsic = intrinsic_ident(op, suffix, vec_ty.n_bits());
                quote! {
                    #unpack_intrinsic(a.into(), b.into()).simd_into(#token)
                }
            }
            256 => {
                let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
                let lo = intrinsic_ident("unpacklo", suffix, vec_ty.n_bits());
                let hi = intrinsic_ident("unpackhi", suffix, vec_ty.n_bits());
                let shuffle_immediate = if select_low {
                    quote! { 0b0010_0000 }
                } else {
                    quote! { 0b0011_0001 }
                };

                let shuffle = intrinsic_ident(
                    match vec_ty.scalar {
                        ScalarType::Float => "permute2f128",
                        _ => "permute2x128",
                    },
                    coarse_type(vec_ty),
                    256,
                );

                quote! {
                    let lo = #lo(a.into(), b.into());
                    let hi = #hi(a.into(), b.into());

                    #shuffle::<#shuffle_immediate>(lo, hi).simd_into(#token)
                }
            }
            _ => unreachable!(),
        })
    }

    pub(crate) fn handle_interleave(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        if *self == Self::Avx512 && vec_ty.scalar != ScalarType::Mask && vec_ty.n_bits() >= 256 {
            let lo_indices = (0..vec_ty.len).map(|i| {
                let source_lane = i / 2;
                if i % 2 == 0 {
                    source_lane
                } else {
                    vec_ty.len + source_lane
                }
            });
            let hi_indices = (0..vec_ty.len).map(|i| {
                let source_lane = (vec_ty.len / 2) + (i / 2);
                if i % 2 == 0 {
                    source_lane
                } else {
                    vec_ty.len + source_lane
                }
            });
            let lo_idx = avx512_index_vector(vec_ty, lo_indices);
            let hi_idx = avx512_index_vector(vec_ty, hi_indices);
            let permute = avx512_permutex2var_intrinsic(vec_ty);
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    let a = a.into();
                    let b = b.into();
                    (
                        #permute(a, #lo_idx, b).simd_into(#token),
                        #permute(a, #hi_idx, b).simd_into(#token),
                    )
                }
            });
        }

        match vec_ty.n_bits() {
            256 => {
                // Optimized path: compute unpacklo and unpackhi once, then use permute2f128 to
                // produce both zip_low and zip_high results. This avoids the redundant unpack
                // operations that occur when zip_low and zip_high are called separately.
                let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
                let lo = intrinsic_ident("unpacklo", suffix, 256);
                let hi = intrinsic_ident("unpackhi", suffix, 256);
                let shuffle = intrinsic_ident(
                    match vec_ty.scalar {
                        ScalarType::Float => "permute2f128",
                        _ => "permute2x128",
                    },
                    coarse_type(vec_ty),
                    256,
                );
                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        let lo = #lo(a.into(), b.into());
                        let hi = #hi(a.into(), b.into());
                        (
                            #shuffle::<0b0010_0000>(lo, hi).simd_into(#token),
                            #shuffle::<0b0011_0001>(lo, hi).simd_into(#token),
                        )
                    }
                })
            }
            _ => {
                // For 128-bit vectors, zip_low/zip_high are single instructions (unpacklo/unpackhi),
                // so there's no redundancy in calling them separately.
                let zip_low = generic_op_name("zip_low", vec_ty);
                let zip_high = generic_op_name("zip_high", vec_ty);
                let method_sig = op.simd_trait_method_sig(vec_ty);
                quote! {
                    #method_sig {
                        (self.#zip_low(a, b), self.#zip_high(a, b))
                    }
                }
            }
        }
    }

    pub(crate) fn handle_deinterleave(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        if *self == Self::Avx512 && avx512_should_use_unzip_permutex2var(vec_ty) {
            let even_indices = (0..vec_ty.len).map(|i| {
                if i < vec_ty.len / 2 {
                    i * 2
                } else {
                    vec_ty.len + ((i - vec_ty.len / 2) * 2)
                }
            });
            let odd_indices = (0..vec_ty.len).map(|i| {
                if i < vec_ty.len / 2 {
                    i * 2 + 1
                } else {
                    vec_ty.len + ((i - vec_ty.len / 2) * 2 + 1)
                }
            });
            let even_idx = avx512_index_vector(vec_ty, even_indices);
            let odd_idx = avx512_index_vector(vec_ty, odd_indices);
            let permute = avx512_permutex2var_intrinsic(vec_ty);
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    let a = a.into();
                    let b = b.into();
                    (
                        #permute(a, #even_idx, b).simd_into(#token),
                        #permute(a, #odd_idx, b).simd_into(#token),
                    )
                }
            });
        }

        if matches!(self, Self::Sse4_2 | Self::Avx2)
            && vec_ty.n_bits() == 128
            && matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned)
            && matches!(vec_ty.scalar_bits, 8 | 16)
        {
            let mask = narrow_unzip_shuffle_mask(vec_ty);
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    let mask = #mask;
                    let a = _mm_shuffle_epi8(a.into(), mask);
                    let b = _mm_shuffle_epi8(b.into(), mask);
                    (
                        _mm_unpacklo_epi64(a, b).simd_into(#token),
                        _mm_unpackhi_epi64(a, b).simd_into(#token),
                    )
                }
            });
        }

        if *self == Self::Avx2
            && vec_ty.n_bits() == 256
            && matches!(vec_ty.scalar, ScalarType::Int | ScalarType::Unsigned)
            && matches!(vec_ty.scalar_bits, 8 | 16)
        {
            let mask = narrow_unzip_shuffle_mask(vec_ty);
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    let mask = #mask;
                    let a = _mm256_shuffle_epi8(a.into(), mask);
                    let b = _mm256_shuffle_epi8(b.into(), mask);
                    let low = _mm256_permute2x128_si256::<0b0010_0000>(a, b);
                    let high = _mm256_permute2x128_si256::<0b0011_0001>(a, b);
                    (
                        _mm256_unpacklo_epi64(low, high).simd_into(#token),
                        _mm256_unpackhi_epi64(low, high).simd_into(#token),
                    )
                }
            });
        }

        match vec_ty.n_bits() {
            256 => {
                // Optimized path: compute the per-input shuffles once, then use permute2f128 /
                // permute2x128 to produce both unzip_low and unzip_high results. This avoids
                // the redundant shuffle operations that occur when unzip_low and unzip_high are
                // called separately.
                let (t1, t2, shuffle) = self.unzip256_intermediates(vec_ty);
                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        let t1 = #t1;
                        let t2 = #t2;
                        (
                            #shuffle::<0b0010_0000>(t1, t2).simd_into(#token),
                            #shuffle::<0b0011_0001>(t1, t2).simd_into(#token),
                        )
                    }
                })
            }
            _ => {
                // For 128-bit vectors, unzip_low/unzip_high are cheap, so there's no
                // redundancy in calling them separately.
                let unzip_low = generic_op_name("unzip_low", vec_ty);
                let unzip_high = generic_op_name("unzip_high", vec_ty);
                let method_sig = op.simd_trait_method_sig(vec_ty);
                quote! {
                    #method_sig {
                        (self.#unzip_low(a, b), self.#unzip_high(a, b))
                    }
                }
            }
        }
    }

    /// Returns `(t1_expr, t2_expr, shuffle_ident)` for 256-bit unzip operations.
    ///
    /// `t1` and `t2` are the per-input shuffles that separate even and odd elements.
    /// `shuffle` is the `permute2f128` / `permute2x128` intrinsic used to select
    /// the low or high halves via immediate `0b0010_0000` or `0b0011_0001`.
    fn unzip256_intermediates(&self, vec_ty: &VecType) -> (TokenStream, TokenStream, Ident) {
        let shuffle = intrinsic_ident(
            match vec_ty.scalar {
                ScalarType::Float => "permute2f128",
                _ => "permute2x128",
            },
            coarse_type(vec_ty),
            256,
        );

        let (t1, t2) = match vec_ty.scalar_bits {
            32 | 64 => {
                let kind = match vec_ty.scalar_bits {
                    32 => "permutevar8x32",
                    64 => "permute4x64",
                    _ => unreachable!(),
                };
                let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
                let intr = intrinsic_ident(kind, suffix, 256);
                let shuf = |input: TokenStream| match vec_ty.scalar_bits {
                    32 => quote! { #intr(#input, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)) },
                    64 => quote! { #intr::<0b11_01_10_00>(#input) },
                    _ => unreachable!(),
                };
                (shuf(quote! { a.into() }), shuf(quote! { b.into() }))
            }
            8 | 16 => {
                let mask = narrow_unzip_shuffle_mask(vec_ty);
                let shuf = |input: TokenStream| {
                    quote! {
                        _mm256_permute4x64_epi64::<0b11_01_10_00>(
                            _mm256_shuffle_epi8(#input, #mask),
                        )
                    }
                };
                (shuf(quote! { a.into() }), shuf(quote! { b.into() }))
            }
            _ => unreachable!(),
        };

        (t1, t2, shuffle)
    }

    pub(crate) fn handle_unzip(&self, op: Op, vec_ty: &VecType, select_even: bool) -> TokenStream {
        if *self == Self::Avx512 && avx512_should_use_unzip_permutex2var(vec_ty) {
            let lane_offset = if select_even { 0 } else { 1 };
            let indices = (0..vec_ty.len).map(|i| {
                if i < vec_ty.len / 2 {
                    i * 2 + lane_offset
                } else {
                    vec_ty.len + ((i - vec_ty.len / 2) * 2 + lane_offset)
                }
            });
            let idx = avx512_index_vector(vec_ty, indices);
            let permute = avx512_permutex2var_intrinsic(vec_ty);
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    #permute(a.into(), #idx, b.into()).simd_into(#token)
                }
            });
        }

        if *self == Self::Sse2
            && vec_ty.n_bits() == 128
            && matches!(
                vec_ty.scalar,
                ScalarType::Int | ScalarType::Mask | ScalarType::Unsigned
            )
            && matches!(vec_ty.scalar_bits, 8 | 16)
        {
            return fallback_method(op, vec_ty);
        }

        self.kernel_method(op, vec_ty, |token| {
            match (vec_ty.scalar, vec_ty.n_bits(), vec_ty.scalar_bits) {
                (ScalarType::Float, 128, _) => {
                    // 128-bit shuffle of floats or doubles; there are built-in SSE intrinsics for this
                    let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
                    let intrinsic = intrinsic_ident("shuffle", suffix, vec_ty.n_bits());

                    let mask = match (vec_ty.scalar_bits, select_even) {
                        (32, true) => quote! { 0b10_00_10_00 },
                        (32, false) => quote! { 0b11_01_11_01 },
                        (64, true) => quote! { 0b00 },
                        (64, false) => quote! { 0b11 },
                        _ => unimplemented!(),
                    };

                    quote! { #intrinsic::<#mask>(a.into(), b.into()).simd_into(#token) }
                }
                (ScalarType::Int | ScalarType::Mask | ScalarType::Unsigned, 128, 64) => {
                    let op = if select_even { "unpacklo" } else { "unpackhi" };
                    let intrinsic = intrinsic_ident(op, "epi64", vec_ty.n_bits());

                    quote! {
                        #intrinsic(a.into(), b.into()).simd_into(#token)
                    }
                }
                (ScalarType::Int | ScalarType::Mask | ScalarType::Unsigned, 128, 32) => {
                    // 128-bit shuffle of 32-bit integers; unlike with floats, there is no single shuffle instruction that
                    // combines two vectors
                    let op = if select_even { "unpacklo" } else { "unpackhi" };
                    let intrinsic = intrinsic_ident(op, "epi64", vec_ty.n_bits());

                    quote! {
                        let t1 = _mm_shuffle_epi32::<0b11_01_10_00>(a.into());
                        let t2 = _mm_shuffle_epi32::<0b11_01_10_00>(b.into());
                        #intrinsic(t1, t2).simd_into(#token)
                    }
                }
                (ScalarType::Int | ScalarType::Mask | ScalarType::Unsigned, 128, 16 | 8) => {
                    // Separate out the even-indexed and odd-indexed elements
                    let mask = narrow_unzip_shuffle_mask(vec_ty);
                    let shuffle_epi8 = intrinsic_ident("shuffle", "epi8", vec_ty.n_bits());

                    // Select either the low or high half of each one
                    let op = if select_even { "unpacklo" } else { "unpackhi" };
                    let unpack_epi64 = intrinsic_ident(op, "epi64", vec_ty.n_bits());

                    quote! {
                        let mask = #mask;

                        let t1 = #shuffle_epi8(a.into(), mask);
                        let t2 = #shuffle_epi8(b.into(), mask);
                        #unpack_epi64(t1, t2).simd_into(#token)
                    }
                }
                (_, 256, _) => {
                    let (t1, t2, shuffle) = self.unzip256_intermediates(vec_ty);
                    let shuffle_immediate = if select_even {
                        quote! { 0b0010_0000 }
                    } else {
                        quote! { 0b0011_0001 }
                    };

                    quote! {
                        let t1 = #t1;
                        let t2 = #t2;
                        #shuffle::<#shuffle_immediate>(t1, t2).simd_into(#token)
                    }
                }
                _ => unimplemented!(),
            }
        })
    }

    pub(crate) fn handle_slide(
        &self,
        method_sig: TokenStream,
        vec_ty: &VecType,
        granularity: SlideGranularity,
    ) -> TokenStream {
        use SlideGranularity::*;

        let block_wrapper = vec_ty.aligned_wrapper();
        let combined_bytes = vec_ty.reinterpret(ScalarType::Unsigned, 8).rust();
        let scalar_bytes = vec_ty.scalar_bits / 8;
        let max_shift = match granularity {
            WithinBlocks => vec_ty.len / (vec_ty.n_bits() / 128),
            AcrossBlocks => vec_ty.len,
        };

        if *self == Self::Avx512
            && granularity == WithinBlocks
            && vec_ty.scalar != ScalarType::Mask
            && vec_ty.n_bits() >= 256
        {
            let alignr = format_ident!("dyn_alignr_{}", vec_ty.n_bits());
            let byte_shift = if scalar_bytes == 1 {
                quote! { SHIFT }
            } else {
                quote! { SHIFT * #scalar_bytes }
            };

            return quote! {
                #method_sig {
                    if SHIFT == 0 {
                        return a;
                    }
                    if SHIFT >= #max_shift {
                        return b;
                    }

                    let a = Bytes::to_bytes(a).val.0;
                    let b = Bytes::to_bytes(b).val.0;
                    let result = #alignr(self, b, a, #byte_shift);
                    Bytes::from_bytes(#combined_bytes {
                        val: #block_wrapper(result),
                        simd: self,
                    })
                }
            };
        }

        if *self == Self::Avx512 && granularity == AcrossBlocks && vec_ty.n_bits() >= 256 {
            let level = self.token();
            let ty = vec_ty.rust();
            let vec = quote! { #ty<#level> };
            // This deliberately expresses slides of every lane width as byte permutations.
            // LLVM folds the constant indices to the same instructions as the equivalent
            // element-width formulation, and the byte form avoids the poor constant-index
            // codegen of the floating-point permutex2var intrinsics:
            // https://github.com/rust-lang/rust/issues/159842
            let byte_ty = vec_ty.reinterpret(ScalarType::Unsigned, 8);
            let base_idx = avx512_index_vector(&byte_ty, 0..byte_ty.len);
            let set_shift = set1_intrinsic(&byte_ty);
            let add = simple_sign_unaware_intrinsic("add", &byte_ty);
            let permute = avx512_permutex2var_intrinsic(&byte_ty);
            let byte_shift = if scalar_bytes == 1 {
                quote! { shift }
            } else {
                quote! { shift * #scalar_bytes }
            };

            return quote! {
                #method_sig {
                    crate::kernel!(
                        #[inline(always)]
                        fn kernel(token: #level, a: #vec, b: #vec, shift: usize) -> #vec {
                            if shift >= #max_shift {
                                return b;
                            }

                            let idx = #add(#base_idx, #set_shift((#byte_shift) as i8));
                            let result = #permute(
                                Bytes::to_bytes(a).val.0,
                                idx,
                                Bytes::to_bytes(b).val.0,
                            );
                            Bytes::from_bytes(#combined_bytes {
                                val: #block_wrapper(result),
                                simd: token,
                            })
                        }
                    );

                    kernel(self, a, b, SHIFT)
                }
            };
        }

        let alignr_op = match (granularity, vec_ty.n_bits(), self) {
            (WithinBlocks, 128, _) => {
                panic!("This should have been handled by generic_op");
            }
            (WithinBlocks, _, _) | (_, 128, _) => {
                // For WithinBlocks, use elements per 128-bit block; for 128-bit vectors, use total elements
                format_ident!("dyn_alignr_{}", vec_ty.n_bits())
            }
            (AcrossBlocks, 256 | 512, Self::Sse2 | Self::Sse4_2) => {
                // Inter-block shift or rotate in 128-bit x86 backends: use cross_block_alignr

                format_ident!("cross_block_alignr_128x{}", vec_ty.n_bits() / 128)
            }
            (AcrossBlocks, 256 | 512, Self::Avx2) => {
                format_ident!("cross_block_alignr_256x{}", vec_ty.n_bits() / 256)
            }
            _ => unimplemented!(),
        };
        let byte_shift = if scalar_bytes == 1 {
            quote! { SHIFT }
        } else {
            quote! { SHIFT * #scalar_bytes }
        };

        quote! {
            #method_sig {
                if SHIFT >= #max_shift {
                    return b;
                }

                // b and a are swapped here to match ARM's vext semantics. For vext, we can think of `a` as the "left",
                // and we concatenate `b` to its "right". This makes sense, since `a` is the left-hand side and `b` is
                // the right-hand side. x86's `alignr` is backwards, and treats `b` as the high/left block.
                let result = #alignr_op(
                    self,
                    Bytes::to_bytes(b).val.0,
                    Bytes::to_bytes(a).val.0,
                    #byte_shift,
                );
                Bytes::from_bytes(#combined_bytes {
                    val: #block_wrapper(result),
                    simd: self,
                })
            }
        }
    }

    pub(crate) fn handle_swizzle_dyn_within_blocks(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let bytes_ty = vec_ty.bytes_ty();
        let bytes = bytes_ty.rust();
        let wrapper = bytes_ty.aligned_wrapper();

        if *self == Self::Sse2 {
            return fallback_method(op, vec_ty);
        }

        self.kernel_method(op, vec_ty, |token| {
            let body = if *self == Self::Avx512 {
                match vec_ty.n_bits() {
                    128 => quote! {
                        let bytes = Bytes::to_bytes(a).val.0;
                        let result = _mm_mask_shuffle_epi8(bytes, u16::MAX, bytes, indices.into());
                    },
                    256 => quote! {
                        let bytes = Bytes::to_bytes(a).val.0;
                        let result = _mm256_mask_shuffle_epi8(bytes, u32::MAX, bytes, indices.into());
                    },
                    512 => quote! {
                        let result =
                            _mm512_shuffle_epi8(Bytes::to_bytes(a).val.0, indices.into());
                    },
                    _ => unreachable!(),
                }
            } else {
                let shuffle = simple_sign_unaware_intrinsic("shuffle", &bytes_ty);
                quote! {
                    let result = #shuffle(Bytes::to_bytes(a).val.0, indices.into());
                }
            };

            quote! {
                #body
                Bytes::from_bytes(#bytes {
                    val: #wrapper(result),
                    simd: #token,
                })
            }
        })
    }

    pub(crate) fn handle_swizzle_dyn(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let bytes_ty = vec_ty.bytes_ty();
        let bytes = bytes_ty.rust();
        let wrapper = bytes_ty.aligned_wrapper();

        if *self == Self::Sse2 || (*self == Self::Sse4_2 && vec_ty.n_bits() == 512) {
            return fallback_method(op, vec_ty);
        }

        // Emulated double-native-width variants delegate to swizzle_dyn_precise
        // because zeroes let us cheaply join the two halves, and on AVX2 zeroing is already very cheap
        // through a clever trick: https://shnatsel.github.io/improving-std-simd-swizzle-dyn/#optimizing-avx2
        if matches!(
            (*self, vec_ty.n_bits()),
            (Self::Sse4_2, 256) | (Self::Avx2, 512)
        ) {
            let method_sig = op.simd_trait_method_sig(vec_ty);
            let precise = generic_op_name("swizzle_dyn_precise", vec_ty);
            return quote! {
                #method_sig {
                    self.#precise(a, indices)
                }
            };
        }

        // lower into native ops for native-width vectors
        self.kernel_method(op, vec_ty, |token| {
            let body = match (*self, vec_ty.n_bits()) {
                (Self::Sse4_2 | Self::Avx2, 128) => quote! {
                    let result =
                        _mm_shuffle_epi8(Bytes::to_bytes(a).val.0, indices.into());
                },
                (Self::Avx2, 256) => quote! {
                    let bytes = Bytes::to_bytes(a).val.0;
                    let indices = indices.into();
                    let swapped = _mm256_permute2x128_si256::<0x01>(bytes, bytes);

                    // For an in-range index, set the sign bit in the shuffle
                    // control for the table half that does not contain the
                    // requested byte. The high output lane has the opposite
                    // local/remote mapping.
                    let lane_bias = _mm256_set_m128i(
                        _mm_set1_epi8(-16),
                        _mm_set1_epi8(112),
                    );
                    let local_control = _mm256_add_epi8(indices, lane_bias);
                    let remote_control =
                        _mm256_xor_si256(local_control, _mm256_set1_epi8(i8::MIN));

                    let local = _mm256_shuffle_epi8(bytes, local_control);
                    let remote = _mm256_shuffle_epi8(swapped, remote_control);
                    let result = _mm256_or_si256(local, remote);
                },
                (Self::Avx512, 128 | 256 | 512) => {
                    let permute = intrinsic_ident("permutexvar", "epi8", vec_ty.n_bits());
                    quote! {
                        let result =
                            #permute(indices.into(), Bytes::to_bytes(a).val.0);
                    }
                }
                _ => unreachable!(),
            };

            quote! {
                #body
                Bytes::from_bytes(#bytes {
                    val: #wrapper(result),
                    simd: #token,
                })
            }
        })
    }

    pub(crate) fn handle_swizzle_dyn_precise(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let bytes_ty = vec_ty.bytes_ty();
        let bytes = bytes_ty.rust();
        let wrapper = bytes_ty.aligned_wrapper();

        if *self == Self::Sse2 || (*self == Self::Sse4_2 && vec_ty.n_bits() == 512) {
            return fallback_method(op, vec_ty);
        }

        self.kernel_method(op, vec_ty, |token| {
            let body = match (*self, vec_ty.n_bits()) {
                (Self::Sse4_2 | Self::Avx2, 128) => quote! {
                    let indices = indices.into();
                    // Preserve the original high bit, and set it for indices 16..=127.
                    // The added value only changes bits that PSHUFB ignores for valid indices.
                    let index_out_of_range = _mm_add_epi8(indices, _mm_set1_epi8(112));
                    let zeroing_indices = _mm_or_si128(indices, index_out_of_range);
                    let result = _mm_shuffle_epi8(Bytes::to_bytes(a).val.0, zeroing_indices);
                    let result_bytes = #bytes { val: #wrapper(result), simd: #token };
                },
                (Self::Sse4_2, 256) | (Self::Avx2, 512) => {
                    recursive_swizzle_dyn_precise_body(vec_ty, token)
                }
                (Self::Avx2, 256) => quote! {
                    let bytes = Bytes::to_bytes(a);
                    let indices = indices.into();
                    let swapped = _mm256_permute2x128_si256::<0x01>(bytes.val.0, bytes.val.0);

                    // Adding 0x60 preserves the low nibble and bit 4 for valid
                    // indices 0..=31. Larger indices get their high bit set, so
                    // VPSHUFB supplies the required out-of-bounds zeroing.
                    let control = _mm256_adds_epu8(indices, _mm256_set1_epi8(0x60));

                    let local = _mm256_shuffle_epi8(bytes.val.0, control);
                    let remote = _mm256_shuffle_epi8(swapped, control);

                    // In the low lane, adding 0x10 moves the valid index's bit 4
                    // into the sign bit. The high lane has the opposite
                    // local/remote mapping, so adding 0x90 flips the selection.
                    // Out-of-range indices already zeroed both shuffle results,
                    // making the blend selection irrelevant for them.
                    let select_bias = _mm256_set_m128i(
                        _mm_set1_epi8(-112),
                        _mm_set1_epi8(16),
                    );
                    let select_remote = _mm256_add_epi8(control, select_bias);
                    let result = _mm256_blendv_epi8(local, remote, select_remote);
                    let result_bytes = #bytes { val: #wrapper(result), simd: #token };
                },
                (Self::Avx512, 128 | 256 | 512) => {
                    let min = intrinsic_ident("min", "epu8", vec_ty.n_bits());
                    let permute = intrinsic_ident("permutex2var", "epi8", vec_ty.n_bits());
                    let set1 = set1_intrinsic(&bytes_ty);
                    let setzero =
                        intrinsic_ident("setzero", coarse_type(&bytes_ty), vec_ty.n_bits());
                    let byte_count = signed_literal(bytes_ty.len as u64, 8);
                    quote! {
                        let bytes = Bytes::to_bytes(a).val.0;
                        let indices = indices.into();
                        // Clamp out-of-range indices to the first byte of a
                        // second, all-zero table.
                        let indices = #min(indices, #set1(#byte_count));
                        let result = #permute(bytes, indices, #setzero());
                        let result_bytes = #bytes { val: #wrapper(result), simd: #token };
                    }
                }
                _ => unreachable!(),
            };

            quote! {
                #body
                Bytes::from_bytes(result_bytes)
            }
        })
    }

    pub(crate) fn handle_cvt(
        &self,
        op: Op,
        vec_ty: &VecType,
        target_scalar: ScalarType,
        target_scalar_bits: usize,
        precise: bool,
    ) -> TokenStream {
        use Precision::{Approx, Precise};
        use ScalarType::{Float, Int, Unsigned};

        // Conversion methods preserve the number of lanes, so their source and destination
        // scalars must have the same width. Widening and narrowing conversions are generated by
        // separate handlers.
        assert_eq!(
            vec_ty.scalar_bits, target_scalar_bits,
            "we currently only support converting between types of the same width"
        );

        // `Precise` float-to-integer conversions reproduce Rust's saturating `as` semantics,
        // including zero for NaN. `Approx` conversions may return the hardware-defined result for
        // NaN and out-of-range inputs. Precision does not affect integer-to-float conversions.
        let precision = if precise { Precise } else { Approx };

        // Keep every property that affects instruction selection in one key. The exhaustive match
        // below then makes additions to the backend or type matrix fail loudly until handled.
        let conversion = (
            *self,
            vec_ty.scalar,
            target_scalar,
            vec_ty.scalar_bits,
            vec_ty.n_bits(),
            precision,
        );

        // For a two-lane signed approximate conversion, scalar casts outperform SIMD formulations.
        // CVTTSD2SI cast is not required to saturate, so it is cheaper than `as` casts too.
        //
        // CVTTSD2SI is available starting with SSE2, so use it for every pre-AVX-512 x86 level.
        // The SIMD implementation below is faster for u64, for precise semantics,
        // and for vectors larger than 2 elements.
        //
        // CVTTSD2SI with a 64-bit destination only exists in 64-bit mode.
        // 32-bit x86 is not an optimization target in 2026, so route it through scalar fallback.
        if matches!(
            conversion,
            (
                Self::Sse2 | Self::Sse4_2 | Self::Avx2,
                Float,
                Int,
                64,
                128,
                Approx,
            )
        ) {
            let unpack_high = simple_intrinsic("unpackhi", vec_ty);
            let specialized = self.kernel_method(op, vec_ty, |token| {
                quote! {
                    let a = a.into();
                    let low = _mm_cvttsd_si64(a);
                    let high = _mm_cvttsd_si64(#unpack_high(a, a));
                    _mm_set_epi64x(high, low).simd_into(#token)
                }
            });
            let fallback = fallback_method(op, vec_ty);
            return quote! {
                #[cfg(target_arch = "x86_64")]
                #specialized
                #[cfg(target_arch = "x86")]
                #fallback
            };
        }

        // Deal with whole-method scalar fallbacks before constructing an intrinsic-backed kernel.
        // Some conversions have no native SIMD versions, or are slower than scalar fallbacks.
        if matches!(
            conversion,
            (Self::Sse2, _, _, 64, 128, _)
                | (Self::Sse4_2, Float, Int | Unsigned, 64, 128, _)
                | (Self::Sse2, _, _, 32, 128, Precise)
                | (Self::Sse2, Unsigned, _, 32, 128, _)
                | (Self::Sse2, _, Unsigned, 32, 128, _)
        ) {
            // These conversions have no hardware support, or their native implementation is
            // slower than the scalar fallback.
            return fallback_method(op, vec_ty);
        }

        self.kernel_method(op, vec_ty, |token| match conversion {
            (Self::Avx512, Unsigned, Float, 32, bits @ (128 | 256), _) => {
                // AVX-512 unsigned integer -> float for narrow vectors.
                // We cannot emit the intrinsics for the conversion instructions
                // because the required intrinsics are mysteriously absent from stdarch:
                // https://github.com/rust-lang/rust/issues/158196
                // Fortunately LLVM optimizes this sequence into the single instruction we're after.
                // TODO: switch to intrinsics once they're added, stabilized, and our MSRV is high enough.
                let zext = format_ident!("_mm512_zextsi{bits}_si512");
                let convert = intrinsic_ident("cvtepu32", "ps", 512);
                let cast = format_ident!("_mm512_castps512_ps{bits}");
                quote! {
                    #cast(#convert(#zext(a.into()))).simd_into(#token)
                }
            }
            (
                Self::Avx512,
                source @ (Int | Unsigned),
                Float,
                scalar_bits @ (32 | 64),
                128 | 256 | 512,
                _,
            ) => {
                // native AVX-512 conversions with available intrinsics
                let target_ty = vec_ty.cast(target_scalar);
                let convert = simple_intrinsic(
                    &format!("cvtep{}{scalar_bits}", source.prefix()),
                    &target_ty,
                );
                quote! {
                    #convert(a.into()).simd_into(#token)
                }
            }
            (
                Self::Avx512,
                Float,
                Int,
                scalar_bits @ (32 | 64),
                128 | 256 | 512,
                Precise,
            ) => {
                // The truncating signed conversion already returns iN::MIN for negative overflow.
                // Mask positive overflow to iN::MAX, then replace NaN with zero to complete Rust's
                // float-to-integer cast semantics.
                let target_ty = vec_ty.cast(target_scalar);
                let float_suffix = op_suffix(Float, scalar_bits, true);
                let int_suffix = op_suffix(Int, scalar_bits, true);
                let masked_convert = intrinsic_ident(
                    &format!("mask_cvtt{float_suffix}"),
                    int_suffix,
                    vec_ty.n_bits(),
                );
                let cmp = intrinsic_ident(
                    "cmp",
                    &format!("{float_suffix}_mask"),
                    vec_ty.n_bits(),
                );
                let blend = avx512_mask_blend_intrinsic(&target_ty);
                let set1_float = set1_intrinsic(vec_ty);
                let set1_int = set1_intrinsic(&target_ty);
                let set0_int =
                    intrinsic_ident("setzero", coarse_type(&target_ty), target_ty.n_bits());
                // `2^(N - 1)` is exactly representable in both f32 and f64, and is the first
                // positive value outside the corresponding signed integer range.
                // Lanes at or above this threshold retain the merge value `iN::MAX`.
                // Negative overflow can use the truncating instruction's signed indefinite result, `iN::MIN`;
                // NaN is replaced with zero below.
                let upper_bound = match scalar_bits {
                    32 => quote! { 2147483648.0 }, // 2^31
                    64 => quote! { 9223372036854775808.0 }, // 2^63
                    _ => unreachable!(),
                };
                let int_max = match scalar_bits {
                    32 => quote! { i32::MAX },
                    64 => quote! { i64::MAX },
                    _ => unreachable!(),
                };
                let lt = avx512_float_compare_predicate("simd_lt");
                let ord = avx512_float_compare_predicate("ord");
                quote! {
                    let a = a.into();
                    let in_range = #cmp::<#lt>(a, #set1_float(#upper_bound));
                    let mut converted = #masked_convert(#set1_int(#int_max), in_range, a);
                    let is_not_nan = #cmp::<#ord>(a, a);
                    converted = #blend(is_not_nan, #set0_int(), converted);
                    converted.simd_into(#token)
                }
            }
            (
                Self::Avx512,
                Float,
                Int,
                scalar_bits @ (32 | 64),
                128 | 256 | 512,
                Approx,
            ) => {
                let target_ty = vec_ty.cast(target_scalar);
                let float_suffix = op_suffix(Float, scalar_bits, true);
                let convert = simple_intrinsic(&format!("cvtt{float_suffix}"), &target_ty);
                quote! {
                    #convert(a.into()).simd_into(#token)
                }
            }
            (
                Self::Avx512,
                Float,
                Unsigned,
                scalar_bits @ (32 | 64),
                128 | 256 | 512,
                Precise,
            ) => {
                // AVX-512's unsigned indefinite result is already uN::MAX, so positive overflow
                // needs no correction. Masking the conversion with an ordered positive comparison
                // makes every negative, zero, and NaN lane zero as required by Rust casts.
                let float_suffix = op_suffix(Float, scalar_bits, true);
                let int_suffix = op_suffix(Unsigned, scalar_bits, true);
                let cmp = intrinsic_ident(
                    "cmp",
                    &format!("{float_suffix}_mask"),
                    vec_ty.n_bits(),
                );
                let convert = intrinsic_ident(
                    &format!("maskz_cvtt{float_suffix}"),
                    int_suffix,
                    vec_ty.n_bits(),
                );
                let set0_float = intrinsic_ident("setzero", coarse_type(vec_ty), vec_ty.n_bits());
                let lt = avx512_float_compare_predicate("simd_lt");
                quote! {
                    let a = a.into();
                    let positive = #cmp::<#lt>(#set0_float(), a);
                    #convert(positive, a).simd_into(#token)
                }
            }
            (
                Self::Avx512,
                Float,
                Unsigned,
                scalar_bits @ (32 | 64),
                128 | 256 | 512,
                Approx,
            ) => {
                let target_ty = vec_ty.cast(target_scalar);
                let float_suffix = op_suffix(Float, scalar_bits, true);
                let convert = simple_intrinsic(&format!("cvtt{float_suffix}"), &target_ty);
                quote! {
                    #convert(a.into()).simd_into(#token)
                }
            }
            (Self::Sse4_2, source @ (Int | Unsigned), Float, 64, 128, _)
            | (Self::Avx2, source @ (Int | Unsigned), Float, 64, 128 | 256, _) => {
                // Before AVX-512 there's no packed 64-bit integer -> f64 instruction.
                // Split each integer into 32-bit halves, embed those halves in the fraction fields
                // of exact powers of two, then remove the biases. This is LLVM's u64 expansion,
                // extended to signed values by biasing the high half around i32::MIN.
                //
                // The subtraction is exact; the final addition performs the one rounding required
                // by the integer -> f64 conversion.
                // https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/llvm/lib/CodeGen/SelectionDAG/TargetLowering.cpp#L8670-L8726
                let target_ty = vec_ty.cast(target_scalar);
                let bits = vec_ty.n_bits();
                let set1_int = set1_intrinsic(vec_ty);
                let set1_float = set1_intrinsic(&target_ty);
                let srli = intrinsic_ident("srli", "epi64", bits);
                let xor = intrinsic_ident("xor", coarse_type(vec_ty), bits);
                let cast_to_float = cast_ident(Int, Float, 64, 64, bits);
                let sub = simple_intrinsic("sub", &target_ty);
                let add = simple_intrinsic("add", &target_ty);
                let (blend, blend_mask) = match bits {
                    128 => (intrinsic_ident("blend", "epi16", bits), quote! { 0xcc }),
                    256 => (intrinsic_ident("blend", "epi32", bits), quote! { 0xaa }),
                    _ => unreachable!(),
                };
                let (high_bias_bits, combined_bias_bits) = match source {
                    Unsigned => (0x4530_0000_0000_0000_i64, 0x4530_0000_0010_0000_u64),
                    Int => (0x4530_0000_8000_0000_i64, 0x4530_0000_8010_0000_u64),
                    _ => unreachable!(),
                };

                quote! {
                    let a = a.into();
                    let low = #blend::<#blend_mask>(a, #set1_int(0x4330_0000_0000_0000));
                    let high = #srli::<32>(a);
                    let high = #xor(high, #set1_int(#high_bias_bits));
                    let high = #sub(
                        #cast_to_float(high),
                        #set1_float(f64::from_bits(#combined_bias_bits)),
                    );
                    #add(#cast_to_float(low), high).simd_into(#token)
                }
            }
            (
                Self::Avx2,
                Float,
                target @ (Int | Unsigned),
                64,
                128 | 256,
                precision,
            ) => {
                // Recover the integer magnitude directly from the binary64 exponent and
                // significand. AVX2 variable shifts produce zero for counts above 63, so doing
                // both shift directions and ORing them avoids a per-lane branch or blend:
                //
                // magnitude = (significand >> (1075 - exponent))
                //           | (significand << (exponent - 1075))
                //
                // Subnormals, values below one, infinities, and NaNs naturally become zero.
                // This adapts LLVM's/compiler-rt's scalar significand-and-exponent expansion to
                // AVX2's packed variable shifts.
                // https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/llvm/lib/CodeGen/SelectionDAG/TargetLowering.cpp#L8497-L8565
                let target_ty = vec_ty.cast(target);
                let bits = vec_ty.n_bits();
                let set1_int = set1_intrinsic(&target_ty);
                let set1_float = set1_intrinsic(vec_ty);
                let setzero = intrinsic_ident("setzero", coarse_type(&target_ty), bits);
                let and = intrinsic_ident("and", coarse_type(&target_ty), bits);
                let or = intrinsic_ident("or", coarse_type(&target_ty), bits);
                let xor = intrinsic_ident("xor", coarse_type(&target_ty), bits);
                let sub = intrinsic_ident("sub", "epi64", bits);
                let srli = intrinsic_ident("srli", "epi64", bits);
                let srlv = intrinsic_ident("srlv", "epi64", bits);
                let sllv = intrinsic_ident("sllv", "epi64", bits);
                let cmpgt = intrinsic_ident("cmpgt", "epi64", bits);
                let blend = intrinsic_ident("blendv", "epi8", bits);
                let cast_to_int = cast_ident(Float, target, 64, 64, bits);
                let cast_to_float = cast_ident(target, Float, 64, 64, bits);
                let cmpge = float_compare_method("simd_ge", vec_ty);

                match (target, precision) {
                    (Unsigned, precision) => {
                        let finish = match precision {
                            Approx => quote! { converted },
                            Precise => quote! {
                                // Keeping the sign bit in `exponent` makes all negative lanes zero.
                                // NaN is also already zero; only positive overflow needs correction.
                                let overflow = #cast_to_int(#cmpge(
                                    a,
                                    #set1_float(18_446_744_073_709_551_616.0),
                                ));
                                #or(converted, overflow)
                            },
                        };

                        quote! {
                            let a = a.into();
                            let bits = #cast_to_int(a);
                            // Deliberately retain the sign bit. For a negative lane it makes both
                            // variable-shift counts exceed 63, producing zero.
                            let exponent = #srli::<52>(bits);
                            let significand = #or(
                                #and(bits, #set1_int(0x000f_ffff_ffff_ffff)),
                                #set1_int(0x0010_0000_0000_0000),
                            );
                            let shift_bias = #set1_int(1075);
                            let right_count = #sub(shift_bias, exponent);
                            let left_count = #sub(exponent, shift_bias);
                            let converted = #or(
                                #srlv(significand, right_count),
                                #sllv(significand, left_count),
                            );
                            #finish.simd_into(#token)
                        }
                    }
                    (Int, precision) => {
                        let finish = match precision {
                            Approx => quote! { converted },
                            Precise => quote! {
                                // At |a| >= 2^63, choose MIN or MAX from the original sign. The
                                // ordered comparison is false for NaN, whose magnitude is already 0.
                                let overflow = #cast_to_int(#cmpge(
                                    #cast_to_float(absolute),
                                    #set1_float(9_223_372_036_854_775_808.0),
                                ));
                                let bound = #xor(#set1_int(i64::MAX), sign);
                                #blend(converted, bound, overflow)
                            },
                        };

                        quote! {
                            let a = a.into();
                            let bits = #cast_to_int(a);
                            let absolute = #and(bits, #set1_int(i64::MAX));
                            let exponent = #srli::<52>(absolute);
                            let significand = #or(
                                #and(bits, #set1_int(0x000f_ffff_ffff_ffff)),
                                #set1_int(0x0010_0000_0000_0000),
                            );
                            let shift_bias = #set1_int(1075);
                            let right_count = #sub(shift_bias, exponent);
                            let left_count = #sub(exponent, shift_bias);
                            let magnitude = #or(
                                #srlv(significand, right_count),
                                #sllv(significand, left_count),
                            );
                            let sign = #cmpgt(#setzero(), bits);
                            let converted = #sub(#xor(magnitude, sign), sign);
                            #finish.simd_into(#token)
                        }
                    }
                    _ => unreachable!(),
                }
            }
            (
                Self::Sse2 | Self::Sse4_2,
                Float,
                target @ (Int | Unsigned),
                32,
                128,
                precision,
            )
            | (
                Self::Avx2,
                Float,
                target @ (Int | Unsigned),
                32,
                128 | 256,
                precision,
            ) => {
                // SSE and AVX2 only provide packed f32 -> i32 conversion.
                // Build unsigned and precise semantics around that signed primitive.
                let target_ty = vec_ty.cast(target);
                let max = simple_intrinsic("max", vec_ty);
                let set0 = intrinsic_ident("setzero", coarse_type(vec_ty), vec_ty.n_bits());
                let cmplt = float_compare_method("simd_lt", vec_ty);
                let cmple = float_compare_method("simd_le", vec_ty);
                let cmpord = float_compare_method("ord", vec_ty);
                let set1_float = set1_intrinsic(vec_ty);
                let set1_int = set1_intrinsic(&target_ty);
                let movemask = simple_intrinsic("movemask", vec_ty);
                let all_ones = match (vec_ty.n_bits(), vec_ty.scalar_bits) {
                    (128, 32) => quote! { 0b1111 },
                    (256, 32) => quote! { 0b11111111 },
                    _ => unimplemented!(),
                };
                let convert = simple_sign_unaware_intrinsic("cvttps", &target_ty);
                let cast_to_int = cast_ident(
                    vec_ty.scalar,
                    target_scalar,
                    vec_ty.scalar_bits,
                    vec_ty.scalar_bits,
                    vec_ty.n_bits(),
                );
                let blend = intrinsic_ident("blendv", "epi8", vec_ty.n_bits());
                let and = intrinsic_ident("and", coarse_type(&target_ty), vec_ty.n_bits());
                let xor = intrinsic_ident("xor", coarse_type(&target_ty), vec_ty.n_bits());
                let andnot = simple_intrinsic("andnot", vec_ty);
                let add_int = simple_sign_unaware_intrinsic("add", &target_ty);
                let sub_float = simple_intrinsic("sub", vec_ty);

                match (target, precision) {
                    (Int, Approx) => {
                        quote! {
                            #convert(a.into()).simd_into(#token)
                        }
                    }
                    (Unsigned, Approx) => {
                        quote! {
                            let mut converted = #convert(a.into());

                            // In the common case where everything is in range of an i32, we don't need to do anything else.
                            let in_range = #cmplt(a.into(), #set1_float(2147483648.0));
                            let all_in_range = #movemask(in_range) == #all_ones;

                            if !all_in_range {
                                // Add any excess (beyond the maximum value)
                                let excess = #sub_float(a.into(), #set1_float(2147483648.0));
                                let excess_converted = #convert(#andnot(in_range, excess));
                                converted = #add_int(converted, excess_converted);
                            }

                            converted.simd_into(#token)
                        }
                    }
                    (Int, Precise) => {
                        quote! {
                            let a = a.into();
                            let converted = #convert(a);
                            // The truncating instruction returns i32::MIN for every invalid lane.
                            // Flipping its bits for positive overflow turns that sentinel into
                            // i32::MAX. Negative overflow already has the desired value.
                            let positive_overflow = #cast_to_int(#cmple(
                                #set1_float(2147483648.0),
                                a,
                            ));
                            let converted = #xor(converted, positive_overflow);
                            // The ordered mask is false only for NaN, which Rust converts to zero.
                            let is_not_nan = #cast_to_int(#cmpord(a, a));
                            #and(converted, is_not_nan).simd_into(#token)
                        }
                    }
                    (Unsigned, Precise) => {
                        quote! {
                            // Clamp out-of-range values (and NaN) to 0. Intel's `_mm_max_ps` always takes the second
                            // operand if the first is NaN.
                            let a = #max(a.into(), #set0());
                            let mut converted = #convert(a);

                            // In the common case where everything is in range of an i32, we don't need to do anything else.
                            let in_range = #cmplt(a, #set1_float(2147483648.0));
                            let all_in_range = #movemask(in_range) == #all_ones;

                            if !all_in_range {
                                let exceeds_unsigned_range = #cast_to_int(#cmplt(#set1_float(4294967040.0), a));
                                // Add any excess (beyond the maximum value)
                                let excess = #sub_float(a, #set1_float(2147483648.0));
                                let excess_converted = #convert(#andnot(in_range, excess));

                                // Clamp to u32::MAX.
                                converted = #add_int(converted, excess_converted);
                                converted = #blend(converted, #set1_int(u32::MAX.cast_signed()), exceeds_unsigned_range);
                            }

                            converted.simd_into(#token)
                        }
                    }
                    _ => unreachable!(),
                }
            }
            (Self::Sse2 | Self::Sse4_2, Int, Float, 32, 128, _)
            | (Self::Avx2, Int, Float, 32, 128 | 256, _) => {
                let target_ty = vec_ty.cast(target_scalar);
                let intrinsic = simple_intrinsic("cvtepi32", &target_ty);
                quote! {
                    #intrinsic(a.into()).simd_into(#token)
                }
            }
            (Self::Sse4_2, Unsigned, Float, 32, 128, _)
            | (Self::Avx2, Unsigned, Float, 32, 128 | 256, _) => {
                let target_ty = vec_ty.cast(target_scalar);
                let set1_int = set1_intrinsic(vec_ty);
                let set1_float = set1_intrinsic(&target_ty);
                let add_float = simple_intrinsic("add", &target_ty);
                let sub_float = simple_intrinsic("sub", &target_ty);
                let blend = intrinsic_ident("blend", "epi16", vec_ty.n_bits());
                let srli = intrinsic_ident("srli", "epi32", vec_ty.n_bits());
                let cast_to_float = cast_ident(
                    vec_ty.scalar,
                    target_scalar,
                    vec_ty.scalar_bits,
                    vec_ty.scalar_bits,
                    vec_ty.n_bits(),
                );

                // Magical mystery algorithm taken from LLVM:
                // https://github.com/llvm/llvm-project/blob/6f8e87b9d097c5ef631f24d2eb2f34eb31b54d3b/llvm/lib/Target/X86/X86ISelLowering.cpp
                // (The file is too big for GitHub to show a preview, so no line numbers.)
                quote! {
                    let a = a.into();
                    let lo = #blend::<0xAA>(a, #set1_int(0x4B000000));
                    let hi = #blend::<0xAA>(#srli::<16>(a), #set1_int(0x53000000));

                    let fhi = #sub_float(#cast_to_float(hi), #set1_float(f32::from_bits(0x53000080)));
                    let result = #add_float(#cast_to_float(lo), fhi);

                    result.simd_into(#token)
                }
            }
            _ => unreachable!(),
        })
    }

    pub(crate) fn handle_mask_reduce(
        &self,
        method_op: Op,
        vec_ty: &VecType,
        quantifier: Quantifier,
        condition: bool,
    ) -> TokenStream {
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "mask reduce ops only operate on masks"
        );

        if *self == Self::Avx512 {
            let lane_mask = avx512_mask_lane_bits(vec_ty);
            let bits = avx512_mask_bits_expr(quote! { a });
            let expr = match (quantifier, condition) {
                (Quantifier::Any, true) => quote! { bits != 0 },
                (Quantifier::Any, false) => quote! { bits != #lane_mask },
                (Quantifier::All, true) => quote! { bits == #lane_mask },
                (Quantifier::All, false) => quote! { bits == 0 },
            };
            let method_sig = method_op.simd_trait_method_sig(vec_ty);
            return quote! {
            #method_sig {
                let bits = #bits & #lane_mask;
                #expr
                }
            };
        }

        let (movemask, all_ones) = match vec_ty.scalar_bits {
            32 | 64 => {
                let float_ty = vec_ty.cast(ScalarType::Float);
                let movemask = simple_intrinsic("movemask", &float_ty);
                let cast = cast_ident(
                    ScalarType::Mask,
                    ScalarType::Float,
                    vec_ty.scalar_bits,
                    vec_ty.scalar_bits,
                    vec_ty.n_bits(),
                );
                let movemask = quote! { #movemask(#cast(a.into())) };
                let all_ones = match vec_ty.len {
                    2 => quote! { 0b11 },
                    4 => quote! { 0b1111 },
                    8 => quote! { 0b11111111 },
                    _ => unimplemented!(),
                };

                (movemask, all_ones)
            }
            8 | 16 => {
                let bits_ty = vec_ty.reinterpret(ScalarType::Int, 8);
                let movemask = simple_intrinsic("movemask", &bits_ty);
                let movemask = quote! { #movemask(a.into()) };
                let all_ones = match vec_ty.n_bits() {
                    128 => quote! { 0xffff },
                    256 => quote! { 0xffffffff },
                    _ => unimplemented!(),
                };

                (movemask, all_ones)
            }
            _ => unreachable!(),
        };

        let op = match (quantifier, condition) {
            (Quantifier::Any, true) => quote! { != 0 },
            (Quantifier::Any, false) => quote! { != #all_ones },
            (Quantifier::All, true) => quote! { == #all_ones },
            (Quantifier::All, false) => quote! { == 0 },
        };

        self.kernel_method(method_op, vec_ty, |_| quote! { #movemask as u32 #op })
    }

    pub(crate) fn handle_load_interleaved(
        &self,
        op: Op,
        vec_ty: &VecType,
        block_size: u16,
        block_count: u16,
    ) -> TokenStream {
        assert_eq!(
            block_size, 128,
            "only 128-bit blocks are currently supported"
        );
        assert_eq!(block_count, 4, "only count of 4 is currently supported");
        if *self == Self::Avx512 {
            return self.handle_avx512_load_interleaved(op, vec_ty, block_size, block_count);
        }
        if *self == Self::Sse2 && matches!(vec_ty.scalar_bits, 8 | 16) {
            return fallback_method(op, vec_ty);
        }
        match vec_ty.scalar_bits {
            64 | 32 | 16 | 8 => {
                let avx2_64 = *self == Self::Avx2 && vec_ty.scalar_bits == 64;
                let block_len = if avx2_64 {
                    4
                } else {
                    block_size as usize / vec_ty.scalar_bits
                };
                let block_ty = VecType::new(vec_ty.scalar, vec_ty.scalar_bits, block_len);
                let scalar_ty = block_ty.scalar.rust(block_ty.scalar_bits);
                let native_ty = self.arch_ty(&block_ty);
                let native_block_ty = self.arch_ty(vec_ty);
                let vec_32 = block_ty.reinterpret(block_ty.scalar, 32);
                let unpacklo_32 = simple_sign_unaware_intrinsic("unpacklo", &vec_32);
                let unpackhi_32 = simple_sign_unaware_intrinsic("unpackhi", &vec_32);
                let vec_64 = block_ty.reinterpret(block_ty.scalar, 64);
                let unpacklo_64 = simple_sign_unaware_intrinsic("unpacklo", &vec_64);
                let unpackhi_64 = simple_sign_unaware_intrinsic("unpackhi", &vec_64);
                let permute_128 = intrinsic_ident(
                    match vec_ty.scalar {
                        ScalarType::Float => "permute2f128",
                        _ => "permute2x128",
                    },
                    coarse_type(&block_ty),
                    256,
                );

                if avx2_64 {
                    return self.kernel_method(op, vec_ty, |token| {
                        quote! {
                            let (chunks, []) = src.as_chunks::<4>() else {
                                unreachable!()
                            };
                            let v0: #native_ty = crate::transmute::checked_transmute_copy::<[#scalar_ty; 4], #native_ty>(&chunks[0]);
                            let v1: #native_ty = crate::transmute::checked_transmute_copy::<[#scalar_ty; 4], #native_ty>(&chunks[1]);

                            let lo = #unpacklo_64(v0, v1); // [0,4,2,6]
                            let hi = #unpackhi_64(v0, v1); // [1,5,3,7]
                            let out0 = #permute_128::<0x20>(lo, hi); // [0,4,1,5]
                            let out1 = #permute_128::<0x31>(lo, hi); // [2,6,3,7]
                            let outputs: [#native_block_ty; 4] =
                                crate::transmute::checked_transmute_copy(&[out0, out1]);

                            [
                                outputs[0].simd_into(#token),
                                outputs[1].simd_into(#token),
                                outputs[2].simd_into(#token),
                                outputs[3].simd_into(#token),
                            ]
                        }
                    });
                }

                let init_shuffle = match vec_ty.scalar_bits {
                    16 => Some(quote! {
                        let mask = _mm_setr_epi8(
                            0, 1, 8, 9,
                            2, 3, 10, 11,
                            4, 5, 12, 13,
                            6, 7, 14, 15,
                        );
                        let v0 = _mm_shuffle_epi8(v0, mask);
                        let v1 = _mm_shuffle_epi8(v1, mask);
                        let v2 = _mm_shuffle_epi8(v2, mask);
                        let v3 = _mm_shuffle_epi8(v3, mask);
                    }),
                    8 => Some(quote! {
                        let mask = _mm_setr_epi8(
                            0, 4, 8, 12,
                            1, 5, 9, 13,
                            2, 6, 10, 14,
                            3, 7, 11, 15,
                        );
                        let v0 = _mm_shuffle_epi8(v0, mask);
                        let v1 = _mm_shuffle_epi8(v1, mask);
                        let v2 = _mm_shuffle_epi8(v2, mask);
                        let v3 = _mm_shuffle_epi8(v3, mask);
                    }),
                    _ => None,
                };

                let initial_unpack = if vec_ty.scalar_bits == 64 {
                    None
                } else {
                    Some(quote! {
                        let tmp0 = #unpacklo_32(v0, v1); // [0,4,1,5]
                        let tmp1 = #unpackhi_32(v0, v1); // [2,6,3,7]
                        let tmp2 = #unpacklo_32(v2, v3); // [8,12,9,13]
                        let tmp3 = #unpackhi_32(v2, v3); // [10,14,11,15]
                    })
                };

                let final_unpack = match (vec_ty.scalar, vec_ty.scalar_bits) {
                    (_, 64) => quote! {
                        let out0 = #unpacklo_64(v0, v2); // [0,4]
                        let out1 = #unpackhi_64(v0, v2); // [1,5]
                        let out2 = #unpacklo_64(v1, v3); // [2,6]
                        let out3 = #unpackhi_64(v1, v3); // [3,7]
                    },
                    (ScalarType::Float, 32) => {
                        // The second stage needs a 64-bit unpack so each pair of f32 lanes
                        // moves together. x86 exposes that as the f64 `pd` intrinsic, whose
                        // register type differs from f32 `ps`, so cast around the unpack.
                        let cast_32 = cast_ident(
                            ScalarType::Float,
                            ScalarType::Float,
                            64,
                            32,
                            block_ty.n_bits(),
                        );
                        let cast_64 = cast_ident(
                            ScalarType::Float,
                            ScalarType::Float,
                            32,
                            64,
                            block_ty.n_bits(),
                        );

                        quote! {
                            let out0 = #cast_32(#unpacklo_64(#cast_64(tmp0), #cast_64(tmp2))); // [0,4,8,12]
                            let out1 = #cast_32(#unpackhi_64(#cast_64(tmp0), #cast_64(tmp2))); // [1,5,9,13]
                            let out2 = #cast_32(#unpacklo_64(#cast_64(tmp1), #cast_64(tmp3))); // [2,6,10,14]
                            let out3 = #cast_32(#unpackhi_64(#cast_64(tmp1), #cast_64(tmp3))); // [3,7,11,15]
                        }
                    }
                    _ => quote! {
                        let out0 = #unpacklo_64(tmp0, tmp2); // [0,4,8,12]
                        let out1 = #unpackhi_64(tmp0, tmp2); // [1,5,9,13]
                        let out2 = #unpacklo_64(tmp1, tmp3); // [2,6,10,14]
                        let out3 = #unpackhi_64(tmp1, tmp3); // [3,7,11,15]
                    },
                };

                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        let (chunks, []) = src.as_chunks::<#block_len>() else {
                            unreachable!()
                        };
                        let v0: #native_ty = crate::transmute::checked_transmute_copy::<[#scalar_ty; #block_len], #native_ty>(
                            &chunks[0],
                        );
                        let v1: #native_ty = crate::transmute::checked_transmute_copy::<[#scalar_ty; #block_len], #native_ty>(
                            &chunks[1],
                        );
                        let v2: #native_ty = crate::transmute::checked_transmute_copy::<[#scalar_ty; #block_len], #native_ty>(
                            &chunks[2],
                        );
                        let v3: #native_ty = crate::transmute::checked_transmute_copy::<[#scalar_ty; #block_len], #native_ty>(
                            &chunks[3],
                        );

                        #init_shuffle

                        #initial_unpack
                        #final_unpack

                        [
                            out0.simd_into(#token),
                            out1.simd_into(#token),
                            out2.simd_into(#token),
                            out3.simd_into(#token),
                        ]
                    }
                })
            }
            _ => unimplemented!(),
        }
    }

    pub(crate) fn handle_avx512_load_interleaved(
        &self,
        op: Op,
        vec_ty: &VecType,
        block_size: u16,
        block_count: u16,
    ) -> TokenStream {
        assert_eq!(
            block_size, 128,
            "only 128-bit blocks are currently supported"
        );
        assert_eq!(block_count, 4, "only count of 4 is currently supported");
        assert_eq!(
            vec_ty.n_bits(),
            128,
            "AVX-512 interleaved loads return 128-bit vectors"
        );
        let total_ty = VecType::new(
            vec_ty.scalar,
            vec_ty.scalar_bits,
            vec_ty.len * block_count as usize,
        );
        let scalar_ty = vec_ty.scalar.rust(vec_ty.scalar_bits);
        let native_ty = self.arch_ty(&total_ty);
        let native_block_ty = self.arch_ty(vec_ty);
        let len = total_ty.len;
        let permute = avx512_permutexvar_intrinsic(&total_ty);
        let indices = avx512_index_vector(
            &total_ty,
            interleaved_load_indices(total_ty.len, block_count as usize),
        );

        self.kernel_method(op, vec_ty, |token| {
            quote! {
                let lanes: #native_ty =
                    crate::transmute::checked_transmute_copy::<[#scalar_ty; #len], #native_ty>(
                        src,
                    );
                let lanes = #permute(#indices, lanes);
                let outputs: [#native_block_ty; 4] =
                    crate::transmute::checked_transmute_copy(&lanes);
                [
                    outputs[0].simd_into(#token),
                    outputs[1].simd_into(#token),
                    outputs[2].simd_into(#token),
                    outputs[3].simd_into(#token),
                ]
            }
        })
    }

    pub(crate) fn handle_store_interleaved(
        &self,
        op: Op,
        vec_ty: &VecType,
        block_size: u16,
        block_count: u16,
    ) -> TokenStream {
        assert_eq!(
            block_size, 128,
            "only 128-bit blocks are currently supported"
        );
        assert_eq!(block_count, 4, "only count of 4 is currently supported");
        if *self == Self::Avx512 {
            return self.handle_avx512_store_interleaved(op, vec_ty, block_size, block_count);
        }
        if *self == Self::Sse2 && matches!(vec_ty.scalar_bits, 8 | 16) {
            return fallback_method(op, vec_ty);
        }
        match vec_ty.scalar_bits {
            64 | 32 | 16 | 8 => {
                let avx2_64 = *self == Self::Avx2 && vec_ty.scalar_bits == 64;
                let block_len = if avx2_64 {
                    4
                } else {
                    block_size as usize / vec_ty.scalar_bits
                };
                let block_ty = VecType::new(vec_ty.scalar, vec_ty.scalar_bits, block_len);
                let scalar_ty = block_ty.scalar.rust(block_ty.scalar_bits);
                let native_ty = self.arch_ty(&block_ty);
                let native_block_ty = self.arch_ty(vec_ty);
                let vec_32 = block_ty.reinterpret(block_ty.scalar, 32);
                let unpacklo_32 = simple_sign_unaware_intrinsic("unpacklo", &vec_32);
                let unpackhi_32 = simple_sign_unaware_intrinsic("unpackhi", &vec_32);
                let vec_64 = block_ty.reinterpret(block_ty.scalar, 64);
                let unpacklo_64 = simple_sign_unaware_intrinsic("unpacklo", &vec_64);
                let unpackhi_64 = simple_sign_unaware_intrinsic("unpackhi", &vec_64);
                let permute_128 = intrinsic_ident(
                    match vec_ty.scalar {
                        ScalarType::Float => "permute2f128",
                        _ => "permute2x128",
                    },
                    coarse_type(&block_ty),
                    256,
                );

                // For 64-bit values, AVX2 permits a more efficient implementation.
                // It is special-cased here as a full early return because plumbing it
                // through the rest of the logic in this function complicates it significantly.
                if avx2_64 {
                    return self.kernel_method(op, vec_ty, |token| {
                        quote! {
                            let _ = #token;
                            let inputs: [#native_block_ty; 4] = [
                                vectors[0].into(),
                                vectors[1].into(),
                                vectors[2].into(),
                                vectors[3].into(),
                            ];
                            let wide_inputs: [#native_ty; 2] =
                                crate::transmute::checked_transmute_copy(&inputs);
                            let v0 = wide_inputs[0];
                            let v1 = wide_inputs[1];

                            let lo = #permute_128::<0x20>(v0, v1); // [0,1,4,5]
                            let hi = #permute_128::<0x31>(v0, v1); // [2,3,6,7]
                            let out0 = #unpacklo_64(lo, hi); // [0,2,4,6]
                            let out1 = #unpackhi_64(lo, hi); // [1,3,5,7]

                            let (chunks, []) = dest.as_chunks_mut::<4>() else {
                                unreachable!()
                            };

                            crate::transmute::checked_transmute_store::<#native_ty, [#scalar_ty; 4]>(out0, &mut chunks[0]);
                            crate::transmute::checked_transmute_store::<#native_ty, [#scalar_ty; 4]>(out1, &mut chunks[1]);
                        }
                    });
                }

                let post_shuffle = match vec_ty.scalar_bits {
                    16 => Some(quote! {
                        let mask = _mm_setr_epi8(
                            0, 1, 4, 5,
                            8, 9, 12, 13,
                            2, 3, 6, 7,
                            10, 11, 14, 15,
                        );
                        let out0 = _mm_shuffle_epi8(out0, mask);
                        let out1 = _mm_shuffle_epi8(out1, mask);
                        let out2 = _mm_shuffle_epi8(out2, mask);
                        let out3 = _mm_shuffle_epi8(out3, mask);
                    }),
                    8 => Some(quote! {
                        let mask = _mm_setr_epi8(
                            0, 4, 8, 12,
                            1, 5, 9, 13,
                            2, 6, 10, 14,
                            3, 7, 11, 15,
                        );
                        let out0 = _mm_shuffle_epi8(out0, mask);
                        let out1 = _mm_shuffle_epi8(out1, mask);
                        let out2 = _mm_shuffle_epi8(out2, mask);
                        let out3 = _mm_shuffle_epi8(out3, mask);
                    }),
                    _ => None,
                };

                let initial_unpack = if vec_ty.scalar_bits == 64 {
                    None
                } else {
                    Some(quote! {
                        let tmp0 = #unpacklo_32(v0, v1); // [0,4,1,5]
                        let tmp1 = #unpackhi_32(v0, v1); // [2,6,3,7]
                        let tmp2 = #unpacklo_32(v2, v3); // [8,12,9,13]
                        let tmp3 = #unpackhi_32(v2, v3); // [10,14,11,15]
                    })
                };

                let final_unpack = match (vec_ty.scalar, vec_ty.scalar_bits) {
                    (_, 64) => quote! {
                        let out0 = #unpacklo_64(v0, v1); // [0,1]
                        let out1 = #unpacklo_64(v2, v3); // [2,3]
                        let out2 = #unpackhi_64(v0, v1); // [4,5]
                        let out3 = #unpackhi_64(v2, v3); // [6,7]
                    },
                    (ScalarType::Float, 32) => {
                        // The second stage needs a 64-bit unpack so each pair of f32 lanes
                        // moves together. x86 exposes that as the f64 `pd` intrinsic, whose
                        // register type differs from f32 `ps`, so cast around the unpack.
                        let cast_32 = cast_ident(
                            ScalarType::Float,
                            ScalarType::Float,
                            64,
                            32,
                            block_ty.n_bits(),
                        );
                        let cast_64 = cast_ident(
                            ScalarType::Float,
                            ScalarType::Float,
                            32,
                            64,
                            block_ty.n_bits(),
                        );

                        quote! {
                            let out0 = #cast_32(#unpacklo_64(#cast_64(tmp0), #cast_64(tmp2))); // [0,4,8,12]
                            let out1 = #cast_32(#unpackhi_64(#cast_64(tmp0), #cast_64(tmp2))); // [1,5,9,13]
                            let out2 = #cast_32(#unpacklo_64(#cast_64(tmp1), #cast_64(tmp3))); // [2,6,10,14]
                            let out3 = #cast_32(#unpackhi_64(#cast_64(tmp1), #cast_64(tmp3))); // [3,7,11,15]
                        }
                    }
                    _ => quote! {
                        let out0 = #unpacklo_64(tmp0, tmp2); // [0,4,8,12]
                        let out1 = #unpackhi_64(tmp0, tmp2); // [1,5,9,13]
                        let out2 = #unpacklo_64(tmp1, tmp3); // [2,6,10,14]
                        let out3 = #unpackhi_64(tmp1, tmp3); // [3,7,11,15]
                    },
                };

                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        let _ = #token;
                        let v0: #native_ty = vectors[0].into();
                        let v1: #native_ty = vectors[1].into();
                        let v2: #native_ty = vectors[2].into();
                        let v3: #native_ty = vectors[3].into();

                        #initial_unpack
                        #final_unpack

                        #post_shuffle

                        let (chunks, []) = dest.as_chunks_mut::<#block_len>() else {
                            unreachable!()
                        };

                        crate::transmute::checked_transmute_store::<#native_ty, [#scalar_ty; #block_len]>(out0, &mut chunks[0]);
                        crate::transmute::checked_transmute_store::<#native_ty, [#scalar_ty; #block_len]>(out1, &mut chunks[1]);
                        crate::transmute::checked_transmute_store::<#native_ty, [#scalar_ty; #block_len]>(out2, &mut chunks[2]);
                        crate::transmute::checked_transmute_store::<#native_ty, [#scalar_ty; #block_len]>(out3, &mut chunks[3]);
                    }
                })
            }
            _ => unimplemented!(),
        }
    }

    pub(crate) fn handle_avx512_store_interleaved(
        &self,
        op: Op,
        vec_ty: &VecType,
        block_size: u16,
        block_count: u16,
    ) -> TokenStream {
        assert_eq!(
            block_size, 128,
            "only 128-bit blocks are currently supported"
        );
        assert_eq!(block_count, 4, "only count of 4 is currently supported");
        assert_eq!(
            vec_ty.n_bits(),
            128,
            "AVX-512 interleaved stores accept 128-bit vectors"
        );
        let total_ty = VecType::new(
            vec_ty.scalar,
            vec_ty.scalar_bits,
            vec_ty.len * block_count as usize,
        );
        let scalar_ty = vec_ty.scalar.rust(vec_ty.scalar_bits);
        let native_ty = self.arch_ty(&total_ty);
        let native_block_ty = self.arch_ty(vec_ty);
        let len = total_ty.len;
        let permute = avx512_permutexvar_intrinsic(&total_ty);
        let indices = avx512_index_vector(
            &total_ty,
            interleaved_store_indices(total_ty.len, block_count as usize),
        );

        self.kernel_method(op, vec_ty, |token| {
            quote! {
                let _ = #token;
                let inputs: [#native_block_ty; 4] = [
                    vectors[0].into(),
                    vectors[1].into(),
                    vectors[2].into(),
                    vectors[3].into(),
                ];
                let lanes: #native_ty = crate::transmute::checked_transmute_copy(&inputs);
                let lanes = #permute(#indices, lanes);
                crate::transmute::checked_transmute_store::<#native_ty, [#scalar_ty; #len]>(
                    lanes,
                    dest,
                );
            }
        })
    }

    /// Generates versions of the "alignr" intrinsics that take the shift amount as a regular argument instead of a
    /// const generic argument, to make them easier to use in higher-level operations. These are low-level helpers that
    /// inherit the semantics of the underlying `alignr` intrinsics, so the argument order is backwards from ARM's
    /// `vext` and our `slide` operation, and the 256-bit AVX2 version still operates *within* 128-bit lanes.
    fn dyn_alignr_helpers(&self) -> TokenStream {
        let mut fns = vec![];
        let token_ty = self.token();

        let vec_widths: &[usize] = match self {
            Self::Sse2 | Self::Sse4_2 => &[128],
            Self::Avx2 => &[128, 256],
            Self::Avx512 => &[128, 256, 512],
        };

        for vec_ty in vec_widths
            .iter()
            .map(|n| VecType::new(ScalarType::Int, 8, *n / 8))
        {
            let arch_ty = self.arch_ty(&vec_ty);

            let helper_name = format_ident!("dyn_alignr_{}", vec_ty.n_bits());
            let shifts = (0_usize..16).map(|shift| {
                let shift_i32 = i32::try_from(shift).unwrap();
                if *self == Self::Sse2 {
                    let inverse_shift_i32 = i32::try_from(16 - shift).unwrap();
                    quote! {
                        #shift => {
                            let lo = _mm_srli_si128::<#shift_i32>(b);
                            let hi = _mm_slli_si128::<#inverse_shift_i32>(a);
                            _mm_or_si128(lo, hi)
                        }
                    }
                } else {
                    let alignr_intrinsic = simple_sign_unaware_intrinsic("alignr", &vec_ty);
                    quote! { #shift => #alignr_intrinsic::<#shift_i32>(a, b) }
                }
            });

            fns.push(quote! {
                crate::kernel!(
                    /// This is a version of the `alignr` intrinsic that takes a non-const shift argument. The shift is still
                    /// expected to be constant in practice, so the match statement will be optimized out. This exists because
                    /// Rust doesn't currently let you do math on const generics.
                    #[inline(always)]
                    fn #helper_name(token: #token_ty, a: #arch_ty, b: #arch_ty, shift: usize) -> #arch_ty {
                        match shift {
                            #(#shifts,)*
                            _ => unreachable!()
                        }
                    }
                );
            });
        }

        quote! { #( #fns )* }
    }

    fn sse_slide_helpers(token_ty: Ident) -> TokenStream {
        let mut fns = vec![];

        for num_blocks in [2_usize, 4_usize] {
            let helper_name = format_ident!("cross_block_alignr_128x{}", num_blocks);
            let blocks_idx = 0..num_blocks;

            // Unroll the construction of the blocks. I tried using `array::from_fn`, but the compiler thought the
            // closure was too big and didn't inline it.
            fns.push(quote! {
                crate::kernel!(
                    /// Concatenates `b` and `a` (each N blocks) and extracts N blocks starting at byte offset `shift_bytes`.
                    /// Extracts from [b : a] (b in low bytes, a in high bytes), matching `alignr` semantics.
                    #[inline(always)]
                    fn #helper_name(token: #token_ty, a: [__m128i; #num_blocks], b: [__m128i; #num_blocks], shift_bytes: usize) -> [__m128i; #num_blocks] {
                        [#({
                            let [lo, hi] = crate::support::cross_block_slide_blocks_at(&b, &a, #blocks_idx, shift_bytes);
                            dyn_alignr_128(token, hi, lo, shift_bytes % 16)
                        }),*]
                    }
                );
            });
        }

        quote! {
            #(#fns)*
        }
    }

    fn avx2_slide_helpers() -> TokenStream {
        quote! {
            crate::kernel!(
                /// Computes one output __m256i for `cross_block_alignr_*` operations.
                ///
                /// Given an array of registers, each containing two 128-bit blocks, extracts two adjacent blocks (`lo_idx` and
                /// `hi_idx` = `lo_idx + 1`) and performs `alignr` with `intra_shift`.
                #[inline(always)]
                fn cross_block_alignr_one(token: Avx2, regs: &[__m256i], block_idx: usize, shift_bytes: usize) -> __m256i {
                    let lo_idx = block_idx + (shift_bytes / 16);
                    let intra_shift = shift_bytes % 16;
                    let lo_blocks = if lo_idx & 1 == 0 {
                        regs[lo_idx / 2]
                    } else {
                        _mm256_permute2x128_si256::<0x21>(regs[lo_idx / 2], regs[(lo_idx / 2) + 1])
                    };

                    // For hi_blocks, we need blocks (`lo_idx + 1`) and (`lo_idx + 2`)
                    let hi_idx = lo_idx + 1;
                    let hi_blocks = if hi_idx & 1 == 0 {
                        regs[hi_idx / 2]
                    } else {
                        _mm256_permute2x128_si256::<0x21>(regs[hi_idx / 2], regs[(hi_idx / 2) + 1])
                    };

                    dyn_alignr_256(token, hi_blocks, lo_blocks, intra_shift)
                }
            );

            crate::kernel!(
                /// Concatenates `b` and `a` (each 2 x __m256i = 4 blocks) and extracts 4 blocks starting at byte offset
                /// `shift_bytes`. Extracts from [b : a] (b in low bytes, a in high bytes), matching alignr semantics.
                #[inline(always)]
                fn cross_block_alignr_256x2(token: Avx2, a: [__m256i; 2], b: [__m256i; 2], shift_bytes: usize) -> [__m256i; 2] {
                    // Concatenation is [b : a], so b blocks come first
                    let regs = [b[0], b[1], a[0], a[1]];

                    [
                        cross_block_alignr_one(token, &regs, 0, shift_bytes),
                        cross_block_alignr_one(token, &regs, 2, shift_bytes),
                    ]
                }
            );

            crate::kernel!(
                /// Concatenates `b` and `a` (each 1 x __m256i = 2 blocks) and extracts 2 blocks starting at byte offset
                /// `shift_bytes`. Extracts from [b : a] (b in low bytes, a in high bytes), matching alignr semantics.
                #[inline(always)]
                fn cross_block_alignr_256x1(token: Avx2, a: __m256i, b: __m256i, shift_bytes: usize) -> __m256i {
                    // Concatenation is [b : a], so b comes first
                    let regs = [b, a];

                    cross_block_alignr_one(token, &regs, 0, shift_bytes)
                }
            );
        }
    }
}
