// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::x86::{
    self, cast_ident, coarse_type, extend_intrinsic, float_compare_method, intrinsic_ident,
    op_suffix, pack_intrinsic, set1_intrinsic, simple_intrinsic, simple_sign_unaware_intrinsic,
    unpack_intrinsic,
};
use crate::generic::{
    generic_as_array, generic_block_combine, generic_block_split, generic_from_array,
    generic_from_bytes, generic_mask_set, generic_op_name, generic_store_array, generic_to_bytes,
    integer_lane_mask_splat_arg, scalar_binary,
};
use crate::level::Level;
use crate::ops::{Op, OpSig, Quantifier, SlideGranularity, valid_reinterpret};
use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{ToTokens as _, format_ident, quote};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum X86 {
    Sse4_2,
    Avx2,
    Avx512,
}

pub(crate) const AVX512_FEATURES: &str = "adx,aes,avx512bitalg,avx512bw,avx512cd,avx512dq,avx512f,avx512ifma,avx512vbmi,avx512vbmi2,avx512vl,avx512vnni,avx512vpopcntdq,bmi1,bmi2,cmpxchg16b,fma,gfni,lzcnt,movbe,pclmulqdq,popcnt,rdrand,rdseed,sha,vaes,vpclmulqdq,xsave,xsavec,xsaveopt,xsaves";

impl Level for X86 {
    fn name(&self) -> &'static str {
        match self {
            Self::Sse4_2 => "Sse4_2",
            Self::Avx2 => "Avx2",
            Self::Avx512 => "Avx512",
        }
    }

    fn native_width(&self) -> usize {
        match self {
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
            Self::Sse4_2 => "sse4.2,cmpxchg16b,popcnt",
            Self::Avx2 => "avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,lzcnt,movbe,popcnt,xsave",
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

    fn token_doc(&self) -> &'static str {
        match self {
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
        quote! {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::*;
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
            Self::Sse4_2 => Self::sse42_slide_helpers(),
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
        let n_bits = ty.n_bits();
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
        match self {
            Self::Sse4_2 => quote! {
                /// Create a SIMD token.
                ///
                /// # Safety
                ///
                /// The `sse4.2`, `cmpxchg16b`, and `popcnt` CPU features must
                /// be available.
                #[inline]
                pub const unsafe fn new_unchecked() -> Self {
                    Sse4_2 { _private: () }
                }
            },
            Self::Avx2 => quote! {
                /// Create a SIMD token.
                ///
                /// # Safety
                ///
                /// The `avx2`, `bmi1`, `bmi2`, `cmpxchg16b`, `f16c`, `fma`,
                /// `lzcnt`, `movbe`, `popcnt`, and `xsave` CPU features must
                /// be available.
                #[inline]
                pub const unsafe fn new_unchecked() -> Self {
                    Self { _private: () }
                }
            },
            Self::Avx512 => quote! {
                /// Create a SIMD token.
                ///
                /// # Safety
                ///
                /// The Ice Lake AVX-512 CPU feature set must be available.
                #[inline]
                pub const unsafe fn new_unchecked() -> Self {
                    Self { _private: () }
                }
            },
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
            OpSig::WidenNarrow { target_ty } => {
                self.handle_widen_narrow(op, method, vec_ty, target_ty)
            }
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
            OpSig::Cvt {
                target_ty,
                scalar_bits,
                precise,
            } => self.handle_cvt(op, vec_ty, target_ty, scalar_bits, precise),
            OpSig::Reinterpret {
                target_ty,
                scalar_bits,
            } => self.handle_reinterpret(self, op, vec_ty, target_ty, scalar_bits),
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
            OpSig::FromArray { kind }
                if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Mask =>
            {
                self.handle_avx512_mask_from_array(op, vec_ty, kind)
            }
            OpSig::FromArray { kind } => generic_from_array(method_sig, vec_ty, kind),
            OpSig::AsArray { kind }
                if *self == Self::Avx512 && vec_ty.scalar == ScalarType::Mask =>
            {
                self.handle_avx512_mask_as_array(op, vec_ty, kind)
            }
            OpSig::AsArray { kind } => {
                generic_as_array(method_sig, vec_ty, kind, self.max_block_size(), |vec_ty| {
                    self.arch_ty(vec_ty)
                })
            }
            OpSig::StoreArray => generic_store_array(method_sig, vec_ty),
            OpSig::FromBytes => generic_from_bytes(method_sig, vec_ty),
            OpSig::ToBytes => generic_to_bytes(method_sig, vec_ty),
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
        vec_ty.scalar == ScalarType::Mask && vec_ty.n_bits() == 512 && vec_ty.scalar_bits == 8
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

    pub(crate) fn handle_avx512_mask_from_array(
        &self,
        op: Op,
        vec_ty: &VecType,
        kind: crate::ops::RefKind,
    ) -> TokenStream {
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "AVX-512 mask array loads only operate on mask types"
        );
        let movepi_mask = intrinsic_ident(
            &format!("movepi{}", vec_ty.scalar_bits),
            "mask",
            vec_ty.n_bits(),
        );
        let transmute_src = if kind == crate::ops::RefKind::Value {
            quote! { &val }
        } else {
            quote! { val }
        };
        // Mask arrays are specified as either 0 or -1 per lane, so the sign bit is the
        // truth value. Other lane values have unspecified results.
        self.kernel_method(op, vec_ty, |token| {
            let result = avx512_mask_register_value_with_simd(
                vec_ty,
                quote! { #movepi_mask(lanes) },
                quote! { #token },
            );
            quote! {
                let lanes = crate::transmute::checked_transmute_copy(#transmute_src);
                #result
            }
        })
    }

    pub(crate) fn handle_avx512_mask_as_array(
        &self,
        op: Op,
        vec_ty: &VecType,
        kind: crate::ops::RefKind,
    ) -> TokenStream {
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "AVX-512 mask array stores only operate on mask types"
        );
        assert!(
            kind == crate::ops::RefKind::Value,
            "mask array references are not exposed"
        );
        let movm = intrinsic_ident(
            "movm",
            op_suffix(vec_ty.scalar, vec_ty.scalar_bits, true),
            vec_ty.n_bits(),
        );
        self.kernel_method(op, vec_ty, |_| {
            quote! {
                let lanes = #movm(a.val);
                crate::transmute::checked_transmute_copy(&lanes)
            }
        })
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
                "simd_lt" | "simd_gt" => {
                    let gt = simple_sign_unaware_intrinsic("cmpgt", vec_ty);

                    if vec_ty.scalar == ScalarType::Unsigned {
                        // Below AVX-512, we only have signed GT/LT, not unsigned.
                        let set = set1_intrinsic(vec_ty);
                        let sign = match vec_ty.scalar_bits {
                            8 => quote! { 0x80u8 },
                            16 => quote! { 0x8000u16 },
                            32 => quote! { 0x80000000u32 },
                            _ => unimplemented!(),
                        };
                        let xor_op = intrinsic_ident("xor", coarse_type(vec_ty), vec_ty.n_bits());
                        let args = if method == "simd_lt" {
                            quote! { b_signed, a_signed }
                        } else {
                            quote! { a_signed, b_signed }
                        };

                        quote! {
                            let sign_bit = #set(#sign.cast_signed());
                            let a_signed = #xor_op(a.into(), sign_bit);
                            let b_signed = #xor_op(b.into(), sign_bit);

                            #gt(#args)
                        }
                    } else {
                        let args = if method == "simd_lt" {
                            quote! { b.into(), a.into() }
                        } else {
                            quote! { a.into(), b.into() }
                        };
                        quote! {
                            #gt(#args)
                        }
                    }
                }
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

    pub(crate) fn handle_widen_narrow(
        &self,
        op: Op,
        method: &str,
        vec_ty: &VecType,
        target_ty: VecType,
    ) -> TokenStream {
        let dst_width = target_ty.n_bits();
        self.kernel_method(op, vec_ty, |token| match method {
            "widen" => {
                match (self, dst_width, vec_ty.n_bits()) {
                    (Self::Avx2 | Self::Avx512, 256, 128) => {
                        let extend = extend_intrinsic(
                            vec_ty.scalar,
                            vec_ty.scalar_bits,
                            target_ty.scalar_bits,
                            dst_width,
                        );
                        quote! {
                            #extend(a.into()).simd_into(#token)
                        }
                    }
                    (Self::Avx512, 512, 256) => {
                        let extend = extend_intrinsic(
                            vec_ty.scalar,
                            vec_ty.scalar_bits,
                            target_ty.scalar_bits,
                            dst_width,
                        );
                        quote! {
                            #extend(a.into()).simd_into(#token)
                        }
                    }
                    (Self::Avx2, 512, 256) => {
                        let extend = extend_intrinsic(
                            vec_ty.scalar,
                            vec_ty.scalar_bits,
                            target_ty.scalar_bits,
                            vec_ty.n_bits(),
                        );
                        let combine = generic_op_name(
                            "combine",
                            &vec_ty.reinterpret(vec_ty.scalar, vec_ty.scalar_bits * 2),
                        );
                        let split = generic_op_name("split", vec_ty);
                        quote! {
                            let (a0, a1) = #token.#split(a);
                            let high = #extend(a0.into()).simd_into(#token);
                            let low = #extend(a1.into()).simd_into(#token);
                            #token.#combine(high, low)
                        }
                    }
                    (Self::Sse4_2, 256, 128) => {
                        let extend = extend_intrinsic(
                            vec_ty.scalar,
                            vec_ty.scalar_bits,
                            target_ty.scalar_bits,
                            vec_ty.n_bits(),
                        );
                        let combine = generic_op_name(
                            "combine",
                            &vec_ty.reinterpret(vec_ty.scalar, vec_ty.scalar_bits * 2),
                        );
                        quote! {
                            let raw = a.into();
                            let high = #extend(raw).simd_into(#token);
                            // Shift by 8 since we want to get the higher part into the
                            // lower position.
                            let low = #extend(_mm_srli_si128::<8>(raw)).simd_into(#token);
                            #token.#combine(high, low)
                        }
                    }
                    _ => unimplemented!(),
                }
            }
            "narrow" => {
                match (self, dst_width, vec_ty.n_bits()) {
                    (Self::Avx512, 128, 256) | (Self::Avx512, 256, 512) => {
                        let narrow = intrinsic_ident("cvtepi16", "epi8", vec_ty.n_bits());
                        quote! {
                            #narrow(a.into()).simd_into(#token)
                        }
                    }
                    (Self::Avx2, 128, 256) => {
                        let mask = match target_ty.scalar_bits {
                            8 => {
                                quote! { 0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1 }
                            }
                            _ => unimplemented!(),
                        };
                        quote! {
                            let mask = _mm256_setr_epi8(#mask, #mask);

                            let shuffled = _mm256_shuffle_epi8(a.into(), mask);
                            let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(shuffled);

                            _mm256_castsi256_si128(packed).simd_into(#token)
                        }
                    }
                    (Self::Avx2, 256, 512) => {
                        let mask = set1_intrinsic(&VecType::new(
                            vec_ty.scalar,
                            vec_ty.scalar_bits,
                            vec_ty.len / 2,
                        ));
                        let pack = pack_intrinsic(
                            vec_ty.scalar_bits,
                            matches!(vec_ty.scalar, ScalarType::Int),
                            target_ty.n_bits(),
                        );
                        let split = generic_op_name("split", vec_ty);
                        quote! {
                            let (a, b) = #token.#split(a);
                            // Note that AVX2 only has an intrinsic for saturating cast,
                            // but not wrapping.
                            let mask = #mask(0xFF);
                            let lo_masked = _mm256_and_si256(a.into(), mask);
                            let hi_masked = _mm256_and_si256(b.into(), mask);
                            // The 256-bit version of packus_epi16 operates lane-wise, so we need to arrange things
                            // properly afterwards.
                            let result = _mm256_permute4x64_epi64::<0b_11_01_10_00>(#pack(lo_masked, hi_masked));
                            result.simd_into(#token)
                        }
                    }
                    (Self::Sse4_2, 128, 256) => {
                        let mask = set1_intrinsic(&VecType::new(
                            vec_ty.scalar,
                            vec_ty.scalar_bits,
                            vec_ty.len / 2,
                        ));
                        let pack = pack_intrinsic(
                            vec_ty.scalar_bits,
                            matches!(vec_ty.scalar, ScalarType::Int),
                            target_ty.n_bits(),
                        );
                        let split = generic_op_name("split", vec_ty);
                        quote! {
                            let (a, b) = #token.#split(a);
                            // Below AVX-512. we only have an intrinsic for saturating cast, but not wrapping.
                            let mask = #mask(0xFF);
                            let lo_masked = _mm_and_si128(a.into(), mask);
                            let hi_masked = _mm_and_si128(b.into(), mask);
                            let result = #pack(lo_masked, hi_masked);
                            result.simd_into(#token)
                        }
                    }
                    _ => unimplemented!(),
                }
            }
            _ => unreachable!(),
        })
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

        match method {
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
                // SSE2 has shift operations, but they shift every lane by the same amount, so we can't use them here.
                let body = match method {
                    "shlv" => scalar_binary(quote!(core::ops::Shl::shl)),
                    "shrv" => scalar_binary(quote!(core::ops::Shr::shr)),
                    _ => unreachable!(),
                };
                quote! {
                    #method_sig {
                        #body
                    }
                }
            }
            _ => self.kernel_method(op, vec_ty, |token| match method {
                "mul" if vec_ty.scalar_bits == 8 => {
                    // https://stackoverflow.com/questions/8193601/sse-multiplication-16-x-uint8-t
                    let mullo = intrinsic_ident("mullo", "epi16", vec_ty.n_bits());
                    let set1 = intrinsic_ident("set1", "epi16", vec_ty.n_bits());
                    let and = intrinsic_ident("and", coarse_type(vec_ty), vec_ty.n_bits());
                    let or = intrinsic_ident("or", coarse_type(vec_ty), vec_ty.n_bits());
                    let slli = intrinsic_ident("slli", "epi16", vec_ty.n_bits());
                    let srli = intrinsic_ident("srli", "epi16", vec_ty.n_bits());
                    quote! {
                        let dst_even = #mullo(a.into(), b.into());
                        let dst_odd = #mullo(#srli::<8>(a.into()), #srli::<8>(b.into()));

                        #or(#slli(dst_odd, 8), #and(dst_even, #set1(0xFF))).simd_into(#token)
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
        let unpack_hi = unpack_intrinsic(ScalarType::Int, 8, false, ty_bits);
        let unpack_lo = unpack_intrinsic(ScalarType::Int, 8, true, ty_bits);
        let set0 = intrinsic_ident("setzero", coarse_type(vec_ty), ty_bits);
        let and = intrinsic_ident("and", coarse_type(vec_ty), ty_bits);
        let set1_epi16 = intrinsic_ident("set1", "epi16", ty_bits);
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

    pub(crate) fn handle_shift(&self, op: Op, method: &str, vec_ty: &VecType) -> TokenStream {
        let shift_op = match (method, vec_ty.scalar) {
            ("shr", ScalarType::Unsigned) => "srl",
            ("shr", ScalarType::Int) => "sra",
            ("shl", _) => "sll",
            _ => unreachable!(),
        };
        let ty_bits = vec_ty.n_bits();
        let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits.max(16), false);
        let shift_intrinsic = intrinsic_ident(shift_op, suffix, ty_bits);

        if vec_ty.scalar_bits == 8 {
            // x86 doesn't have shifting for 8-bit, so we first convert into 16-bit, shift, and then back to 8-bit.

            let unpack_hi = unpack_intrinsic(ScalarType::Int, 8, false, ty_bits);
            let unpack_lo = unpack_intrinsic(ScalarType::Int, 8, true, ty_bits);

            let set0 = intrinsic_ident("setzero", coarse_type(vec_ty), ty_bits);
            let extend_expr = |expr| match vec_ty.scalar {
                ScalarType::Unsigned => quote! {
                    #expr(val, #set0())
                },
                ScalarType::Int => {
                    let sign_bits = if *self == Self::Avx512 && ty_bits == 512 {
                        quote! {
                            _mm512_movm_epi8(_mm512_cmpgt_epi8_mask(#set0(), val))
                        }
                    } else {
                        let cmp_intrinsic = intrinsic_ident("cmpgt", "epi8", ty_bits);
                        quote! { #cmp_intrinsic(#set0(), val) }
                    };
                    quote! {
                        #expr(val, #sign_bits)
                    }
                }
                _ => unimplemented!(),
            };

            let extend_intrinsic_lo = extend_expr(unpack_lo);
            let extend_intrinsic_hi = extend_expr(unpack_hi);
            let pack_intrinsic = pack_intrinsic(16, vec_ty.scalar == ScalarType::Int, ty_bits);

            self.kernel_method(op, vec_ty, |token| {
                quote! {
                    let val = a.into();
                    let shift_count = _mm_cvtsi32_si128(shift.cast_signed());

                    let lo_16 = #extend_intrinsic_lo;
                    let hi_16 = #extend_intrinsic_hi;

                    let lo_shifted = #shift_intrinsic(lo_16, shift_count);
                    let hi_shifted = #shift_intrinsic(hi_16, shift_count);

                    #pack_intrinsic(lo_shifted, hi_shifted).simd_into(#token)
                }
            })
        } else {
            self.kernel_method(
                op,
                vec_ty,
                |token| {
                    quote! { #shift_intrinsic(a.into(), _mm_cvtsi32_si128(shift.cast_signed())).simd_into(#token) }
                },
            )
        }
    }

    pub(crate) fn handle_ternary(
        &self,
        op: Op,
        method_sig: TokenStream,
        method: &str,
        vec_ty: &VecType,
    ) -> TokenStream {
        match method {
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
                let mask = match vec_ty.scalar_bits {
                    8 => quote! { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 },
                    16 => quote! { 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 },
                    _ => unreachable!(),
                };
                let shuf = |input: TokenStream| {
                    quote! {
                        _mm256_permute4x64_epi64::<0b11_01_10_00>(
                            _mm256_shuffle_epi8(#input, _mm256_setr_epi8(#mask, #mask)),
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
                    let mask = match vec_ty.scalar_bits {
                        8 => {
                            quote! { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 }
                        }
                        16 => {
                            quote! { 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 }
                        }
                        _ => unreachable!(),
                    };
                    let mask_reg = match vec_ty.n_bits() {
                        128 => quote! { _mm_setr_epi8(#mask) },
                        256 => quote! { _mm256_setr_epi8(#mask, #mask) },
                        _ => unreachable!(),
                    };
                    let shuffle_epi8 = intrinsic_ident("shuffle", "epi8", vec_ty.n_bits());

                    // Select either the low or high half of each one
                    let op = if select_even { "unpacklo" } else { "unpackhi" };
                    let unpack_epi64 = intrinsic_ident(op, "epi64", vec_ty.n_bits());

                    quote! {
                        let mask = #mask_reg;

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
        let to_bytes = generic_op_name("cvt_to_bytes", vec_ty);
        let from_bytes = generic_op_name("cvt_from_bytes", vec_ty);

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

                    let a = self.#to_bytes(a).val.0;
                    let b = self.#to_bytes(b).val.0;
                    let result = #alignr(self, b, a, #byte_shift);
                    self.#from_bytes(#combined_bytes { val: #block_wrapper(result), simd: self })
                }
            };
        }

        if *self == Self::Avx512 && granularity == AcrossBlocks && vec_ty.n_bits() >= 256 {
            let level = self.token();
            let ty = vec_ty.rust();
            let vec = quote! { #ty<#level> };
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
                                token.#to_bytes(a).val.0,
                                idx,
                                token.#to_bytes(b).val.0,
                            );
                            token.#from_bytes(#combined_bytes { val: #block_wrapper(result), simd: token })
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
            (AcrossBlocks, 256 | 512, Self::Sse4_2) => {
                // Inter-block shift or rotate in SSE4.2: use cross_block_alignr

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
                let result = #alignr_op(self, self.#to_bytes(b).val.0, self.#to_bytes(a).val.0, #byte_shift);
                self.#from_bytes(#combined_bytes { val: #block_wrapper(result), simd: self })
            }
        }
    }

    pub(crate) fn handle_swizzle_dyn_within_blocks(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let bytes_ty = vec_ty.bytes_ty();
        let bytes = bytes_ty.rust();
        let wrapper = bytes_ty.aligned_wrapper();
        let to_bytes = generic_op_name("cvt_to_bytes", vec_ty);
        let from_bytes = generic_op_name("cvt_from_bytes", vec_ty);

        self.kernel_method(op, vec_ty, |token| {
            let body = if *self == Self::Avx512 {
                match vec_ty.n_bits() {
                    128 => quote! {
                        let bytes = #token.#to_bytes(a).val.0;
                        let result = _mm_mask_shuffle_epi8(bytes, u16::MAX, bytes, indices.into());
                    },
                    256 => quote! {
                        let bytes = #token.#to_bytes(a).val.0;
                        let result = _mm256_mask_shuffle_epi8(bytes, u32::MAX, bytes, indices.into());
                    },
                    512 => quote! {
                        let result = _mm512_shuffle_epi8(#token.#to_bytes(a).val.0, indices.into());
                    },
                    _ => unreachable!(),
                }
            } else {
                let shuffle = simple_sign_unaware_intrinsic("shuffle", &bytes_ty);
                quote! {
                    let result = #shuffle(#token.#to_bytes(a).val.0, indices.into());
                }
            };

            quote! {
                #body
                #token.#from_bytes(#bytes { val: #wrapper(result), simd: #token })
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
        assert_eq!(
            vec_ty.scalar_bits, target_scalar_bits,
            "we currently only support converting between types of the same width"
        );

        if *self == Self::Avx512
            && vec_ty.scalar == ScalarType::Float
            && target_scalar == ScalarType::Unsigned
        {
            let target_ty = vec_ty.reinterpret(target_scalar, target_scalar_bits);
            let convert = intrinsic_ident("cvttps", "epu32", vec_ty.n_bits());
            return self.kernel_method(op, vec_ty, |token| {
                if precise {
                    let max = simple_intrinsic("max", vec_ty);
                    let cmp = intrinsic_ident("cmp", "ps_mask", vec_ty.n_bits());
                    let blend = avx512_mask_blend_intrinsic(&target_ty);
                    let set1_float = set1_intrinsic(vec_ty);
                    let set1_int = set1_intrinsic(&target_ty);
                    let set0_float =
                        intrinsic_ident("setzero", coarse_type(vec_ty), vec_ty.n_bits());
                    let lt = avx512_float_compare_predicate("simd_lt");
                    quote! {
                        let a = #max(a.into(), #set0_float());
                        let mut converted = #convert(a);
                        let exceeds_unsigned_range = #cmp::<#lt>(#set1_float(4294967040.0), a);
                        converted = #blend(
                            exceeds_unsigned_range,
                            converted,
                            #set1_int(u32::MAX.cast_signed()),
                        );
                        converted.simd_into(#token)
                    }
                } else {
                    quote! {
                        #convert(a.into()).simd_into(#token)
                    }
                }
            });
        }

        if *self == Self::Avx512
            && vec_ty.scalar == ScalarType::Float
            && target_scalar == ScalarType::Int
            && vec_ty.scalar_bits == 32
        {
            let target_ty = vec_ty.reinterpret(target_scalar, target_scalar_bits);
            let convert = intrinsic_ident("cvttps", "epi32", vec_ty.n_bits());
            return self.kernel_method(op, vec_ty, |token| {
                if precise {
                    let masked_convert = intrinsic_ident("mask_cvttps", "epi32", vec_ty.n_bits());
                    let cmp = intrinsic_ident("cmp", "ps_mask", vec_ty.n_bits());
                    let blend = avx512_mask_blend_intrinsic(&target_ty);
                    let set1_float = set1_intrinsic(vec_ty);
                    let set1_int = set1_intrinsic(&target_ty);
                    let set0_int =
                        intrinsic_ident("setzero", coarse_type(&target_ty), target_ty.n_bits());
                    let lt = avx512_float_compare_predicate("simd_lt");
                    let ord = avx512_float_compare_predicate("ord");
                    quote! {
                        let a = a.into();
                        let in_range = #cmp::<#lt>(a, #set1_float(2147483648.0));
                        let mut converted = #masked_convert(#set1_int(i32::MAX), in_range, a);
                        let is_not_nan = #cmp::<#ord>(a, a);
                        converted = #blend(is_not_nan, #set0_int(), converted);
                        converted.simd_into(#token)
                    }
                } else {
                    quote! {
                        #convert(a.into()).simd_into(#token)
                    }
                }
            });
        }

        if *self == Self::Avx512
            && matches!(vec_ty.n_bits(), 128 | 256)
            && vec_ty.scalar == ScalarType::Unsigned
            && target_scalar == ScalarType::Float
            && vec_ty.scalar_bits == 32
        {
            // We cannot emit the intrinsics for the conversion instructions
            // because the required intrinsics are mysteriously absent from stdarch:
            // https://github.com/rust-lang/rust/issues/158196
            // Fortunately LLVM optimizes this sequence into the single instruction we're after.
            // TODO: switch to intrinsics once they're added, stabilized, and our MSRV is high enough.
            let bits = vec_ty.n_bits();
            let zext = format_ident!("_mm512_zextsi{bits}_si512");
            let convert = intrinsic_ident("cvtepu32", "ps", 512);
            let cast = format_ident!("_mm512_castps512_ps{bits}");
            return self.kernel_method(op, vec_ty, |token| {
                quote! {
                    #cast(#convert(#zext(a.into()))).simd_into(#token)
                }
            });
        }

        if *self == Self::Avx512 && vec_ty.n_bits() == 512 {
            let target_ty = vec_ty.reinterpret(target_scalar, target_scalar_bits);
            return self.kernel_method(op, vec_ty, |token| match (vec_ty.scalar, target_scalar) {
                (ScalarType::Int, ScalarType::Float) => {
                    let intrinsic = simple_intrinsic("cvtepi32", &target_ty);
                    quote! {
                        #intrinsic(a.into()).simd_into(#token)
                    }
                }
                (ScalarType::Unsigned, ScalarType::Float) => {
                    let intrinsic = simple_intrinsic("cvtepu32", &target_ty);
                    quote! {
                        #intrinsic(a.into()).simd_into(#token)
                    }
                }
                _ => unimplemented!(),
            });
        }

        self.kernel_method(op, vec_ty, |token| match (vec_ty.scalar, target_scalar) {
            (ScalarType::Float, ScalarType::Int | ScalarType::Unsigned) => {
                let target_ty = vec_ty.reinterpret(target_scalar, target_scalar_bits);
                let max = simple_intrinsic("max", vec_ty);
                let set0 = intrinsic_ident("setzero", coarse_type(vec_ty), vec_ty.n_bits());
                let cmplt = float_compare_method("simd_lt", vec_ty);
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
                let andnot = simple_intrinsic("andnot", vec_ty);
                let add_int = simple_sign_unaware_intrinsic("add", &target_ty);
                let sub_float = simple_intrinsic("sub", vec_ty);

                match (target_scalar, precise) {
                    (ScalarType::Int, false) => {
                        quote! {
                            #convert(a.into()).simd_into(#token)
                        }
                    }
                    (ScalarType::Unsigned, false) => {
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
                    (ScalarType::Int, true) => {
                        quote! {
                            let a = a.into();

                            let mut converted = #convert(a);

                            // In the common case where everything is in range, we don't need to do anything else.
                            let in_range = #cmplt(a, #set1_float(2147483648.0));
                            let all_in_range = #movemask(in_range) == #all_ones;

                            if !all_in_range {
                                // If we are above i32::MAX (2147483647), clamp to it.
                                converted = #blend(#set1_int(i32::MAX), converted, #cast_to_int(in_range));
                                // Set NaN to 0. Using `and` seems slightly faster than `blend`.
                                let is_not_nan = #cast_to_int(#cmpord(a, a));
                                converted = #and(converted, is_not_nan);
                                // We don't need to handle negative overflow because Intel's "invalid result" sentinel
                                // value is -2147483648, which is what we want anyway.
                            }

                            converted.simd_into(#token)
                        }
                    }
                    (ScalarType::Unsigned, true) => {
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
            (ScalarType::Int, ScalarType::Float) => {
                assert_eq!(
                    vec_ty.scalar_bits, 32,
                    "i64 to f64 conversions do not exist until AVX-512 and require special consideration"
                );
                let target_ty = vec_ty.reinterpret(target_scalar, target_scalar_bits);
                let intrinsic = simple_intrinsic("cvtepi32", &target_ty);
                quote! {
                    #intrinsic(a.into()).simd_into(#token)
                }
            }
            (ScalarType::Unsigned, ScalarType::Float) => {
                assert_eq!(
                    vec_ty.scalar_bits, 32,
                    "u64 to f64 conversions do not exist until AVX-512 and require special consideration"
                );

                let target_ty = vec_ty.reinterpret(target_scalar, target_scalar_bits);
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
            _ => unimplemented!(),
        })
    }

    pub(crate) fn handle_reinterpret(
        &self,
        level: &impl Level,
        op: Op,
        vec_ty: &VecType,
        target_ty: ScalarType,
        scalar_bits: usize,
    ) -> TokenStream {
        let dst_ty = vec_ty.reinterpret(target_ty, scalar_bits);
        assert!(
            valid_reinterpret(vec_ty, target_ty, scalar_bits),
            "{vec_ty:?} must be reinterpretable as {dst_ty:?}"
        );

        if coarse_type(vec_ty) == coarse_type(&dst_ty) {
            let arch_ty = level.arch_ty(vec_ty);
            self.kernel_method(
                op,
                vec_ty,
                |token| quote! { #arch_ty::from(a).simd_into(#token) },
            )
        } else {
            let ident = cast_ident(
                vec_ty.scalar,
                target_ty,
                vec_ty.scalar_bits,
                scalar_bits,
                vec_ty.n_bits(),
            );
            self.kernel_method(
                op,
                vec_ty,
                |token| quote! { #ident(a.into()).simd_into(#token) },
            )
        }
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
        if *self == Self::Avx512 && vec_ty.n_bits() == 512 {
            return self.handle_avx512_load_interleaved(op, vec_ty, block_size, block_count);
        }
        match vec_ty.scalar_bits {
            32 | 16 | 8 => {
                let block_ty =
                    VecType::new(vec_ty.scalar, vec_ty.scalar_bits, 128 / vec_ty.scalar_bits);
                let scalar_ty = block_ty.scalar.rust(block_ty.scalar_bits);
                let native_ty = self.arch_ty(&block_ty);
                let vec_32 = block_ty.reinterpret(block_ty.scalar, 32);
                let unpacklo_32 = simple_sign_unaware_intrinsic("unpacklo", &vec_32);
                let unpackhi_32 = simple_sign_unaware_intrinsic("unpackhi", &vec_32);
                let vec_64 = block_ty.reinterpret(block_ty.scalar, 64);
                let unpacklo_64 = simple_sign_unaware_intrinsic("unpacklo", &vec_64);
                let unpackhi_64 = simple_sign_unaware_intrinsic("unpackhi", &vec_64);

                let vec_combined =
                    VecType::new(block_ty.scalar, block_ty.scalar_bits, block_ty.len * 2);
                let combine_half = Ident::new(
                    &format!("combine_{}", block_ty.rust_name()),
                    Span::call_site(),
                );
                let combine_full = Ident::new(
                    &format!("combine_{}", vec_combined.rust_name()),
                    Span::call_site(),
                );
                let block_len = block_size as usize / vec_ty.scalar_bits;

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

                let final_unpack = if vec_ty.scalar == ScalarType::Float && vec_ty.scalar_bits == 32
                {
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
                } else {
                    quote! {
                        let out0 = #unpacklo_64(tmp0, tmp2); // [0,4,8,12]
                        let out1 = #unpackhi_64(tmp0, tmp2); // [1,5,9,13]
                        let out2 = #unpacklo_64(tmp1, tmp3); // [2,6,10,14]
                        let out3 = #unpackhi_64(tmp1, tmp3); // [3,7,11,15]
                    }
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

                        let tmp0 = #unpacklo_32(v0, v1); // [0,4,1,5]
                        let tmp1 = #unpackhi_32(v0, v1); // [2,6,3,7]
                        let tmp2 = #unpacklo_32(v2, v3); // [8,12,9,13]
                        let tmp3 = #unpackhi_32(v2, v3); // [10,14,11,15]

                        #final_unpack

                        #token.#combine_full(
                            #token.#combine_half(out0.simd_into(#token), out1.simd_into(#token)),
                            #token.#combine_half(out2.simd_into(#token), out3.simd_into(#token)),
                        )
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
            512,
            "AVX-512 interleaved loads only specialize 512-bit vectors"
        );
        let scalar_ty = vec_ty.scalar.rust(vec_ty.scalar_bits);
        let native_ty = self.arch_ty(vec_ty);
        let len = vec_ty.len;
        let permute = avx512_permutexvar_intrinsic(vec_ty);
        let indices = avx512_index_vector(
            vec_ty,
            interleaved_load_indices(vec_ty.len, block_count as usize),
        );

        self.kernel_method(op, vec_ty, |token| {
            quote! {
                let lanes: #native_ty =
                    crate::transmute::checked_transmute_copy::<[#scalar_ty; #len], #native_ty>(
                        src,
                    );
                #permute(#indices, lanes).simd_into(#token)
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
        if *self == Self::Avx512 && vec_ty.n_bits() == 512 {
            return self.handle_avx512_store_interleaved(op, vec_ty, block_size, block_count);
        }
        match vec_ty.scalar_bits {
            32 | 16 | 8 => {
                let block_ty =
                    VecType::new(vec_ty.scalar, vec_ty.scalar_bits, 128 / vec_ty.scalar_bits);
                let scalar_ty = block_ty.scalar.rust(block_ty.scalar_bits);
                let native_ty = self.arch_ty(&block_ty);
                let vec_32 = block_ty.reinterpret(block_ty.scalar, 32);
                let unpacklo_32 = simple_sign_unaware_intrinsic("unpacklo", &vec_32);
                let unpackhi_32 = simple_sign_unaware_intrinsic("unpackhi", &vec_32);
                let vec_64 = block_ty.reinterpret(block_ty.scalar, 64);
                let unpacklo_64 = simple_sign_unaware_intrinsic("unpacklo", &vec_64);
                let unpackhi_64 = simple_sign_unaware_intrinsic("unpackhi", &vec_64);

                let vec_combined =
                    VecType::new(block_ty.scalar, block_ty.scalar_bits, block_ty.len * 2);
                let split_half = Ident::new(
                    &format!("split_{}", vec_combined.rust_name()),
                    Span::call_site(),
                );
                let split_full =
                    Ident::new(&format!("split_{}", vec_ty.rust_name()), Span::call_site());
                let block_len = block_size as usize / vec_ty.scalar_bits;

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

                let final_unpack = if vec_ty.scalar == ScalarType::Float && vec_ty.scalar_bits == 32
                {
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
                } else {
                    quote! {
                        let out0 = #unpacklo_64(tmp0, tmp2); // [0,4,8,12]
                        let out1 = #unpackhi_64(tmp0, tmp2); // [1,5,9,13]
                        let out2 = #unpacklo_64(tmp1, tmp3); // [2,6,10,14]
                        let out3 = #unpackhi_64(tmp1, tmp3); // [3,7,11,15]
                    }
                };

                self.kernel_method(op, vec_ty, |token| {
                    quote! {
                        let (v01, v23) = #token.#split_full(a);
                        let (v0, v1) = #token.#split_half(v01);
                        let (v2, v3) = #token.#split_half(v23);
                        let v0 = v0.into();
                        let v1 = v1.into();
                        let v2 = v2.into();
                        let v3 = v3.into();

                        let tmp0 = #unpacklo_32(v0, v1); // [0,4,1,5]
                        let tmp1 = #unpackhi_32(v0, v1); // [2,6,3,7]
                        let tmp2 = #unpacklo_32(v2, v3); // [8,12,9,13]
                        let tmp3 = #unpackhi_32(v2, v3); // [10,14,11,15]

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
            512,
            "AVX-512 interleaved stores only specialize 512-bit vectors"
        );
        let scalar_ty = vec_ty.scalar.rust(vec_ty.scalar_bits);
        let native_ty = self.arch_ty(vec_ty);
        let len = vec_ty.len;
        let permute = avx512_permutexvar_intrinsic(vec_ty);
        let indices = avx512_index_vector(
            vec_ty,
            interleaved_store_indices(vec_ty.len, block_count as usize),
        );

        self.kernel_method(op, vec_ty, |_| {
            quote! {
                let lanes = #permute(#indices, a.into());
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
            Self::Sse4_2 => &[128],
            Self::Avx2 => &[128, 256],
            Self::Avx512 => &[128, 256, 512],
        };

        for vec_ty in vec_widths
            .iter()
            .map(|n| VecType::new(ScalarType::Int, 8, *n / 8))
        {
            let arch_ty = self.arch_ty(&vec_ty);

            let helper_name = format_ident!("dyn_alignr_{}", vec_ty.n_bits());
            let alignr_intrinsic = simple_sign_unaware_intrinsic("alignr", &vec_ty);
            let shifts = (0_usize..16).map(|shift| {
                let shift_i32 = i32::try_from(shift).unwrap();
                quote! { #shift => #alignr_intrinsic::<#shift_i32>(a, b) }
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

    fn sse42_slide_helpers() -> TokenStream {
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
                    fn #helper_name(token: Sse4_2, a: [__m128i; #num_blocks], b: [__m128i; #num_blocks], shift_bytes: usize) -> [__m128i; #num_blocks] {
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
