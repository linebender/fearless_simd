// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::x86::{
    self, cast_ident, coarse_type, extend_intrinsic, float_compare_method, intrinsic_ident,
    op_suffix, pack_intrinsic, set1_intrinsic, simple_intrinsic, simple_sign_unaware_intrinsic,
    unpack_intrinsic,
};
use crate::generic::{
    generic_as_array, generic_block_combine, generic_block_split, generic_from_array,
    generic_from_bytes, generic_op_name, generic_store_array, generic_to_bytes, scalar_binary,
};
use crate::level::Level;
use crate::ops::{Op, OpSig, Quantifier, SlideGranularity, valid_reinterpret};
use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens as _, format_ident, quote};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum X86 {
    Sse4_2,
    Avx2,
    Avx512,
}

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
            // Ice Lake feature set (avx512f implies avx, avx2, f16c, fxsr, sse, sse2, sse3, sse4.1, sse4.2, ssse3)
            Self::Avx512 => {
                "adx,aes,avx512bitalg,avx512bw,avx512cd,avx512dq,avx512f,avx512ifma,avx512vbmi,avx512vbmi2,avx512vl,avx512vnni,avx512vpopcntdq,bmi1,bmi2,cmpxchg16b,fma,gfni,lzcnt,movbe,pclmulqdq,popcnt,rdrand,rdseed,sha,vaes,vpclmulqdq,xsave,xsavec,xsaveopt,xsaves"
            }
        })
    }

    fn arch_ty(&self, vec_ty: &VecType) -> TokenStream {
        let suffix = match (vec_ty.scalar, vec_ty.scalar_bits) {
            (ScalarType::Float, 32) => "",
            (ScalarType::Float, 64) => "d",
            (ScalarType::Float, _) => unimplemented!(),
            (ScalarType::Unsigned | ScalarType::Int | ScalarType::Mask, _) => "i",
        };
        let name = format!("__m{}{}", vec_ty.scalar_bits * vec_ty.len, suffix);
        Ident::new(&name, Span::call_site()).into_token_stream()
    }

    fn token_doc(&self) -> &'static str {
        match self {
            Self::Sse4_2 => r#"The SIMD token for the "SSE4.2" level."#,
            Self::Avx2 => r#"The SIMD token for the "AVX2" and "FMA" level."#,
            Self::Avx512 => r#"The SIMD token for the "AVX-512" level (Ice Lake feature set)."#,
        }
    }

    fn token_inner(&self) -> TokenStream {
        match self {
            Self::Sse4_2 => quote!(crate::core_arch::x86::Sse4_2),
            Self::Avx2 => quote!(crate::core_arch::x86::Avx2),
            Self::Avx512 => quote!(crate::core_arch::x86::Avx512),
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

    fn make_module_footer(&self) -> TokenStream {
        let alignr_helpers = self.dyn_alignr_helpers();
        let slide_helpers = match self {
            Self::Sse4_2 => Self::sse42_slide_helpers(),
            Self::Avx2 => Self::avx2_slide_helpers(),
            Self::Avx512 => {
                let avx2_helpers = Self::avx2_slide_helpers();
                let avx512_helpers = Self::avx512_slide_helpers();
                quote! {
                    #avx2_helpers
                    #avx512_helpers
                }
            }
        };

        quote! {
            #alignr_helpers
            #slide_helpers
        }
    }

    fn make_level_body(&self) -> TokenStream {
        let level_tok = self.token();
        match self {
            Self::Sse4_2 => quote! {
                #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
                return Level::#level_tok(self);
                #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
                {
                    Level::baseline()
                }
            },
            Self::Avx2 => quote! {
                #[cfg(not(target_feature = "avx512f"))]
                return Level::#level_tok(self);
                #[cfg(target_feature = "avx512f")]
                {
                    Level::baseline()
                }
            },
            Self::Avx512 => quote! {
                Level::#level_tok(self)
            },
        }
    }

    fn make_impl_body(&self) -> TokenStream {
        match self {
            Self::Sse4_2 => quote! {
                /// Create a SIMD token.
                ///
                /// # Safety
                ///
                /// The SSE4.2 CPU feature must be available.
                #[inline]
                pub const unsafe fn new_unchecked() -> Self {
                    Sse4_2 {
                        sse4_2: unsafe { crate::core_arch::x86::Sse4_2::new_unchecked() },
                    }
                }
            },
            Self::Avx2 => quote! {
                /// Create a SIMD token.
                ///
                /// # Safety
                ///
                /// The AVX2 and FMA CPU features must be available.
                #[inline]
                pub const unsafe fn new_unchecked() -> Self {
                    Self {
                        avx2: unsafe { crate::core_arch::x86::Avx2::new_unchecked() },
                    }
                }
            },
            Self::Avx512 => quote! {
                /// Create a SIMD token.
                ///
                /// # Safety
                ///
                /// The AVX-512 (Ice Lake feature set) CPU features must be available.
                #[inline]
                pub const unsafe fn new_unchecked() -> Self {
                    Self {
                        avx512: unsafe { crate::core_arch::x86::Avx512::new_unchecked() },
                    }
                }
            },
        }
    }

    fn make_method(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let Op { sig, method, .. } = op;
        let method_sig = op.simd_trait_method_sig(vec_ty);

        match sig {
            OpSig::Splat => self.handle_splat(method_sig, vec_ty),
            OpSig::Compare => self.handle_compare(method_sig, method, vec_ty),
            OpSig::Unary => self.handle_unary(method_sig, method, vec_ty),
            OpSig::WidenNarrow { target_ty } => {
                self.handle_widen_narrow(method_sig, method, vec_ty, target_ty)
            }
            OpSig::Binary => self.handle_binary(method_sig, method, vec_ty),
            OpSig::Shift => self.handle_shift(method_sig, method, vec_ty),
            OpSig::Ternary => self.handle_ternary(method_sig, method, vec_ty),
            OpSig::Select => self.handle_select(method_sig, vec_ty),
            OpSig::Combine { combined_ty } => self.handle_combine(method_sig, vec_ty, &combined_ty),
            OpSig::Split { half_ty } => self.handle_split(method_sig, vec_ty, &half_ty),
            OpSig::Zip { select_low } => self.handle_zip(method_sig, vec_ty, select_low),
            OpSig::Unzip { select_even } => self.handle_unzip(method_sig, vec_ty, select_even),
            OpSig::Slide { granularity } => self.handle_slide(method_sig, vec_ty, granularity),
            OpSig::Cvt {
                target_ty,
                scalar_bits,
                precise,
            } => self.handle_cvt(method_sig, vec_ty, target_ty, scalar_bits, precise),
            OpSig::Reinterpret {
                target_ty,
                scalar_bits,
            } => self.handle_reinterpret(self, method_sig, vec_ty, target_ty, scalar_bits),
            OpSig::MaskReduce {
                quantifier,
                condition,
            } => self.handle_mask_reduce(method_sig, vec_ty, quantifier, condition),
            OpSig::LoadInterleaved {
                block_size,
                block_count,
            } => self.handle_load_interleaved(method_sig, vec_ty, block_size, block_count),
            OpSig::StoreInterleaved {
                block_size,
                block_count,
            } => self.handle_store_interleaved(method_sig, vec_ty, block_size, block_count),
            OpSig::FromArray { kind } => generic_from_array(method_sig, vec_ty, kind),
            OpSig::AsArray { kind } => {
                generic_as_array(method_sig, vec_ty, kind, self.max_block_size(), |vec_ty| {
                    self.arch_ty(vec_ty)
                })
            }
            OpSig::StoreArray => generic_store_array(method_sig, vec_ty),
            OpSig::FromBytes => generic_from_bytes(method_sig, vec_ty),
            OpSig::ToBytes => generic_to_bytes(method_sig, vec_ty),
        }
    }
}

impl X86 {
    pub(crate) fn handle_splat(&self, method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
        let intrinsic = set1_intrinsic(vec_ty);
        let cast = match vec_ty.scalar {
            ScalarType::Unsigned => quote!(.cast_signed()),
            _ => quote!(),
        };
        quote! {
            #method_sig {
                unsafe {
                    #intrinsic(val #cast).simd_into(self)
                }
            }
        }
    }

    pub(crate) fn handle_compare(
        &self,
        method_sig: TokenStream,
        method: &str,
        vec_ty: &VecType,
    ) -> TokenStream {
        // AVX-512 has native comparison intrinsics that return masks
        // We then convert the mask back to a vector using movm intrinsics
        if *self == Self::Avx512 && vec_ty.n_bits() == 512 {
            return self.handle_compare_avx512(method_sig, method, vec_ty);
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

        quote! {
            #method_sig {
                unsafe { #expr.simd_into(self) }
            }
        }
    }

    /// Handle AVX-512 comparisons using native mask-returning intrinsics
    fn handle_compare_avx512(
        &self,
        method_sig: TokenStream,
        method: &str,
        vec_ty: &VecType,
    ) -> TokenStream {
        // AVX-512 comparisons return __mmask* types
        // We need to convert back to vector masks using _mm512_movm_epi*

        // Get the movm intrinsic to convert mask register to vector
        let movm = match vec_ty.scalar_bits {
            8 => format_ident!("_mm512_movm_epi8"),
            16 => format_ident!("_mm512_movm_epi16"),
            32 => format_ident!("_mm512_movm_epi32"),
            64 => format_ident!("_mm512_movm_epi64"),
            _ => unreachable!(),
        };

        let expr = if vec_ty.scalar == ScalarType::Float {
            // Float comparisons use _mm512_cmp_ps_mask / _mm512_cmp_pd_mask with predicates
            let cmp_mask = match vec_ty.scalar_bits {
                32 => format_ident!("_mm512_cmp_ps_mask"),
                64 => format_ident!("_mm512_cmp_pd_mask"),
                _ => unreachable!(),
            };
            // Predicate values from Intel docs (same as used in float_compare_method)
            let predicate = match method {
                "simd_eq" => 0x00,
                "simd_lt" => 0x11,
                "simd_le" => 0x12,
                "simd_ge" => 0x1D,
                "simd_gt" => 0x1E,
                _ => unreachable!(),
            };
            // The mask type is stored as __m512i internally, so movm gives us what we need
            quote! {
                let mask = #cmp_mask::<#predicate>(a.into(), b.into());
                #movm(mask)
            }
        } else {
            // Integer comparisons
            let suffix = match vec_ty.scalar_bits {
                8 => "epi8",
                16 => "epi16",
                32 => "epi32",
                64 => "epi64",
                _ => unreachable!(),
            };
            let unsigned_suffix = match vec_ty.scalar_bits {
                8 => "epu8",
                16 => "epu16",
                32 => "epu32",
                64 => "epu64",
                _ => unreachable!(),
            };

            match method {
                "simd_eq" => {
                    let cmp = format_ident!("_mm512_cmpeq_{}_mask", suffix);
                    quote! {
                        let mask = #cmp(a.into(), b.into());
                        #movm(mask)
                    }
                }
                "simd_lt" => {
                    if vec_ty.scalar == ScalarType::Unsigned {
                        let cmp = format_ident!("_mm512_cmplt_{}_mask", unsigned_suffix);
                        quote! {
                            let mask = #cmp(a.into(), b.into());
                            #movm(mask)
                        }
                    } else {
                        let cmp = format_ident!("_mm512_cmplt_{}_mask", suffix);
                        quote! {
                            let mask = #cmp(a.into(), b.into());
                            #movm(mask)
                        }
                    }
                }
                "simd_le" => {
                    if vec_ty.scalar == ScalarType::Unsigned {
                        let cmp = format_ident!("_mm512_cmple_{}_mask", unsigned_suffix);
                        quote! {
                            let mask = #cmp(a.into(), b.into());
                            #movm(mask)
                        }
                    } else {
                        let cmp = format_ident!("_mm512_cmple_{}_mask", suffix);
                        quote! {
                            let mask = #cmp(a.into(), b.into());
                            #movm(mask)
                        }
                    }
                }
                "simd_gt" => {
                    if vec_ty.scalar == ScalarType::Unsigned {
                        let cmp = format_ident!("_mm512_cmpgt_{}_mask", unsigned_suffix);
                        quote! {
                            let mask = #cmp(a.into(), b.into());
                            #movm(mask)
                        }
                    } else {
                        let cmp = format_ident!("_mm512_cmpgt_{}_mask", suffix);
                        quote! {
                            let mask = #cmp(a.into(), b.into());
                            #movm(mask)
                        }
                    }
                }
                "simd_ge" => {
                    if vec_ty.scalar == ScalarType::Unsigned {
                        let cmp = format_ident!("_mm512_cmpge_{}_mask", unsigned_suffix);
                        quote! {
                            let mask = #cmp(a.into(), b.into());
                            #movm(mask)
                        }
                    } else {
                        let cmp = format_ident!("_mm512_cmpge_{}_mask", suffix);
                        quote! {
                            let mask = #cmp(a.into(), b.into());
                            #movm(mask)
                        }
                    }
                }
                _ => unreachable!(),
            }
        };

        quote! {
            #method_sig {
                unsafe { #expr.simd_into(self) }
            }
        }
    }

    pub(crate) fn handle_unary(
        &self,
        method_sig: TokenStream,
        method: &str,
        vec_ty: &VecType,
    ) -> TokenStream {
        match method {
            "fract" => {
                let trunc_op = generic_op_name("trunc", vec_ty);
                quote! {
                    #method_sig {
                        a - self.#trunc_op(a)
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
                quote! {
                    #method_sig {
                        unsafe { #expr.simd_into(self) }
                    }
                }
            }
        }
    }

    pub(crate) fn handle_widen_narrow(
        &self,
        method_sig: TokenStream,
        method: &str,
        vec_ty: &VecType,
        target_ty: VecType,
    ) -> TokenStream {
        let dst_width = target_ty.n_bits();
        let expr = match method {
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
                            unsafe {
                                #extend(a.into()).simd_into(self)
                            }
                        }
                    }
                    (Self::Avx512, 512, 256) => {
                        // AVX-512 has native _mm512_cvt* intrinsics that extend 256-bit to 512-bit directly
                        let extend = extend_intrinsic(
                            vec_ty.scalar,
                            vec_ty.scalar_bits,
                            target_ty.scalar_bits,
                            dst_width, // Use 512-bit intrinsic directly
                        );
                        quote! {
                            unsafe {
                                #extend(a.into()).simd_into(self)
                            }
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
                            unsafe {
                                let (a0, a1) = self.#split(a);
                                let high = #extend(a0.into()).simd_into(self);
                                let low = #extend(a1.into()).simd_into(self);
                                self.#combine(high, low)
                            }
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
                            unsafe {
                                let raw = a.into();
                                let high = #extend(raw).simd_into(self);
                                // Shift by 8 since we want to get the higher part into the
                                // lower position.
                                let low = #extend(_mm_srli_si128::<8>(raw)).simd_into(self);
                                self.#combine(high, low)
                            }
                        }
                    }
                    _ => unimplemented!(),
                }
            }
            "narrow" => {
                match (self, dst_width, vec_ty.n_bits()) {
                    (Self::Avx2 | Self::Avx512, 128, 256) => {
                        let mask = match target_ty.scalar_bits {
                            8 => {
                                quote! { 0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1 }
                            }
                            _ => unimplemented!(),
                        };
                        quote! {
                            unsafe {
                                let mask = _mm256_setr_epi8(#mask, #mask);

                                let shuffled = _mm256_shuffle_epi8(a.into(), mask);
                                let packed = _mm256_permute4x64_epi64::<0b11_01_10_00>(shuffled);

                                _mm256_castsi256_si128(packed).simd_into(self)
                            }
                        }
                    }
                    (Self::Avx512, 256, 512) => {
                        // AVX-512 has native truncation intrinsics: _mm512_cvtepi16_epi8 etc.
                        // These directly narrow 512-bit to 256-bit with truncation
                        let narrow = match (vec_ty.scalar_bits, target_ty.scalar_bits) {
                            (16, 8) => format_ident!("_mm512_cvtepi16_epi8"),
                            (32, 16) => format_ident!("_mm512_cvtepi32_epi16"),
                            (64, 32) => format_ident!("_mm512_cvtepi64_epi32"),
                            _ => unimplemented!(
                                "narrow from {} to {} bits",
                                vec_ty.scalar_bits,
                                target_ty.scalar_bits
                            ),
                        };
                        quote! {
                            unsafe {
                                #narrow(a.into()).simd_into(self)
                            }
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
                            let (a, b) = self.#split(a);
                            unsafe {
                                // Note that AVX2 only has an intrinsic for saturating cast,
                                // but not wrapping.
                                let mask = #mask(0xFF);
                                let lo_masked = _mm256_and_si256(a.into(), mask);
                                let hi_masked = _mm256_and_si256(b.into(), mask);
                                // The 256-bit version of packus_epi16 operates lane-wise, so we need to arrange things
                                // properly afterwards.
                                let result = _mm256_permute4x64_epi64::<0b_11_01_10_00>(#pack(lo_masked, hi_masked));
                                result.simd_into(self)
                            }
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
                            let (a, b) = self.#split(a);
                            unsafe {
                                // Below AVX-512. we only have an intrinsic for saturating cast, but not wrapping.
                                let mask = #mask(0xFF);
                                let lo_masked = _mm_and_si128(a.into(), mask);
                                let hi_masked = _mm_and_si128(b.into(), mask);
                                let result = #pack(lo_masked, hi_masked);
                                result.simd_into(self)
                            }
                        }
                    }
                    _ => unimplemented!(),
                }
            }
            _ => unreachable!(),
        };

        quote! {
            #method_sig {
                #expr
            }
        }
    }

    pub(crate) fn handle_binary(
        &self,
        method_sig: TokenStream,
        method: &str,
        vec_ty: &VecType,
    ) -> TokenStream {
        let body = match method {
            "mul" if vec_ty.scalar_bits == 8 => {
                // https://stackoverflow.com/questions/8193601/sse-multiplication-16-x-uint8-t
                let mullo = intrinsic_ident("mullo", "epi16", vec_ty.n_bits());
                let set1 = intrinsic_ident("set1", "epi16", vec_ty.n_bits());
                let and = intrinsic_ident("and", coarse_type(vec_ty), vec_ty.n_bits());
                let or = intrinsic_ident("or", coarse_type(vec_ty), vec_ty.n_bits());
                let slli = intrinsic_ident("slli", "epi16", vec_ty.n_bits());
                let srli = intrinsic_ident("srli", "epi16", vec_ty.n_bits());
                quote! {
                    unsafe {
                        let dst_even = #mullo(a.into(), b.into());
                        let dst_odd = #mullo(#srli::<8>(a.into()), #srli::<8>(b.into()));

                        #or(#slli(dst_odd, 8), #and(dst_even, #set1(0xFF))).simd_into(self)
                    }
                }
            }
            "shlv" | "shrv"
                if (*self == Self::Avx2 || *self == Self::Avx512) && vec_ty.scalar_bits >= 32 =>
            {
                let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
                let name = match (method, vec_ty.scalar) {
                    ("shrv", ScalarType::Int) => "srav",
                    ("shrv", _) => "srlv",
                    ("shlv", _) => "sllv",
                    _ => unreachable!(),
                };
                let intrinsic = intrinsic_ident(name, suffix, vec_ty.n_bits());
                quote! {
                    unsafe { #intrinsic(a.into(), b.into()).simd_into(self) }
                }
            }
            // SSE2 has shift operations, but they shift every lane by the same amount, so we can't use them here.
            "shlv" => scalar_binary(quote!(core::ops::Shl::shl)),
            "shrv" => scalar_binary(quote!(core::ops::Shr::shr)),
            _ => {
                let args = [quote! { a.into() }, quote! { b.into() }];
                let expr = x86::expr(method, vec_ty, &args);
                quote! {
                    unsafe { #expr.simd_into(self) }
                }
            }
        };

        quote! {
            #method_sig {
                #body
            }
        }
    }

    pub(crate) fn handle_shift(
        &self,
        method_sig: TokenStream,
        method: &str,
        vec_ty: &VecType,
    ) -> TokenStream {
        let op = match (method, vec_ty.scalar) {
            ("shr", ScalarType::Unsigned) => "srl",
            ("shr", ScalarType::Int) => "sra",
            ("shl", _) => "sll",
            _ => unreachable!(),
        };
        let ty_bits = vec_ty.n_bits();
        let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits.max(16), false);
        let shift_intrinsic = intrinsic_ident(op, suffix, ty_bits);

        if vec_ty.scalar_bits == 8 {
            // x86 doesn't have shifting for 8-bit, so we first convert into 16-bit, shift, and then back to 8-bit.

            // AVX-512 uses different intrinsics - cmpgt returns a mask, not a vector
            if *self == Self::Avx512 && ty_bits == 512 {
                // For AVX-512, use cvtepi8_epi16 for sign extension (simpler than unpack + cmpgt)
                // We split 512-bit into two 256-bit halves, extend each to 512-bit, shift, then narrow back
                // Use _mm512_cvtepi16_epi8 to truncate (not packs which interleaves within lanes)

                let extend = match vec_ty.scalar {
                    ScalarType::Unsigned => format_ident!("_mm512_cvtepu8_epi16"),
                    ScalarType::Int => format_ident!("_mm512_cvtepi8_epi16"),
                    _ => unimplemented!(),
                };

                return quote! {
                    #method_sig {
                        unsafe {
                            let val = a.into();
                            let shift_count = _mm_cvtsi32_si128(shift.cast_signed());

                            // Split into low and high 256-bit halves
                            let lo_256 = _mm512_castsi512_si256(val);
                            let hi_256 = _mm512_extracti64x4_epi64::<1>(val);

                            // Extend each half from 8-bit to 16-bit (256-bit -> 512-bit)
                            let lo_16 = #extend(lo_256);
                            let hi_16 = #extend(hi_256);

                            // Shift
                            let lo_shifted = #shift_intrinsic(lo_16, shift_count);
                            let hi_shifted = #shift_intrinsic(hi_16, shift_count);

                            // Truncate back to 8-bit using cvtepi16_epi8 (not packs which interleaves)
                            // cvtepi16_epi8 takes 512-bit and returns 256-bit, so we combine the results
                            let lo_narrow = _mm512_cvtepi16_epi8(lo_shifted);
                            let hi_narrow = _mm512_cvtepi16_epi8(hi_shifted);

                            // Combine the two 256-bit results into a 512-bit result
                            let result = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo_narrow), hi_narrow);
                            result.simd_into(self)
                        }
                    }
                };
            }

            let unpack_hi = unpack_intrinsic(ScalarType::Int, 8, false, ty_bits);
            let unpack_lo = unpack_intrinsic(ScalarType::Int, 8, true, ty_bits);

            let set0 = intrinsic_ident("setzero", coarse_type(vec_ty), ty_bits);
            let extend_expr = |expr| match vec_ty.scalar {
                ScalarType::Unsigned => quote! {
                    #expr(val, #set0())
                },
                ScalarType::Int => {
                    let cmp_intrinsic = intrinsic_ident("cmpgt", "epi8", ty_bits);
                    quote! {
                        #expr(val, #cmp_intrinsic(#set0(), val))
                    }
                }
                _ => unimplemented!(),
            };

            let extend_intrinsic_lo = extend_expr(unpack_lo);
            let extend_intrinsic_hi = extend_expr(unpack_hi);
            let pack_intrinsic = pack_intrinsic(16, vec_ty.scalar == ScalarType::Int, ty_bits);

            quote! {
                #method_sig {
                    unsafe {
                        let val = a.into();
                        let shift_count = _mm_cvtsi32_si128(shift.cast_signed());

                        let lo_16 = #extend_intrinsic_lo;
                        let hi_16 = #extend_intrinsic_hi;

                        let lo_shifted = #shift_intrinsic(lo_16, shift_count);
                        let hi_shifted = #shift_intrinsic(hi_16, shift_count);

                        #pack_intrinsic(lo_shifted, hi_shifted).simd_into(self)
                    }
                }
            }
        } else {
            quote! {
                #method_sig {
                    unsafe { #shift_intrinsic(a.into(), _mm_cvtsi32_si128(shift.cast_signed())).simd_into(self) }
                }
            }
        }
    }

    pub(crate) fn handle_ternary(
        &self,
        method_sig: TokenStream,
        method: &str,
        vec_ty: &VecType,
    ) -> TokenStream {
        match method {
            "mul_add" if *self == Self::Avx2 || *self == Self::Avx512 => {
                let intrinsic = simple_intrinsic("fmadd", vec_ty);
                quote! {
                    #method_sig {
                        unsafe { #intrinsic(a.into(), b.into(), c.into()).simd_into(self) }
                    }
                }
            }
            "mul_sub" if *self == Self::Avx2 || *self == Self::Avx512 => {
                let intrinsic = simple_intrinsic("fmsub", vec_ty);
                quote! {
                    #method_sig {
                        unsafe { #intrinsic(a.into(), b.into(), c.into()).simd_into(self) }
                    }
                }
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
                quote! {
                    #method_sig {
                    #expr.simd_into(self)
                    }
                }
            }
        }
    }

    pub(crate) fn handle_select(&self, method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
        // AVX-512 uses mask registers instead of blendv
        if *self == Self::Avx512 && vec_ty.n_bits() == 512 {
            // Convert vector mask to __mmask* using movepi*_mask, then use mask_blend
            // select(mask, a, b) = where mask is true, take a; else take b
            // _mm512_mask_blend_*(k, a, b) = where k bit is 0, take a; where 1, take b
            // So we need: mask_blend(mask, c, b) where c is "false" value, b is "true" value
            let (move_mask, blend) = match (vec_ty.scalar, vec_ty.scalar_bits) {
                (ScalarType::Float, 32) => (
                    format_ident!("_mm512_movepi32_mask"),
                    format_ident!("_mm512_mask_blend_ps"),
                ),
                (ScalarType::Float, 64) => (
                    format_ident!("_mm512_movepi64_mask"),
                    format_ident!("_mm512_mask_blend_pd"),
                ),
                (_, 8) => (
                    format_ident!("_mm512_movepi8_mask"),
                    format_ident!("_mm512_mask_blend_epi8"),
                ),
                (_, 16) => (
                    format_ident!("_mm512_movepi16_mask"),
                    format_ident!("_mm512_mask_blend_epi16"),
                ),
                (_, 32) => (
                    format_ident!("_mm512_movepi32_mask"),
                    format_ident!("_mm512_mask_blend_epi32"),
                ),
                (_, 64) => (
                    format_ident!("_mm512_movepi64_mask"),
                    format_ident!("_mm512_mask_blend_epi64"),
                ),
                _ => unreachable!(),
            };

            // The mask 'a' is a mask type (mask32x16, etc.) which stores as __m512i
            // We just need to convert it to __m512i and call movepi*_mask
            let mask_expr = quote! { #move_mask(a.into()) };

            return quote! {
                #method_sig {
                    unsafe {
                        // a is the mask, b is the "true" value, c is the "false" value
                        // mask_blend: where mask bit is 0, take first arg; where 1, take second
                        let k = #mask_expr;
                        #blend(k, c.into(), b.into()).simd_into(self)
                    }
                }
            };
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

        quote! {
            #method_sig {
                unsafe { #expr.simd_into(self) }
            }
        }
    }

    pub(crate) fn handle_split(
        &self,
        method_sig: TokenStream,
        vec_ty: &VecType,
        half_ty: &VecType,
    ) -> TokenStream {
        if (*self == Self::Avx2 || *self == Self::Avx512) && half_ty.n_bits() == 128 {
            let extract_op = match vec_ty.scalar {
                ScalarType::Float => "extractf128",
                _ => "extracti128",
            };
            let extract_intrinsic = intrinsic_ident(extract_op, coarse_type(vec_ty), 256);
            quote! {
                #method_sig {
                    unsafe {
                        (
                            #extract_intrinsic::<0>(a.into()).simd_into(self),
                            #extract_intrinsic::<1>(a.into()).simd_into(self),
                        )
                    }
                }
            }
        } else if *self == Self::Avx512 && half_ty.n_bits() == 256 {
            // Split a 512-bit vector into two 256-bit halves using AVX-512 intrinsics
            let (cast_intrinsic, extract_intrinsic) = match vec_ty.scalar {
                ScalarType::Float if vec_ty.scalar_bits == 32 => (
                    format_ident!("_mm512_castps512_ps256"),
                    format_ident!("_mm512_extractf32x8_ps"),
                ),
                ScalarType::Float => (
                    format_ident!("_mm512_castpd512_pd256"),
                    format_ident!("_mm512_extractf64x4_pd"),
                ),
                _ => (
                    format_ident!("_mm512_castsi512_si256"),
                    format_ident!("_mm512_extracti64x4_epi64"),
                ),
            };
            quote! {
                #method_sig {
                    unsafe {
                        (
                            #cast_intrinsic(a.into()).simd_into(self),
                            #extract_intrinsic::<1>(a.into()).simd_into(self),
                        )
                    }
                }
            }
        } else {
            generic_block_split(method_sig, half_ty, self.max_block_size())
        }
    }

    pub(crate) fn handle_combine(
        &self,
        method_sig: TokenStream,
        vec_ty: &VecType,
        combined_ty: &VecType,
    ) -> TokenStream {
        if (*self == Self::Avx2 || *self == Self::Avx512) && combined_ty.n_bits() == 256 {
            let suffix = match (vec_ty.scalar, vec_ty.scalar_bits) {
                (ScalarType::Float, 32) => "m128",
                (ScalarType::Float, 64) => "m128d",
                _ => "m128i",
            };
            let set_intrinsic = intrinsic_ident("setr", suffix, 256);
            quote! {
                #method_sig {
                    unsafe {
                        #set_intrinsic(a.into(), b.into()).simd_into(self)
                    }
                }
            }
        } else if *self == Self::Avx512 && combined_ty.n_bits() == 512 {
            // Combine two 256-bit vectors into one 512-bit vector using AVX-512 intrinsics
            let (cast_intrinsic, insert_intrinsic) = match (vec_ty.scalar, vec_ty.scalar_bits) {
                (ScalarType::Float, 32) => (
                    format_ident!("_mm512_castps256_ps512"),
                    format_ident!("_mm512_insertf32x8"),
                ),
                (ScalarType::Float, 64) => (
                    format_ident!("_mm512_castpd256_pd512"),
                    format_ident!("_mm512_insertf64x4"),
                ),
                _ => (
                    format_ident!("_mm512_castsi256_si512"),
                    format_ident!("_mm512_inserti64x4"),
                ),
            };
            quote! {
                #method_sig {
                    unsafe {
                        let lo = #cast_intrinsic(a.into());
                        #insert_intrinsic::<1>(lo, b.into()).simd_into(self)
                    }
                }
            }
        } else {
            generic_block_combine(method_sig, combined_ty, self.max_block_size())
        }
    }

    pub(crate) fn handle_zip(
        &self,
        method_sig: TokenStream,
        vec_ty: &VecType,
        select_low: bool,
    ) -> TokenStream {
        let expr = match vec_ty.n_bits() {
            128 => {
                let op = if select_low { "unpacklo" } else { "unpackhi" };

                let suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
                let unpack_intrinsic = intrinsic_ident(op, suffix, vec_ty.n_bits());
                quote! {
                    unsafe {  #unpack_intrinsic(a.into(), b.into()).simd_into(self) }
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
                    unsafe {
                        let lo = #lo(a.into(), b.into());
                        let hi = #hi(a.into(), b.into());

                        #shuffle::<#shuffle_immediate>(lo, hi).simd_into(self)
                    }
                }
            }
            512 => {
                // AVX-512 zip using permutex2var instructions for efficient single-instruction interleaving.
                // The control mask selects elements from `a` (indices 0..n-1) and `b` (indices n..2n-1).
                let half_len = vec_ty.len / 2;

                // Generate the index pattern for zip_low or zip_high
                // For zip_low with 16 elements: interleave elements 0-7 from each vector
                // For zip_high with 16 elements: interleave elements 8-15 from each vector
                let base_offset = if select_low { 0usize } else { half_len };
                let b_offset = vec_ty.len; // b's elements are at indices n..2n-1

                // Build index array: [base, base+b_offset, base+1, base+1+b_offset, ...]
                let indices: Vec<_> = (0..half_len)
                    .flat_map(|i| {
                        let a_idx = base_offset + i;
                        let b_idx = base_offset + i + b_offset;
                        [a_idx, b_idx]
                    })
                    .collect();

                // Choose the appropriate permutex2var intrinsic based on element type
                // Note: for floats, use the epi32/epi64 set intrinsic since idx is always __m512i
                let (permute_intrinsic, set_intrinsic, index_bits) =
                    match (vec_ty.scalar, vec_ty.scalar_bits) {
                        (ScalarType::Float, 32) => (
                            format_ident!("_mm512_permutex2var_ps"),
                            format_ident!("_mm512_set_epi32"),
                            32usize,
                        ),
                        (ScalarType::Float, 64) => (
                            format_ident!("_mm512_permutex2var_pd"),
                            format_ident!("_mm512_set_epi64"),
                            64usize,
                        ),
                        (_, 8) => (
                            format_ident!("_mm512_permutex2var_epi8"),
                            format_ident!("_mm512_set_epi8"),
                            8usize,
                        ),
                        (_, 16) => (
                            format_ident!("_mm512_permutex2var_epi16"),
                            format_ident!("_mm512_set_epi16"),
                            16usize,
                        ),
                        (_, 32) => (
                            format_ident!("_mm512_permutex2var_epi32"),
                            format_ident!("_mm512_set_epi32"),
                            32usize,
                        ),
                        (_, 64) => (
                            format_ident!("_mm512_permutex2var_epi64"),
                            format_ident!("_mm512_set_epi64"),
                            64usize,
                        ),
                        _ => unreachable!(),
                    };

                // _mm512_set_* takes arguments from most-significant to least-significant,
                // so we need to reverse the indices
                let reversed_indices: Vec<_> = indices.iter().rev().copied().collect();
                let index_literals: Vec<_> = reversed_indices
                    .iter()
                    .map(|&i| {
                        // Use the appropriate integer type for each set intrinsic
                        match index_bits {
                            8 => {
                                let i = i as i8;
                                quote! { #i }
                            }
                            16 => {
                                let i = i as i16;
                                quote! { #i }
                            }
                            32 => {
                                let i = i as i32;
                                quote! { #i }
                            }
                            64 => {
                                let i = i as i64;
                                quote! { #i }
                            }
                            _ => unreachable!(),
                        }
                    })
                    .collect();

                quote! {
                    unsafe {
                        let idx = #set_intrinsic(#(#index_literals),*);
                        #permute_intrinsic(a.into(), idx, b.into()).simd_into(self)
                    }
                }
            }
            _ => unreachable!(),
        };

        quote! {
            #method_sig {
                #expr
            }
        }
    }

    pub(crate) fn handle_unzip(
        &self,
        method_sig: TokenStream,
        vec_ty: &VecType,
        select_even: bool,
    ) -> TokenStream {
        let expr = match (vec_ty.scalar, vec_ty.n_bits(), vec_ty.scalar_bits) {
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

                quote! { unsafe { #intrinsic::<#mask>(a.into(), b.into()).simd_into(self) } }
            }
            (ScalarType::Int | ScalarType::Mask | ScalarType::Unsigned, 128, 32) => {
                // 128-bit shuffle of 32-bit integers; unlike with floats, there is no single shuffle instruction that
                // combines two vectors
                let op = if select_even { "unpacklo" } else { "unpackhi" };
                let intrinsic = intrinsic_ident(op, "epi64", vec_ty.n_bits());

                quote! {
                        unsafe {
                            let t1 = _mm_shuffle_epi32::<0b11_01_10_00>(a.into());
                            let t2 = _mm_shuffle_epi32::<0b11_01_10_00>(b.into());
                            #intrinsic(t1, t2).simd_into(self)
                    }
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
                    unsafe {
                        let mask = #mask_reg;

                        let t1 = #shuffle_epi8(a.into(), mask);
                        let t2 = #shuffle_epi8(b.into(), mask);
                        #unpack_epi64(t1, t2).simd_into(self)
                    }
                }
            }
            (_, 256, 64 | 32) => {
                // First we perform a lane-crossing shuffle to move the even-indexed elements of each input to the lower
                // half, and the odd-indexed ones to the upper half.
                // e.g. [0, 1, 2, 3, 4, 5, 6, 7] becomes [0, 2, 4, 6, 1, 3, 5, 7]).
                let low_shuffle_kind = match vec_ty.scalar_bits {
                    32 => "permutevar8x32",
                    64 => "permute4x64",
                    _ => unreachable!(),
                };
                let low_shuffle_suffix = op_suffix(vec_ty.scalar, vec_ty.scalar_bits, false);
                let low_shuffle_intrinsic =
                    intrinsic_ident(low_shuffle_kind, low_shuffle_suffix, 256);
                let low_shuffle = |input_name: TokenStream| match vec_ty.scalar_bits {
                    32 => {
                        quote! { #low_shuffle_intrinsic(#input_name, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)) }
                    }
                    64 => quote! { #low_shuffle_intrinsic::<0b11_01_10_00>(#input_name) },
                    _ => unreachable!(),
                };
                let shuf_t1 = low_shuffle(quote! { a.into() });
                let shuf_t2 = low_shuffle(quote! { b.into() });

                // Then we combine the lower or upper halves.
                let high_shuffle = intrinsic_ident(
                    match vec_ty.scalar {
                        ScalarType::Float => "permute2f128",
                        _ => "permute2x128",
                    },
                    coarse_type(vec_ty),
                    256,
                );
                let high_shuffle_immediate = if select_even {
                    quote! { 0b0010_0000 }
                } else {
                    quote! { 0b0011_0001 }
                };

                quote! {
                    unsafe {
                        let t1 = #shuf_t1;
                        let t2 = #shuf_t2;

                        #high_shuffle::<#high_shuffle_immediate>(t1, t2).simd_into(self)
                    }
                }
            }
            (_, 256, 16 | 8) => {
                // Separate out the even-indexed and odd-indexed elements within each 128-bit lane
                let mask = match vec_ty.scalar_bits {
                    8 => {
                        quote! { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 }
                    }
                    16 => {
                        quote! { 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 }
                    }
                    _ => unreachable!(),
                };

                // We then permute the even-indexed and odd-indexed blocks across lanes, and finally do a 2x128 permute to
                // select either the even- or odd-indexed elements
                let high_shuffle_immediate = if select_even {
                    quote! { 0b0010_0000 }
                } else {
                    quote! { 0b0011_0001 }
                };

                quote! {
                    unsafe {
                        let mask = _mm256_setr_epi8(#mask, #mask);
                        let a_shuffled = _mm256_shuffle_epi8(a.into(), mask);
                        let b_shuffled = _mm256_shuffle_epi8(b.into(), mask);

                        let packed = _mm256_permute2x128_si256::<#high_shuffle_immediate>(
                            _mm256_permute4x64_epi64::<0b11_01_10_00>(a_shuffled),
                            _mm256_permute4x64_epi64::<0b11_01_10_00>(b_shuffled)
                        );

                        packed.simd_into(self)
                    }
                }
            }
            (ScalarType::Float, 512, 32) => {
                // 512-bit shuffle of 32-bit floats
                // First permute within each 256-bit half to group evens/odds
                let permute_mask = quote! { _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15) };
                let shuffle_immediate = if select_even {
                    quote! { 0b01_00_01_00 }
                } else {
                    quote! { 0b11_10_11_10 }
                };

                quote! {
                    unsafe {
                        let t1 = _mm512_permutexvar_ps(#permute_mask, a.into());
                        let t2 = _mm512_permutexvar_ps(#permute_mask, b.into());
                        _mm512_shuffle_f32x4::<#shuffle_immediate>(t1, t2).simd_into(self)
                    }
                }
            }
            (ScalarType::Float, 512, 64) => {
                // 512-bit shuffle of 64-bit floats
                let permute_mask = quote! { _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7) };
                let shuffle_immediate = if select_even {
                    quote! { 0b01_00_01_00 }
                } else {
                    quote! { 0b11_10_11_10 }
                };

                quote! {
                    unsafe {
                        let t1 = _mm512_permutexvar_pd(#permute_mask, a.into());
                        let t2 = _mm512_permutexvar_pd(#permute_mask, b.into());
                        _mm512_shuffle_f64x2::<#shuffle_immediate>(t1, t2).simd_into(self)
                    }
                }
            }
            (_, 512, 64) => {
                // 512-bit shuffle of 64-bit integers
                let permute_mask = quote! { _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7) };
                let shuffle_immediate = if select_even {
                    quote! { 0b01_00_01_00 }
                } else {
                    quote! { 0b11_10_11_10 }
                };

                quote! {
                    unsafe {
                        let t1 = _mm512_permutexvar_epi64(#permute_mask, a.into());
                        let t2 = _mm512_permutexvar_epi64(#permute_mask, b.into());
                        _mm512_shuffle_i64x2::<#shuffle_immediate>(t1, t2).simd_into(self)
                    }
                }
            }
            (_, 512, 32) => {
                // 512-bit shuffle of 32-bit integers
                let permute_mask = quote! { _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15) };
                let shuffle_immediate = if select_even {
                    quote! { 0b01_00_01_00 }
                } else {
                    quote! { 0b11_10_11_10 }
                };

                quote! {
                    unsafe {
                        let t1 = _mm512_permutexvar_epi32(#permute_mask, a.into());
                        let t2 = _mm512_permutexvar_epi32(#permute_mask, b.into());
                        _mm512_shuffle_i32x4::<#shuffle_immediate>(t1, t2).simd_into(self)
                    }
                }
            }
            (_, 512, 16 | 8) => {
                // 512-bit shuffle of 8 or 16-bit integers
                // Separate out the even-indexed and odd-indexed elements within each 128-bit lane
                let mask = match vec_ty.scalar_bits {
                    8 => {
                        // For 8-bit elements: move even indices to low half, odd to high half
                        quote! {
                            _mm512_set4_epi64(
                                0x0F0D0B0907050301u64.cast_signed(), 0x0E0C0A0806040200u64.cast_signed(),
                                0x0F0D0B0907050301u64.cast_signed(), 0x0E0C0A0806040200u64.cast_signed()
                            )
                        }
                    }
                    16 => {
                        // For 16-bit elements: move even indices to low half, odd to high half
                        quote! {
                            _mm512_set4_epi64(
                                0x0F0E0B0A07060302u64.cast_signed(), 0x0D0C090805040100u64.cast_signed(),
                                0x0F0E0B0A07060302u64.cast_signed(), 0x0D0C090805040100u64.cast_signed()
                            )
                        }
                    }
                    _ => unreachable!(),
                };

                let shuffle_immediate = if select_even {
                    quote! { 0b01_00_01_00 }
                } else {
                    quote! { 0b11_10_11_10 }
                };

                quote! {
                    unsafe {
                        let mask = #mask;
                        let a_shuffled = _mm512_shuffle_epi8(a.into(), mask);
                        let b_shuffled = _mm512_shuffle_epi8(b.into(), mask);

                        let a_packed = _mm512_permutex_epi64::<0b11_01_10_00>(a_shuffled);
                        let b_packed = _mm512_permutex_epi64::<0b11_01_10_00>(b_shuffled);

                        _mm512_shuffle_i64x2::<#shuffle_immediate>(a_packed, b_packed).simd_into(self)
                    }
                }
            }
            _ => unimplemented!(),
        };

        quote! {
            #method_sig {
                #expr
            }
        }
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
        let byte_shift = if scalar_bytes == 1 {
            quote! { SHIFT }
        } else {
            quote! { SHIFT * #scalar_bytes }
        };

        // For AVX-512 with 512-bit AcrossBlocks, we need special handling since we have __m512i directly
        if *self == Self::Avx512 && vec_ty.n_bits() == 512 && granularity == AcrossBlocks {
            return quote! {
                #method_sig {
                    unsafe {
                        if SHIFT >= #max_shift {
                            return b;
                        }

                        // b and a are swapped here to match ARM's vext semantics
                        let result = cross_block_alignr_512(
                            self.#to_bytes(b).val.0,
                            self.#to_bytes(a).val.0,
                            #byte_shift
                        );
                        self.#from_bytes(#combined_bytes { val: #block_wrapper(result), simd: self })
                    }
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
            (AcrossBlocks, 256, Self::Sse4_2) => {
                // Inter-block shift or rotate in SSE4.2: use cross_block_alignr
                format_ident!("cross_block_alignr_128x{}", vec_ty.n_bits() / 128)
            }
            (AcrossBlocks, 256, Self::Avx2 | Self::Avx512) => {
                format_ident!("cross_block_alignr_256x{}", vec_ty.n_bits() / 256)
            }
            (AcrossBlocks, 512, Self::Sse4_2) => {
                // 512-bit AcrossBlocks for SSE4.2: use 128-bit blocks
                format_ident!("cross_block_alignr_128x{}", vec_ty.n_bits() / 128)
            }
            (AcrossBlocks, 512, Self::Avx2) => {
                // 512-bit AcrossBlocks for AVX2: use 256-bit blocks
                format_ident!("cross_block_alignr_256x{}", vec_ty.n_bits() / 256)
            }
            _ => unimplemented!(),
        };

        quote! {
            #method_sig {
                unsafe {
                    if SHIFT >= #max_shift {
                        return b;
                    }

                    // b and a are swapped here to match ARM's vext semantics. For vext, we can think of `a` as the "left",
                    // and we concatenate `b` to its "right". This makes sense, since `a` is the left-hand side and `b` is
                    // the right-hand side. x86's `alignr` is backwards, and treats `b` as the high/left block.
                    let result = #alignr_op(self.#to_bytes(b).val.0, self.#to_bytes(a).val.0, #byte_shift);
                    self.#from_bytes(#combined_bytes { val: #block_wrapper(result), simd: self })
                }
            }
        }
    }

    pub(crate) fn handle_cvt(
        &self,
        method_sig: TokenStream,
        vec_ty: &VecType,
        target_scalar: ScalarType,
        target_scalar_bits: usize,
        precise: bool,
    ) -> TokenStream {
        assert_eq!(
            vec_ty.scalar_bits, target_scalar_bits,
            "we currently only support converting between types of the same width"
        );

        // AVX-512 has native unsigned conversion intrinsics, so handle it specially
        if *self == Self::Avx512 && vec_ty.n_bits() == 512 {
            let expr = match (vec_ty.scalar, target_scalar, precise) {
                (ScalarType::Float, ScalarType::Int, false) => {
                    // f32 -> i32: _mm512_cvttps_epi32
                    quote! {
                        unsafe {
                            _mm512_cvttps_epi32(a.into()).simd_into(self)
                        }
                    }
                }
                (ScalarType::Float, ScalarType::Int, true) => {
                    // f32 -> i32 precise: handle out-of-range values and NaN
                    // Saturate to i32::MAX if >= 2147483648.0, NaN becomes 0
                    quote! {
                        unsafe {
                            let a = a.into();
                            let mut converted = _mm512_cvttps_epi32(a);

                            // In the common case where everything is in range, we don't need to do anything else.
                            let in_range_mask = _mm512_cmp_ps_mask::<{ _CMP_LT_OQ }>(a, _mm512_set1_ps(2147483648.0));
                            let all_in_range = in_range_mask == 0xFFFF;

                            if !all_in_range {
                                // If we are above i32::MAX (2147483647), clamp to it.
                                converted = _mm512_mask_blend_epi32(in_range_mask, _mm512_set1_epi32(i32::MAX), converted);
                                // Set NaN to 0.
                                let is_not_nan_mask = _mm512_cmp_ps_mask::<{ _CMP_ORD_Q }>(a, a);
                                converted = _mm512_maskz_mov_epi32(is_not_nan_mask, converted);
                                // We don't need to handle negative overflow because Intel's "invalid result" sentinel
                                // value is -2147483648, which is what we want anyway.
                            }

                            converted.simd_into(self)
                        }
                    }
                }
                (ScalarType::Float, ScalarType::Unsigned, false) => {
                    // f32 -> u32: _mm512_cvttps_epu32 (native unsigned support!)
                    quote! {
                        unsafe {
                            _mm512_cvttps_epu32(a.into()).simd_into(self)
                        }
                    }
                }
                (ScalarType::Float, ScalarType::Unsigned, true) => {
                    // f32 -> u32 precise: handle out-of-range values and NaN
                    // Saturate to u32::MAX if >= 4294967040.0, NaN and negative become 0
                    quote! {
                        unsafe {
                            // Clamp negative values and NaN to 0. Intel's `_mm512_max_ps` always takes the second
                            // operand if the first is NaN.
                            let a = _mm512_max_ps(a.into(), _mm512_setzero_ps());
                            let mut converted = _mm512_cvttps_epu32(a);

                            // Check if any value exceeds u32::MAX representable in f32 (4294967040.0)
                            let exceeds_range_mask = _mm512_cmp_ps_mask::<{ _CMP_GT_OQ }>(a, _mm512_set1_ps(4294967040.0));

                            if exceeds_range_mask != 0 {
                                // Clamp to u32::MAX.
                                converted = _mm512_mask_blend_epi32(exceeds_range_mask, converted, _mm512_set1_epi32(u32::MAX.cast_signed()));
                            }

                            converted.simd_into(self)
                        }
                    }
                }
                (ScalarType::Int, ScalarType::Float, _) => {
                    // i32 -> f32: _mm512_cvtepi32_ps
                    quote! {
                        unsafe {
                            _mm512_cvtepi32_ps(a.into()).simd_into(self)
                        }
                    }
                }
                (ScalarType::Unsigned, ScalarType::Float, _) => {
                    // u32 -> f32: _mm512_cvtepu32_ps (native unsigned support!)
                    quote! {
                        unsafe {
                            _mm512_cvtepu32_ps(a.into()).simd_into(self)
                        }
                    }
                }
                _ => unimplemented!(
                    "512-bit conversion from {:?} to {:?}",
                    vec_ty.scalar,
                    target_scalar
                ),
            };

            return quote! {
                #method_sig {
                    #expr
                }
            };
        }

        let expr = match (vec_ty.scalar, target_scalar) {
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
                            unsafe {
                                #convert(a.into()).simd_into(self)
                            }
                        }
                    }
                    (ScalarType::Unsigned, false) => {
                        quote! {
                            unsafe {
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

                                converted.simd_into(self)
                            }
                        }
                    }
                    (ScalarType::Int, true) => {
                        quote! {
                            unsafe {
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

                                converted.simd_into(self)
                            }
                        }
                    }
                    (ScalarType::Unsigned, true) => {
                        quote! {
                            unsafe {
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

                                converted.simd_into(self)
                            }
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
                    unsafe {
                        #intrinsic(a.into()).simd_into(self)
                    }
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
                    unsafe {
                        let a = a.into();
                        let lo = #blend::<0xAA>(a, #set1_int(0x4B000000));
                        let hi = #blend::<0xAA>(#srli::<16>(a), #set1_int(0x53000000));

                        let fhi = #sub_float(#cast_to_float(hi), #set1_float(f32::from_bits(0x53000080)));
                        let result = #add_float(#cast_to_float(lo), fhi);

                        result.simd_into(self)
                    }
                }
            }
            _ => unimplemented!(),
        };

        quote! {
            #method_sig {
                #expr
            }
        }
    }

    pub(crate) fn handle_reinterpret(
        &self,
        level: &impl Level,
        method_sig: TokenStream,
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
            quote! {
                #method_sig {
                    #arch_ty::from(a).simd_into(self)
                }
            }
        } else {
            let ident = cast_ident(
                vec_ty.scalar,
                target_ty,
                vec_ty.scalar_bits,
                scalar_bits,
                vec_ty.n_bits(),
            );
            quote! {
                #method_sig {
                    unsafe {
                        #ident(a.into()).simd_into(self)
                    }
                }
            }
        }
    }

    pub(crate) fn handle_mask_reduce(
        &self,
        method_sig: TokenStream,
        vec_ty: &VecType,
        quantifier: Quantifier,
        condition: bool,
    ) -> TokenStream {
        assert_eq!(
            vec_ty.scalar,
            ScalarType::Mask,
            "mask reduce ops only operate on masks"
        );

        // AVX-512 uses mask registers instead of movemask
        if *self == Self::Avx512 && vec_ty.n_bits() == 512 {
            // Use _mm512_movepi*_mask to convert to a mask register, then check the mask
            // The mask return type depends on the element size:
            // - 8-bit elements: __mmask64 (u64)
            // - 16-bit elements: __mmask32 (u32)
            // - 32-bit elements: __mmask16 (u16)
            // - 64-bit elements: __mmask8 (u8)
            let (move_mask, all_ones) = match vec_ty.scalar_bits {
                8 => (
                    format_ident!("_mm512_movepi8_mask"),
                    quote! { 0xFFFFFFFFFFFFFFFFu64 },
                ),
                16 => (
                    format_ident!("_mm512_movepi16_mask"),
                    quote! { 0xFFFFFFFFu32 },
                ),
                32 => (format_ident!("_mm512_movepi32_mask"), quote! { 0xFFFFu16 }),
                64 => (format_ident!("_mm512_movepi64_mask"), quote! { 0xFFu8 }),
                _ => unreachable!(),
            };

            let op = match (quantifier, condition) {
                (Quantifier::Any, true) => quote! { != 0 },
                (Quantifier::Any, false) => quote! { != #all_ones },
                (Quantifier::All, true) => quote! { == #all_ones },
                (Quantifier::All, false) => quote! { == 0 },
            };

            return quote! {
                #method_sig {
                    unsafe {
                        #move_mask(a.into()) #op
                    }
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

        quote! {
            #method_sig {
                unsafe {
                    #movemask as u32 #op
                }
            }
        }
    }

    pub(crate) fn handle_load_interleaved(
        &self,
        method_sig: TokenStream,
        vec_ty: &VecType,
        block_size: u16,
        block_count: u16,
    ) -> TokenStream {
        assert_eq!(
            block_size, 128,
            "only 128-bit blocks are currently supported"
        );
        assert_eq!(block_count, 4, "only count of 4 is currently supported");
        let expr = match vec_ty.scalar_bits {
            32 | 16 | 8 => {
                let block_ty =
                    VecType::new(vec_ty.scalar, vec_ty.scalar_bits, 128 / vec_ty.scalar_bits);
                let load_unaligned =
                    intrinsic_ident("loadu", coarse_type(&block_ty), block_ty.n_bits());
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

                quote! {
                    unsafe {
                        let v0 = #load_unaligned(src.as_ptr() as *const _);
                        let v1 = #load_unaligned(src.as_ptr().add(#block_len) as *const _);
                        let v2 = #load_unaligned(src.as_ptr().add(2 * #block_len) as *const _);
                        let v3 = #load_unaligned(src.as_ptr().add(3 * #block_len) as *const _);

                        #init_shuffle

                        let tmp0 = #unpacklo_32(v0, v1); // [0,4,1,5]
                        let tmp1 = #unpackhi_32(v0, v1); // [2,6,3,7]
                        let tmp2 = #unpacklo_32(v2, v3); // [8,12,9,13]
                        let tmp3 = #unpackhi_32(v2, v3); // [10,14,11,15]

                        #final_unpack

                        self.#combine_full(
                            self.#combine_half(out0.simd_into(self), out1.simd_into(self)),
                            self.#combine_half(out2.simd_into(self), out3.simd_into(self)),
                        )
                    }
                }
            }
            _ => unimplemented!(),
        };

        quote! {
            #method_sig {
                #expr
            }
        }
    }

    pub(crate) fn handle_store_interleaved(
        &self,
        method_sig: TokenStream,
        vec_ty: &VecType,
        block_size: u16,
        block_count: u16,
    ) -> TokenStream {
        assert_eq!(
            block_size, 128,
            "only 128-bit blocks are currently supported"
        );
        assert_eq!(block_count, 4, "only count of 4 is currently supported");
        let expr = match vec_ty.scalar_bits {
            32 | 16 | 8 => {
                let block_ty =
                    VecType::new(vec_ty.scalar, vec_ty.scalar_bits, 128 / vec_ty.scalar_bits);
                let store_unaligned =
                    intrinsic_ident("storeu", coarse_type(&block_ty), block_ty.n_bits());
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

                quote! {
                    let (v01, v23) = self.#split_full(a);
                    let (v0, v1) = self.#split_half(v01);
                    let (v2, v3) = self.#split_half(v23);
                    let v0 = v0.into();
                    let v1 = v1.into();
                    let v2 = v2.into();
                    let v3 = v3.into();

                    unsafe {
                        let tmp0 = #unpacklo_32(v0, v1); // [0,4,1,5]
                        let tmp1 = #unpackhi_32(v0, v1); // [2,6,3,7]
                        let tmp2 = #unpacklo_32(v2, v3); // [8,12,9,13]
                        let tmp3 = #unpackhi_32(v2, v3); // [10,14,11,15]

                        #final_unpack

                        #post_shuffle

                        #store_unaligned(dest.as_mut_ptr() as *mut _, out0);
                        #store_unaligned(dest.as_mut_ptr().add(#block_len) as *mut _, out1);
                        #store_unaligned(dest.as_mut_ptr().add(2 * #block_len) as *mut _, out2);
                        #store_unaligned(dest.as_mut_ptr().add(3 * #block_len) as *mut _, out3);
                    }
                }
            }
            _ => unimplemented!(),
        };

        quote! {
            #method_sig {
                #expr
            }
        }
    }

    /// Generates versions of the "alignr" intrinsics that take the shift amount as a regular argument instead of a
    /// const generic argument, to make them easier to use in higher-level operations. These are low-level helpers that
    /// inherit the semantics of the underlying `alignr` intrinsics, so the argument order is backwards from ARM's
    /// `vext` and our `slide` operation, and the 256-bit AVX2 version still operates *within* 128-bit lanes.
    fn dyn_alignr_helpers(&self) -> TokenStream {
        let mut fns = vec![];

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
                /// This is a version of the `alignr` intrinsic that takes a non-const shift argument. The shift is still
                /// expected to be constant in practice, so the match statement will be optimized out. This exists because
                /// Rust doesn't currently let you do math on const generics.
                #[inline(always)]
                unsafe fn #helper_name(a: #arch_ty, b: #arch_ty, shift: usize) -> #arch_ty {
                    unsafe {
                        match shift {
                            #(#shifts,)*
                            _ => unreachable!()
                        }
                    }
                }
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
                /// Concatenates `b` and `a` (each N blocks) and extracts N blocks starting at byte offset `shift_bytes`.
                /// Extracts from [b : a] (b in low bytes, a in high bytes), matching `alignr` semantics.
                #[inline(always)]
                unsafe fn #helper_name(a: [__m128i; #num_blocks], b: [__m128i; #num_blocks], shift_bytes: usize) -> [__m128i; #num_blocks] {
                    [#({
                        let [lo, hi] = crate::support::cross_block_slide_blocks_at(&b, &a, #blocks_idx, shift_bytes);
                        unsafe { dyn_alignr_128(hi, lo, shift_bytes % 16) }
                    }),*]
                }
            });
        }

        quote! {
            #(#fns)*
        }
    }

    fn avx2_slide_helpers() -> TokenStream {
        quote! {
            /// Computes one output __m256i for `cross_block_alignr_*` operations.
            ///
            /// Given an array of registers, each containing two 128-bit blocks, extracts two adjacent blocks (`lo_idx` and
            /// `hi_idx` = `lo_idx + 1`) and performs `alignr` with `intra_shift`.
            #[inline(always)]
            unsafe fn cross_block_alignr_one(regs: &[__m256i], block_idx: usize, shift_bytes: usize) -> __m256i {
                let lo_idx = block_idx + (shift_bytes / 16);
                let intra_shift = shift_bytes % 16;
                let lo_blocks = if lo_idx & 1 == 0 {
                    regs[lo_idx / 2]
                } else {
                    unsafe { _mm256_permute2x128_si256::<0x21>(regs[lo_idx / 2], regs[(lo_idx / 2) + 1]) }
                };

                // For hi_blocks, we need blocks (`lo_idx + 1`) and (`lo_idx + 2`)
                let hi_idx = lo_idx + 1;
                let hi_blocks = if hi_idx & 1 == 0 {
                    regs[hi_idx / 2]
                } else {
                    unsafe { _mm256_permute2x128_si256::<0x21>(regs[hi_idx / 2], regs[(hi_idx / 2) + 1]) }
                };

                unsafe { dyn_alignr_256(hi_blocks, lo_blocks, intra_shift) }
            }

            /// Concatenates `b` and `a` (each 2 x __m256i = 4 blocks) and extracts 4 blocks starting at byte offset
            /// `shift_bytes`. Extracts from [b : a] (b in low bytes, a in high bytes), matching alignr semantics.
            #[inline(always)]
            unsafe fn cross_block_alignr_256x2(a: [__m256i; 2], b: [__m256i; 2], shift_bytes: usize) -> [__m256i; 2] {
                // Concatenation is [b : a], so b blocks come first
                let regs = [b[0], b[1], a[0], a[1]];

                unsafe {
                    [
                        cross_block_alignr_one(&regs, 0, shift_bytes),
                        cross_block_alignr_one(&regs, 2, shift_bytes),
                    ]
                }
            }

            /// Concatenates `b` and `a` (each 1 x __m256i = 2 blocks) and extracts 2 blocks starting at byte offset
            /// `shift_bytes`. Extracts from [b : a] (b in low bytes, a in high bytes), matching alignr semantics.
            #[inline(always)]
            unsafe fn cross_block_alignr_256x1(a: __m256i, b: __m256i, shift_bytes: usize) -> __m256i {
                // Concatenation is [b : a], so b comes first
                let regs = [b, a];

                unsafe {
                    cross_block_alignr_one(&regs, 0, shift_bytes)
                }
            }
        }
    }

    fn avx512_slide_helpers() -> TokenStream {
        // Generate index vectors for each possible shift amount (0..64).
        // For shift S, we want to extract bytes [S..S+64) from the concatenation [b : a].
        // In _mm512_permutex2var_epi8, indices 0-63 select from the first arg, 64-127 from the second.
        // Since we want [b : a] with b in low bytes: b is first arg, a is second arg.
        // So index i in result should get byte (S + i) from [b : a]:
        //   - if (S + i) < 64: from b at position (S + i), index = S + i
        //   - if (S + i) >= 64: from a at position (S + i - 64), index = 64 + (S + i - 64) = S + i
        // So the index is simply (S + i) for all cases, which works because indices wrap at 128.
        let match_arms: Vec<_> = (0usize..64)
            .map(|shift| {
                let indices: Vec<_> = (0u8..64)
                    .map(|i| (shift as u8).wrapping_add(i) as i8)
                    .collect();
                // _mm512_set_epi8 takes arguments in reverse order (element 63 first, element 0 last)
                let indices_rev: Vec<_> = indices.into_iter().rev().collect();
                quote! {
                    #shift => _mm512_set_epi8(#(#indices_rev),*)
                }
            })
            .collect();

        quote! {
            /// Concatenates `b` and `a` (each __m512i = 4 x 128-bit blocks) and extracts 4 blocks starting at byte offset
            /// `shift_bytes`. Extracts from [b : a] (b in low bytes, a in high bytes), matching alignr semantics.
            /// Uses AVX-512 VBMI's permutex2var for efficient cross-lane byte shuffling.
            #[inline(always)]
            unsafe fn cross_block_alignr_512(a: __m512i, b: __m512i, shift_bytes: usize) -> __m512i {
                // Use _mm512_permutex2var_epi8 (VBMI) to select 64 bytes from the 128-byte concatenation.
                // The index vector specifies which byte to select: 0-63 from b, 64-127 from a.
                let idx = unsafe {
                    match shift_bytes {
                        #(#match_arms,)*
                        _ => unreachable!()
                    }
                };
                unsafe { _mm512_permutex2var_epi8(b, idx, a) }
            }
        }
    }
}
