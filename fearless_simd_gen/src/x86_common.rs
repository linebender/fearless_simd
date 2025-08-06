use crate::arch::Arch;
use crate::arch::sse4_2::{
    Sse4_2,
};
use crate::generic::{generic_combine, generic_split};
use crate::ops::{OpSig, TyFlavor, reinterpret_ty, valid_reinterpret};
use crate::types::{ScalarType, VecType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use crate::types::ScalarType::{Float, Int, Mask, Unsigned};

pub(crate) fn make_method(
    method: &str,
    sig: OpSig,
    vec_ty: &VecType,
    arch: impl Arch,
    bit_length: usize,
) -> TokenStream {
    let scalar_bits = vec_ty.scalar_bits;
    let ty_name = vec_ty.rust_name();
    let method_name = format!("{method}_{ty_name}");
    let method_ident = Ident::new(&method_name, Span::call_site());
    let ret_ty = sig.ret_ty(vec_ty, TyFlavor::SimdTrait);
    let args = sig.simd_trait_args(vec_ty);
    let method_sig = quote! {
        #[inline(always)]
        fn #method_ident(#args) -> #ret_ty
    };

    match sig {
        OpSig::Splat => {
            let intrinsic = set1_intrinsic(vec_ty.scalar, scalar_bits);
            let cast = match vec_ty.scalar {
                ScalarType::Unsigned => quote!(as _),
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
        OpSig::Compare => {
            let args = [quote! { a.into() }, quote! { b.into() }];

            let mut expr = if vec_ty.scalar != ScalarType::Float {
                if matches!(method, "simd_le" | "simd_ge") {
                    let max_min = match method {
                        "simd_le" => "min",
                        "simd_ge" => "max",
                        _ => unreachable!(),
                    };

                    let eq_intrinsic =
                        simple_sign_unaware_intrinsic("cmpeq", vec_ty.scalar, vec_ty.scalar_bits);

                    let max_min_expr = arch.expr(max_min, vec_ty, &args);
                    quote! { #eq_intrinsic(#max_min_expr, a.into()) }
                } else if vec_ty.scalar == ScalarType::Unsigned {
                    // SSE4.2 only has signed GT/LT, but not unsigned.
                    let set = set1_intrinsic(vec_ty.scalar, vec_ty.scalar_bits);
                    let sign = match vec_ty.scalar_bits {
                        8 => quote! { 0x80u8 },
                        16 => quote! { 0x8000u16 },
                        32 => quote! { 0x80000000u32 },
                        _ => unimplemented!(),
                    };
                    let gt =
                        simple_sign_unaware_intrinsic("cmpgt", vec_ty.scalar, vec_ty.scalar_bits);
                    let args = if method == "simd_lt" {
                        quote! { b_signed, a_signed }
                    } else {
                        quote! { a_signed, b_signed }
                    };

                    quote! {
                        let sign_bit = #set(#sign as _);
                        let a_signed = _mm_xor_si128(a.into(), sign_bit);
                        let b_signed = _mm_xor_si128(b.into(), sign_bit);

                        #gt(#args)
                    }
                } else {
                    arch.expr(method, vec_ty, &args)
                }
            } else {
                arch.expr(method, vec_ty, &args)
            };

            if vec_ty.scalar == ScalarType::Float {
                let suffix = op_suffix(vec_ty.scalar, scalar_bits, false);
                let ident = format_ident!("_mm_cast{suffix}_si128");
                expr = quote! { #ident(#expr) }
            }

            quote! {
                #method_sig {
                    unsafe { #expr.simd_into(self) }
                }
            }
        }
        OpSig::Unary => match method {
            "fract" => {
                quote! {
                    #method_sig {
                        a - a.trunc()
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
                let expr = arch.expr(method, vec_ty, &args);
                quote! {
                    #method_sig {
                        unsafe { #expr.simd_into(self) }
                    }
                }
            }
        },
        OpSig::WidenNarrow(t) => match method {
            "widen" => {
                let extend = extend_intrinsic(vec_ty.scalar, scalar_bits, t.scalar_bits);
                let combine = format_ident!(
                    "combine_{}",
                    VecType {
                        len: vec_ty.len / 2,
                        scalar_bits: scalar_bits * 2,
                        ..*vec_ty
                    }
                    .rust_name()
                );
                quote! {
                    #method_sig {
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
            }
            "narrow" => {
                let mask = set1_intrinsic(vec_ty.scalar, scalar_bits);
                let pack = pack_intrinsic(scalar_bits, matches!(vec_ty.scalar, ScalarType::Int));
                let split = format_ident!("split_{}", vec_ty.rust_name());
                quote! {
                    #method_sig {
                        let (a, b) = self.#split(a);
                        unsafe {
                            // Note that SSE4.2 only has an intrinsic for saturating cast,
                            // but not wrapping.
                            let mask = #mask(0xFF);
                            let lo_masked = _mm_and_si128(a.into(), mask);
                            let hi_masked = _mm_and_si128(b.into(), mask);
                            let result = #pack(lo_masked, hi_masked);
                            result.simd_into(self)
                        }
                    }
                }
            }
            _ => unreachable!(),
        },
        OpSig::Binary => {
            if method == "mul" && vec_ty.scalar_bits == 8 {
                quote! {
                    #method_sig {
                        todo!()
                    }
                }
            } else {
                let args = [quote! { a.into() }, quote! { b.into() }];
                let expr = arch.expr(method, vec_ty, &args);
                quote! {
                    #method_sig {
                        unsafe { #expr.simd_into(self) }
                    }
                }
            }
        }
        OpSig::Shift => {
            let op = match vec_ty.scalar {
                ScalarType::Unsigned => "srl",
                ScalarType::Int => "sra",
                _ => unreachable!(),
            };
            let suffix = op_suffix(vec_ty.scalar, scalar_bits.max(16), false);
            let shift_intrinsic = format_ident!("_mm_{op}_{suffix}");

            if scalar_bits == 8 {
                // SSE doesn't have shifting for 8-bit, so we first convert into
                // 16 bit, shift, and then back to 8-bit

                let unpack_hi = unpack_intrinsic(ScalarType::Int, 8, false);
                let unpack_lo = unpack_intrinsic(ScalarType::Int, 8, true);

                let extend_expr = |expr| match vec_ty.scalar {
                    ScalarType::Unsigned => quote! {
                        #expr(val, _mm_setzero_si128())
                    },
                    ScalarType::Int => quote! {
                         #expr(val, _mm_cmplt_epi8(val, _mm_setzero_si128()))
                    },
                    _ => unimplemented!(),
                };

                let extend_intrinsic_lo = extend_expr(unpack_lo);
                let extend_intrinsic_hi = extend_expr(unpack_hi);
                let pack_intrinsic = pack_intrinsic(16, vec_ty.scalar == ScalarType::Int);

                quote! {
                    #method_sig {
                        unsafe {
                            let val = a.into();
                            let shift_count = _mm_cvtsi32_si128(shift as i32);

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
                        unsafe { #shift_intrinsic(a.into(), _mm_cvtsi32_si128(shift as _)).simd_into(self) }
                    }
                }
            }
        }
        OpSig::Ternary => match method {
            "madd" => {
                quote! {
                    #method_sig {
                        a + b * c
                    }
                }
            }
            "msub" => {
                quote! {
                    #method_sig {
                        a - b * c
                    }
                }
            }
            _ => {
                let args = [
                    quote! { a.into() },
                    quote! { b.into() },
                    quote! { c.into() },
                ];

                let expr = Sse4_2.expr(method, vec_ty, &args);
                quote! {
                    #method_sig {
                       #expr.simd_into(self)
                    }
                }
            }
        },
        OpSig::Select => {
            let expr = if vec_ty.scalar == ScalarType::Float {
                let suffix = op_suffix(vec_ty.scalar, scalar_bits, false);
                let (i1, i2, i3, i4) = (
                    format_ident!("_mm_castsi128_{suffix}"),
                    format_ident!("_mm_or_{suffix}"),
                    format_ident!("_mm_and_{suffix}"),
                    format_ident!("_mm_andnot_{suffix}"),
                );
                quote! {
                    let mask = #i1(a.into());

                    #i2(
                        #i3(mask, b.into()),
                        #i4(mask, c.into())
                    )
                }
            } else {
                quote! {
                    _mm_or_si128(
                        _mm_and_si128(a.into(), b.into()),
                        _mm_andnot_si128(a.into(), c.into())
                    )
                }
            };

            quote! {
                #method_sig {
                   unsafe {
                         #expr.simd_into(self)
                    }
                }
            }
        }
        OpSig::Combine => generic_combine(vec_ty),
        OpSig::Split => generic_split(vec_ty),
        OpSig::Zip(zip1) => {
            let op = if zip1 { "lo" } else { "hi" };

            let suffix = op_suffix(vec_ty.scalar, scalar_bits, false);
            let intrinsic = format_ident!("_mm_unpack{op}_{suffix}");

            quote! {
                #method_sig {
                   unsafe {  #intrinsic(a.into(), b.into()).simd_into(self) }
                }
            }
        }
        OpSig::Unzip(select_even) => {
            let expr = if vec_ty.scalar == ScalarType::Float {
                let suffix = op_suffix(vec_ty.scalar, scalar_bits, false);
                let intrinsic = format_ident!("_mm_shuffle_{suffix}");

                let mask = match (vec_ty.scalar_bits, select_even) {
                    (32, true) => quote! { 0b10_00_10_00 },
                    (32, false) => quote! { 0b11_01_11_01 },
                    (64, true) => quote! { 0b00 },
                    (64, false) => quote! { 0b11 },
                    _ => unimplemented!(),
                };

                quote! { unsafe { #intrinsic::<#mask>(a.into(), b.into()).simd_into(self) } }
            } else {
                match vec_ty.scalar_bits {
                    32 => {
                        let op = if select_even { "lo" } else { "hi" };

                        let intrinsic = format_ident!("_mm_unpack{op}_epi64");

                        quote! {
                              unsafe {
                                  let t1 = _mm_shuffle_epi32::<0b11_01_10_00>(a.into());
                                  let t2 = _mm_shuffle_epi32::<0b11_01_10_00>(b.into());
                                  #intrinsic(t1, t2).simd_into(self)
                            }
                        }
                    }
                    16 | 8 => {
                        let mask = match (scalar_bits, select_even) {
                            (8, true) => {
                                quote! { 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14  }
                            }
                            (8, false) => {
                                quote! { 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15  }
                            }
                            (16, true) => {
                                quote! { 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13 }
                            }
                            (16, false) => {
                                quote! {  2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15 }
                            }
                            _ => unreachable!(),
                        };

                        quote! {
                            unsafe {
                                let mask = _mm_setr_epi8(#mask);

                                let t1 = _mm_shuffle_epi8(a.into(), mask);
                                let t2 = _mm_shuffle_epi8(b.into(), mask);
                                _mm_unpacklo_epi64(t1, t2).simd_into(self)
                            }
                        }
                    }
                    _ => quote! { todo!() },
                }
            };

            quote! {
                #method_sig {
                    #expr
                }
            }
        }
        OpSig::Cvt(scalar, scalar_bits) => {
            // IMPORTANT TODO: for f32 to u32, we are currently converting it to i32 instead
            // of u32. We need to properly polyfill this.
            let cvt_intrinsic =
                cvt_intrinsic(*vec_ty, VecType::new(scalar, scalar_bits, vec_ty.len));

            let expr = if vec_ty.scalar == ScalarType::Float {
                let floor_intrinsic = simple_intrinsic("floor", vec_ty.scalar, vec_ty.scalar_bits);
                let max_intrinsic = simple_intrinsic("max", vec_ty.scalar, vec_ty.scalar_bits);
                let set = set1_intrinsic(vec_ty.scalar, vec_ty.scalar_bits);

                if scalar == ScalarType::Unsigned {
                    quote! { #max_intrinsic(#floor_intrinsic(a.into()), #set(0.0)) }
                } else {
                    quote! { a.trunc().into() }
                }
            } else {
                quote! { a.into() }
            };

            quote! {
                #method_sig {
                    unsafe { #cvt_intrinsic(#expr).simd_into(self) }
                }
            }
        }
        OpSig::Reinterpret(scalar, scalar_bits) => {
            if valid_reinterpret(vec_ty, scalar, scalar_bits) {
                let to_ty = reinterpret_ty(vec_ty, scalar, scalar_bits).rust();

                quote! {
                    #method_sig {
                        #to_ty {
                            val: bytemuck::cast(a.val),
                            simd: a.simd,
                        }
                    }
                }
            } else {
                quote! {}
            }
        }
        OpSig::LoadInterleaved(block_size, _) => {
            // Implementing interleaved loading/storing for 32-bit is still quite doable, It's unclear
            // how hard it would be for u16/u8. For now we only implement it for u32 since this is needed
            // in packing in vello_cpu, where performance is very critical.
            let expr = if block_size == 128
                && vec_ty.scalar == ScalarType::Unsigned
                && vec_ty.scalar_bits == 32
            {
                quote! {
                    unsafe {
                        // TODO: Once we support u64, we could do all of this using just zip + unzip
                        let v0 = _mm_loadu_si128(src.as_ptr().add(0) as *const __m128i);
                        let v1 = _mm_loadu_si128(src.as_ptr().add(4) as *const __m128i);
                        let v2 = _mm_loadu_si128(src.as_ptr().add(8) as *const __m128i);
                        let v3 = _mm_loadu_si128(src.as_ptr().add(12) as *const __m128i);

                        let tmp0 = _mm_unpacklo_epi32(v0, v1); // [0,4,1,5]
                        let tmp1 = _mm_unpackhi_epi32(v0, v1); // [2,6,3,7]
                        let tmp2 = _mm_unpacklo_epi32(v2, v3); // [8,12,9,13]
                        let tmp3 = _mm_unpackhi_epi32(v2, v3); // [10,14,11,15]

                        let out0 = _mm_unpacklo_epi64(tmp0, tmp2); // [0,4,8,12]
                        let out1 = _mm_unpackhi_epi64(tmp0, tmp2); // [1,5,9,13]
                        let out2 = _mm_unpacklo_epi64(tmp1, tmp3); // [2,6,10,14]
                        let out3 = _mm_unpackhi_epi64(tmp1, tmp3); // [3,7,11,15]

                        self.combine_u32x8(
                            self.combine_u32x4(out0.simd_into(self), out1.simd_into(self)),
                            self.combine_u32x4(out2.simd_into(self), out3.simd_into(self)),
                        )
                    }
                }
            } else {
                quote! { crate::Fallback::new().#method_ident(src).val.simd_into(self) }
            };

            quote! {
                #method_sig {
                    #expr
                }
            }
        }
        OpSig::StoreInterleaved(_, _) => {
            quote! {
                #method_sig {
                    let fb = crate::Fallback::new();
                    fb.#method_ident(a.val.simd_into(fb), dest);
                }
            }
        }
    }
}


pub(crate) fn op_suffix(mut ty: ScalarType, bits: usize, sign_aware: bool) -> &'static str {
    use ScalarType::*;
    if !sign_aware && ty == Unsigned {
        ty = Int;
    }
    match (ty, bits) {
        (Float, 32) => "ps",
        (Float, 64) => "pd",
        (Float, _) => unimplemented!("{bits} bit floats"),
        (Int | Mask, 8) => "epi8",
        (Int | Mask, 16) => "epi16",
        (Int | Mask, 32) => "epi32",
        (Int | Mask, 64) => "epi64",
        (Unsigned, 8) => "epu8",
        (Unsigned, 16) => "epu16",
        (Unsigned, 32) => "epu32",
        (Unsigned, 64) => "epu64",
        _ => unreachable!(),
    }
}

pub(crate) fn set1_intrinsic(ty: ScalarType, bits: usize) -> Ident {
    use ScalarType::*;
    let suffix = match (ty, bits) {
        (Int | Unsigned | Mask, 64) => "epi64x",
        _ => op_suffix(ty, bits, false),
    };
    format_ident!("_mm_set1_{suffix}")
}

pub(crate) fn simple_intrinsic(name: &str, ty: ScalarType, bits: usize) -> Ident {
    let suffix = op_suffix(ty, bits, true);
    format_ident!("_mm_{name}_{suffix}")
}

pub(crate) fn simple_sign_unaware_intrinsic(name: &str, ty: ScalarType, bits: usize) -> Ident {
    let suffix = op_suffix(ty, bits, false);
    format_ident!("_mm_{name}_{suffix}")
}

pub(crate) fn extend_intrinsic(ty: ScalarType, from_bits: usize, to_bits: usize) -> Ident {
    let from_suffix = op_suffix(ty, from_bits, true);
    let to_suffix = op_suffix(ty, to_bits, false);
    format_ident!("_mm_cvt{from_suffix}_{to_suffix}")
}

pub(crate) fn cvt_intrinsic(from: VecType, to: VecType) -> Ident {
    let from_suffix = op_suffix(from.scalar, from.scalar_bits, false);
    let to_suffix = op_suffix(to.scalar, to.scalar_bits, false);
    format_ident!("_mm_cvt{from_suffix}_{to_suffix}")
}

pub(crate) fn pack_intrinsic(from_bits: usize, signed: bool) -> Ident {
    let unsigned = match signed {
        true => "",
        false => "u",
    };
    let suffix = op_suffix(ScalarType::Int, from_bits, false);
    format_ident!("_mm_pack{unsigned}s_{suffix}")
}

pub(crate) fn unpack_intrinsic(scalar_type: ScalarType, scalar_bits: usize, low: bool) -> Ident {
    let suffix = op_suffix(scalar_type, scalar_bits, false);

    let low_pref = if low { "lo" } else { "hi" };
    format_ident!("_mm_unpack{low_pref}_{suffix}")
}
