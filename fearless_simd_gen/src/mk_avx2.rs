// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::Arch;
use crate::arch::avx2::Avx2;
use crate::arch::sse4_2::Sse4_2;
use crate::generic::{generic_combine, generic_op, generic_split, scalar_binary};
use crate::mk_sse4_2;
use crate::ops::{OpSig, TyFlavor, ops_for_type};
use crate::types::{SIMD_TYPES, ScalarType, VecType, type_imports};
use crate::x86_common::{cast_ident, simple_intrinsic};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

#[derive(Clone, Copy)]
pub(crate) struct Level;

impl Level {
    fn name(self) -> &'static str {
        "Avx2"
    }

    fn token(self) -> TokenStream {
        let ident = Ident::new(self.name(), Span::call_site());
        quote! { #ident }
    }
}

pub(crate) fn mk_avx2_impl() -> TokenStream {
    let imports = type_imports();
    let simd_impl = mk_simd_impl();
    let ty_impl = mk_type_impl();

    quote! {
        // Until we have implemented all functions.
        #![expect(
            unused_variables,
            clippy::todo,
            reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
        )]

        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        use core::ops::*;
        use crate::{seal::Seal, Level, Simd, SimdFrom, SimdInto};

        #imports

        /// The SIMD token for the "AVX2" and "FMA" level.
        #[derive(Clone, Copy, Debug)]
        pub struct Avx2 {
            pub avx2: crate::core_arch::x86::Avx2,
        }

        impl Avx2 {
            /// Create a SIMD token.
            ///
            /// # Safety
            ///
            /// The AVX2 and FMA CPU feature must be available.
            #[inline]
            pub const unsafe fn new_unchecked() -> Self {
                Avx2 {
                    avx2: unsafe { crate::core_arch::x86::Avx2::new_unchecked() },
                }
            }
        }

        impl Seal for Avx2 {}

        #simd_impl

        #ty_impl
    }
}

fn mk_simd_impl() -> TokenStream {
    let level_tok = Level.token();
    let mut methods = vec![];
    for vec_ty in SIMD_TYPES {
        for (method, sig) in ops_for_type(vec_ty, true) {
            let too_wide = vec_ty.n_bits() > 256;

            let acceptable_wide_op = matches!(method, "load_interleaved_128")
                || matches!(method, "store_interleaved_128");

            if too_wide && !acceptable_wide_op {
                methods.push(generic_op(method, sig, vec_ty));
                continue;
            }

            let method = make_method(method, sig, vec_ty, Sse4_2, vec_ty.n_bits());

            methods.push(method);
        }
    }

    // Note: the `vectorize` implementation is pretty boilerplate and should probably
    // be factored out for DRY.
    quote! {
        impl Simd for #level_tok {
            type f32s = f32x4<Self>;
            type u8s = u8x16<Self>;
            type i8s = i8x16<Self>;
            type u16s = u16x8<Self>;
            type i16s = i16x8<Self>;
            type u32s = u32x4<Self>;
            type i32s = i32x4<Self>;
            type mask8s = mask8x16<Self>;
            type mask16s = mask16x8<Self>;
            type mask32s = mask32x4<Self>;
            #[inline(always)]
            fn level(self) -> Level {
                Level::#level_tok(self)
            }

            #[inline]
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                #[target_feature(enable = "avx2,fma")]
                #[inline]
                unsafe fn vectorize_avx2<F: FnOnce() -> R, R>(f: F) -> R {
                    f()
                }
                unsafe { vectorize_avx2(f) }
            }

            #( #methods )*
        }
    }
}

fn mk_type_impl() -> TokenStream {
    let mut result = vec![];
    for ty in SIMD_TYPES {
        let n_bits = ty.n_bits();
        if n_bits != 256 {
            continue;
        }
        let simd = ty.rust();
        let arch = Avx2.arch_ty(ty);
        result.push(quote! {
            impl<S: Simd> SimdFrom<#arch, S> for #simd<S> {
                #[inline(always)]
                fn simd_from(arch: #arch, simd: S) -> Self {
                    Self {
                        val: unsafe { core::mem::transmute(arch) },
                        simd
                    }
                }
            }
            impl<S: Simd> From<#simd<S>> for #arch {
                #[inline(always)]
                fn from(value: #simd<S>) -> Self {
                    unsafe { core::mem::transmute(value.val) }
                }
            }
        });
    }
    quote! {
        #( #result )*
    }
}

fn make_method(
    method: &str,
    sig: OpSig,
    vec_ty: &VecType,
    arch: impl Arch,
    ty_bits: usize,
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

    if method == "shrv" && scalar_bits < 32 {
        return scalar_binary(&method_ident, quote!(core::ops::Shr::shr), vec_ty);
    }

    match sig {
        OpSig::Splat => mk_sse4_2::handle_splat(method_sig, vec_ty, scalar_bits, ty_bits),
        OpSig::Compare => handle_compare(method_sig, method, vec_ty, scalar_bits, ty_bits, arch),
        OpSig::Unary => mk_sse4_2::handle_unary(method_sig, method, vec_ty, arch),
        OpSig::WidenNarrow(t) => {
            if vec_ty.n_bits() > 128 && method == "widen" {
                generic_op(method, sig, vec_ty)
            } else {
                mk_sse4_2::handle_widen_narrow(method_sig, method, vec_ty, scalar_bits, ty_bits, t)
            }
        }
        OpSig::Binary => mk_sse4_2::handle_binary(method_sig, method, vec_ty, arch),
        OpSig::Shift => mk_sse4_2::handle_shift(method_sig, method, vec_ty, scalar_bits, ty_bits),
        OpSig::Ternary => match method {
            "madd" => {
                let intrinsic =
                    simple_intrinsic("fmadd", vec_ty.scalar, vec_ty.scalar_bits, vec_ty.n_bits());
                quote! {
                    #method_sig {
                        unsafe { #intrinsic(a.into(), b.into(), c.into()).simd_into(self) }
                    }
                }
            }
            "msub" => {
                let intrinsic =
                    simple_intrinsic("fmsub", vec_ty.scalar, vec_ty.scalar_bits, vec_ty.n_bits());
                quote! {
                    #method_sig {
                        unsafe { #intrinsic(a.into(), b.into(), c.into()).simd_into(self) }
                    }
                }
            }
            _ => mk_sse4_2::handle_ternary(method_sig, &method_ident, method, vec_ty),
        },
        OpSig::Select => mk_sse4_2::handle_select(method_sig, vec_ty, scalar_bits),
        OpSig::Combine => generic_combine(vec_ty),
        OpSig::Split => generic_split(vec_ty),
        OpSig::Zip(zip1) => mk_sse4_2::handle_zip(method_sig, vec_ty, scalar_bits, zip1),
        OpSig::Unzip(select_even) => {
            mk_sse4_2::handle_unzip(method_sig, vec_ty, scalar_bits, select_even)
        }
        OpSig::Cvt(scalar, target_scalar_bits) => {
            mk_sse4_2::handle_cvt(method_sig, vec_ty, ty_bits, scalar, target_scalar_bits)
        }
        OpSig::Reinterpret(scalar, target_scalar_bits) => {
            mk_sse4_2::handle_reinterpret(method_sig, vec_ty, scalar, target_scalar_bits)
        }
        OpSig::LoadInterleaved(block_size, _) => {
            mk_sse4_2::handle_load_interleaved(method_sig, &method_ident, vec_ty, block_size)
        }
        OpSig::StoreInterleaved(_, _) => {
            mk_sse4_2::handle_store_interleaved(method_sig, &method_ident)
        }
    }
}

pub(crate) fn handle_compare(
    method_sig: TokenStream,
    method: &str,
    vec_ty: &VecType,
    scalar_bits: usize,
    ty_bits: usize,
    arch: impl Arch,
) -> TokenStream {
    if vec_ty.scalar == ScalarType::Float {
        // For AVX2 and up, Intel gives us a generic comparison intrinsic that takes a predicate. There are 32,
        // of which only a few are useful and the rest will violate IEEE754 and/or raise a SIGFPE on NaN.
        //
        // https://www.felixcloutier.com/x86/cmppd#tbl-3-1
        let order_predicate = match method {
            "simd_eq" => 0x00,
            "simd_lt" => 0x11,
            "simd_le" => 0x12,
            "simd_ge" => 0x1D,
            "simd_gt" => 0x1E,
            _ => unreachable!(),
        };
        let intrinsic = simple_intrinsic("cmp", vec_ty.scalar, scalar_bits, ty_bits);
        let cast = cast_ident(ScalarType::Float, ScalarType::Mask, scalar_bits, ty_bits);

        quote! {
            #method_sig {
                unsafe { #cast(#intrinsic::<#order_predicate>(a.into(), b.into())).simd_into(self) }
            }
        }
    } else {
        mk_sse4_2::handle_compare(method_sig, method, vec_ty, scalar_bits, ty_bits, arch)
    }
}
