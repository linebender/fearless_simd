// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::Arch;
use crate::arch::sse4_2::{
    Sse4_2, cvt_intrinsic, extend_intrinsic, op_suffix, pack_intrinsic, set1_intrinsic,
    simple_intrinsic, simple_sign_unaware_intrinsic, unpack_intrinsic,
};
use crate::generic::{generic_combine, generic_op, generic_split};
use crate::ops::{OpSig, TyFlavor, ops_for_type, reinterpret_ty, valid_reinterpret};
use crate::types::{SIMD_TYPES, ScalarType, VecType, type_imports};
use crate::x86_common::make_method;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

#[derive(Clone, Copy)]
pub(crate) struct Level;

impl Level {
    fn name(self) -> &'static str {
        "Sse4_2"
    }

    fn token(self) -> TokenStream {
        let ident = Ident::new(self.name(), Span::call_site());
        quote! { #ident }
    }
}

pub(crate) fn mk_sse4_2_impl() -> TokenStream {
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

        use core::arch::x86_64::*;
        use core::ops::*;
        use crate::{seal::Seal, Level, Simd, SimdFrom, SimdInto};

        #imports

        /// The SIMD token for the "SSE 4.2" level.
        #[derive(Clone, Copy, Debug)]
        pub struct Sse4_2 {
            pub sse4_2: crate::core_arch::x86_64::Sse4_2,
        }

        impl Sse4_2 {
            /// Create a SIMD token.
            ///
            /// # Safety
            ///
            /// The SSE4.2 CPU feature must be available.
            #[inline]
            pub unsafe fn new_unchecked() -> Self {
                Sse4_2 {
                    sse4_2: unsafe { crate::core_arch::x86_64::Sse4_2::new_unchecked() },
                }
            }
        }

        impl Seal for Sse4_2 {}

        #simd_impl

        #ty_impl
    }
}

fn mk_simd_impl() -> TokenStream {
    let level_tok = Level.token();
    let mut methods = vec![];
    for vec_ty in SIMD_TYPES {
        for (method, sig) in ops_for_type(vec_ty, true) {
            let b1 = (vec_ty.n_bits() > 128 && !matches!(method, "split" | "narrow"))
                || vec_ty.n_bits() > 256;

            let b2 = !matches!(method, "load_interleaved_128")
                && !matches!(method, "store_interleaved_128");

            if b1 && b2 {
                methods.push(generic_op(method, sig, vec_ty));
                continue;
            }

            let method = make_method(method, sig, vec_ty, Sse4_2, 128);

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
                #[target_feature(enable = "sse4.2")]
                #[inline]
                unsafe fn vectorize_sse4_2<F: FnOnce() -> R, R>(f: F) -> R {
                    f()
                }
                unsafe { vectorize_sse4_2(f) }
            }

            #( #methods )*
        }
    }
}

fn mk_type_impl() -> TokenStream {
    let mut result = vec![];
    for ty in SIMD_TYPES {
        let n_bits = ty.n_bits();
        if n_bits != 128 {
            continue;
        }
        let simd = ty.rust();
        let arch = Sse4_2.arch_ty(ty);
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
