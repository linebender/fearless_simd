// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    ops::{Op, TyFlavor, ops_for_type, overloaded_ops_for, vec_trait_ops_for},
    types::{SIMD_TYPES, ScalarType, type_imports},
};

pub(crate) fn mk_simd_trait() -> TokenStream {
    let imports = type_imports();
    let mut methods = vec![];
    // Float methods
    for vec_ty in SIMD_TYPES {
        let ty_name = vec_ty.rust_name();
        for Op {
            method, sig, doc, ..
        } in &ops_for_type(vec_ty)
        {
            let method_name = format!("{method}_{ty_name}");
            let method_sig = sig.simd_trait_method_sig(vec_ty, &method_name);
            let doc = sig.format_docstring(doc, TyFlavor::SimdTrait);
            methods.extend(quote! {
                #[doc = #doc]
                #method_sig;
            });
        }
    }
    let mut code = quote! {
        use crate::{seal::Seal, Level, SimdElement, SimdFrom, SimdInto, SimdCvtTruncate, SimdCvtFloat, Select, Bytes};
        #imports
        /// The main SIMD trait, implemented by all SIMD token types.
        ///
        /// Each implementor of this trait (e.g. `Avx2`, `Sse4_2`, `Neon`, `Fallback`) is a zero-sized "token" type
        /// representing a specific SIMD instruction set. These tokens are obtained at runtime via [`Level`] and the
        /// [`dispatch!`](crate::dispatch) macro, which selects the best available backend for the current CPU.
        ///
        /// This trait defines all the low-level SIMD operations (e.g. [`add_f32x4`](Simd::add_f32x4),
        /// [`mul_u32x4`](Simd::mul_u32x4)) that are implemented by each token type using platform-specific intrinsics.
        /// However, you typically won't call these methods directly. Instead, you'll probably be using the methods
        /// defined on the vector types themselves.
        ///
        /// # Associated Types
        ///
        /// The trait defines associated types for the highest "native" vector width of each scalar type (e.g. `f32s`,
        /// `u32s`). These are always at least 128 bits, but may be larger. Currently, they are 128 bits everywhere but
        /// AVX2, where they are 256 bits.
        ///
        /// # Example
        ///
        /// ```
        /// # use fearless_simd::{prelude::*, f32x4, dispatch, Level};
        ///
        /// #[inline(always)]
        /// fn add_vectors<S: Simd>(simd: S, a: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
        ///     a + b  // Uses operator overloading, which calls simd.add_f32x4 internally
        /// }
        ///
        /// let level = Level::new();
        /// dispatch!(level, simd => {
        ///     let a = [1.0, 2.0, 3.0, 4.0].simd_into(simd);
        ///     let b = [5.0, 6.0, 7.0, 8.0].simd_into(simd);
        ///     let result = add_vectors(simd, a, b);
        ///     # assert_eq!(result.val, [6.0, 8.0, 10.0, 12.0]);
        /// });
        /// ```
        pub trait Simd: Sized + Clone + Copy + Send + Sync + Seal + 'static {
            /// A native-width SIMD vector of [`f32`]s.
            type f32s: SimdFloat<f32, Self, Block = f32x4<Self>, Mask = Self::mask32s, Bytes = <Self::u32s as Bytes>::Bytes> + SimdCvtFloat<Self::u32s> + SimdCvtFloat<Self::i32s>;
            /// A native-width SIMD vector of [`f64`]s.
            type f64s: SimdFloat<f64, Self, Block = f64x2<Self>, Mask = Self::mask64s>;
            /// A native-width SIMD vector of [`u8`]s.
            type u8s: SimdInt<u8, Self, Block = u8x16<Self>, Mask = Self::mask8s>;
            /// A native-width SIMD vector of [`i8`]s.
            type i8s: SimdInt<i8, Self, Block = i8x16<Self>, Mask = Self::mask8s, Bytes = <Self::u8s as Bytes>::Bytes> + core::ops::Neg<Output = Self::i8s>;
            /// A native-width SIMD vector of [`u16`]s.
            type u16s: SimdInt<u16, Self, Block = u16x8<Self>, Mask = Self::mask16s>;
            /// A native-width SIMD vector of [`i16`]s.
            type i16s: SimdInt<i16, Self, Block = i16x8<Self>, Mask = Self::mask16s, Bytes = <Self::u16s as Bytes>::Bytes> + core::ops::Neg<Output = Self::i16s>;
            /// A native-width SIMD vector of [`u32`]s.
            type u32s: SimdInt<u32, Self, Block = u32x4<Self>, Mask = Self::mask32s> + SimdCvtTruncate<Self::f32s>;
            /// A native-width SIMD vector of [`i32`]s.
            type i32s: SimdInt<i32, Self, Block = i32x4<Self>, Mask = Self::mask32s, Bytes = <Self::u32s as Bytes>::Bytes> + SimdCvtTruncate<Self::f32s>
                + core::ops::Neg<Output = Self::i32s>;
            /// A native-width SIMD mask with 8-bit lanes.
            type mask8s: SimdMask<i8, Self, Block = mask8x16<Self>, Bytes = <Self::u8s as Bytes>::Bytes> + Select<Self::u8s> + Select<Self::i8s> + Select<Self::mask8s>;
            /// A native-width SIMD mask with 16-bit lanes.
            type mask16s: SimdMask<i16, Self, Block = mask16x8<Self>, Bytes = <Self::u16s as Bytes>::Bytes> + Select<Self::u16s> + Select<Self::i16s> + Select<Self::mask16s>;
            /// A native-width SIMD mask with 32-bit lanes.
            type mask32s: SimdMask<i32, Self, Block = mask32x4<Self>, Bytes = <Self::u32s as Bytes>::Bytes>
                + Select<Self::f32s> + Select<Self::u32s> + Select<Self::i32s> + Select<Self::mask32s>;
            /// A native-width SIMD mask with 64-bit lanes.
            type mask64s: SimdMask<i64, Self, Block = mask64x2<Self>> + Select<Self::f64s> + Select<Self::mask64s>;

            /// This SIMD token's feature level.
            fn level(self) -> Level;

            /// Call function with CPU features enabled.
            ///
            /// For performance, the provided function should be `#[inline(always)]`.
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R;
            #( #methods )*
        }
    };
    code.extend(mk_simd_base());
    code.extend(mk_simd_float());
    code.extend(mk_simd_int());
    code.extend(mk_simd_mask());
    code
}

fn mk_simd_base() -> TokenStream {
    quote! {
        /// Base functionality implemented by all SIMD vectors.
        pub trait SimdBase<Element: SimdElement, S: Simd>:
            Copy + Sync + Send + 'static
            + crate::Bytes + SimdFrom<Element, S>
            + core::ops::Index<usize, Output = Element> + core::ops::IndexMut<usize, Output = Element>
        {
            /// This vector type's lane count. This is useful when you're
            /// working with a native-width vector (e.g. [`Simd::f32s`]) and
            /// want to process data in native-width chunks.
            const N: usize;
            /// A SIMD vector mask with the same number of elements.
            ///
            /// The mask element is represented as an integer which is
            /// all-0 for `false` and all-1 for `true`. When we get deep
            /// into AVX-512, we need to think about predication masks.
            ///
            /// One possibility to consider is that the SIMD trait grows
            /// `maskAxB` associated types.
            type Mask: SimdMask<Element::Mask, S>;
            /// A 128-bit SIMD vector of the same scalar type.
            type Block: SimdBase<Element, S>;
            /// Get the [`Simd`] implementation associated with this type.
            fn witness(&self) -> S;
            fn as_slice(&self) -> &[Element];
            fn as_mut_slice(&mut self) -> &mut [Element];
            /// Create a SIMD vector from a slice.
            ///
            /// The slice must be the proper width.
            fn from_slice(simd: S, slice: &[Element]) -> Self;
            /// Create a SIMD vector with all elements set to the given value.
            fn splat(simd: S, val: Element) -> Self;
            /// Create a SIMD vector from a 128-bit vector of the same scalar
            /// type, repeated.
            fn block_splat(block: Self::Block) -> Self;
            /// Create a SIMD vector where each element is produced by
            /// calling `f` with that element's lane index (from 0 to
            /// [`SimdBase::N`] - 1).
            fn from_fn(simd: S, f: impl FnMut(usize) -> Element) -> Self;
        }
    }
}

fn mk_simd_float() -> TokenStream {
    let methods = methods_for_vec_trait(ScalarType::Float);
    let overloaded_ops = overloaded_ops_for(ScalarType::Float);
    let op_traits = overloaded_ops
        .iter()
        .flat_map(|(op, _, _)| op.trait_bounds());
    quote! {
        /// Functionality implemented by floating-point SIMD vectors.
        pub trait SimdFloat<Element: SimdElement, S: Simd>: SimdBase<Element, S>
            #(+ #op_traits)*
        {
            #[inline(always)]
            fn to_int<T: SimdCvtTruncate<Self>>(self) -> T { T::truncate_from(self) }

            #[inline(always)]
            fn to_int_precise<T: SimdCvtTruncate<Self>>(self) -> T { T::truncate_from_precise(self) }

            #( #methods )*
        }
    }
}

fn mk_simd_int() -> TokenStream {
    let methods = methods_for_vec_trait(ScalarType::Unsigned);
    let overloaded_ops = overloaded_ops_for(ScalarType::Unsigned);
    let op_traits = overloaded_ops
        .iter()
        .flat_map(|(op, _, _)| op.trait_bounds());
    quote! {
        /// Functionality implemented by (signed and unsigned) integer SIMD vectors.
        pub trait SimdInt<Element: SimdElement, S: Simd>: SimdBase<Element, S>
            #(+ #op_traits)*
        {
            #[inline(always)]
            fn to_float<T: SimdCvtFloat<Self>>(self) -> T { T::float_from(self) }

            #( #methods )*
        }
    }
}

fn mk_simd_mask() -> TokenStream {
    let methods = methods_for_vec_trait(ScalarType::Mask);
    let overloaded_ops = overloaded_ops_for(ScalarType::Mask);
    let op_traits = overloaded_ops
        .iter()
        .flat_map(|(op, _, _)| op.trait_bounds());
    quote! {
        /// Functionality implemented by SIMD masks.
        pub trait SimdMask<Element: SimdElement, S: Simd>: SimdBase<Element, S>
            #(+ #op_traits)*
        {
            #( #methods )*
        }
    }
}

fn methods_for_vec_trait(scalar: ScalarType) -> Vec<TokenStream> {
    let mut methods = vec![];
    for Op {
        method, sig, doc, ..
    } in vec_trait_ops_for(scalar)
    {
        let doc = sig.format_docstring(doc, TyFlavor::VecImpl);
        if let Some(method_sig) = sig.vec_trait_method_sig(method) {
            methods.push(quote! {
                #[doc = #doc]
                #method_sig;
            });
        }
    }
    methods
}
