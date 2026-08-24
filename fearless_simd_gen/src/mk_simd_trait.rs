// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{format_ident, quote};

use crate::{
    generic::{byte_swizzle_op, generic_op, reversed_compare_op},
    ops::{
        CoreOpTrait, OpKind, OpSig, TyFlavor, base_trait_ops, ops_for_type, overloaded_ops_for,
        vec_trait_ops_for,
    },
    types::{SIMD_TYPES, ScalarType, type_imports},
};

pub(crate) fn mk_simd_trait() -> TokenStream {
    let imports = type_imports();
    let mut methods = vec![];
    // Float methods
    for vec_ty in SIMD_TYPES {
        for op in ops_for_type(vec_ty) {
            let method_sig = op.simd_trait_method_sig(vec_ty);
            let doc = op.format_docstring(TyFlavor::SimdTrait);
            let doc_alias = op
                .doc_alias()
                .map(|alias| quote! { #[doc(alias = #alias)] });
            if op.sig.should_route_swizzle_through_bytes(vec_ty) {
                let method = byte_swizzle_op(&op, vec_ty);
                methods.extend(quote! {
                    #[doc = #doc]
                    #doc_alias
                    #[inline(always)]
                    #method
                });
            } else if let Some(method) = reversed_compare_op(&op, vec_ty) {
                methods.extend(quote! {
                    #[doc = #doc]
                    #doc_alias
                    #[inline(always)]
                    #method
                });
            } else if op.sig.should_use_generic_op(vec_ty, 128) {
                let method = generic_op(&op, vec_ty);
                methods.extend(quote! {
                    #[doc = #doc]
                    #doc_alias
                    #[inline(always)]
                    #method
                });
            } else {
                methods.extend(quote! {
                    #[doc = #doc]
                    #doc_alias
                    #method_sig;
                });
            }
        }
    }
    let mut code = quote! {
        use core::fmt::Debug;
        use crate::{seal::Seal, Level, SimdElement, SimdIntElement, SimdFloatElement, SimdFrom, SimdInto, SimdCvtTruncate, SimdCvtFloat, SimdWiden, SimdNarrow, Select, Bytes};
        #imports
        /// The main SIMD trait, implemented by all SIMD token types.
        ///
        /// Each implementor of this trait (e.g. `Avx2`, `Sse4_2`, `Sse2`, `Neon`, `Fallback`) is a zero-sized "token" type
        /// representing a specific SIMD instruction set. These tokens are obtained at runtime via [`Level`] and the
        /// [`dispatch!`](crate::dispatch) macro, which selects the best available backend for the current CPU.
        ///
        /// This trait defines all the low-level SIMD operations (e.g. [`add_f32x4`](Simd::add_f32x4),
        /// [`mul_u32x4`](Simd::mul_u32x4)) that are implemented by each token type using platform-specific intrinsics.
        /// However, you typically won't call these methods directly. Instead, you'll probably be using the methods
        /// defined on the vector types themselves, such as [`f32x4`] or [`u32x4`].
        ///
        /// # Associated Types
        ///
        /// The trait defines associated types for the highest "native" vector width of each scalar type (e.g. `f32s`,
        /// `u32s`). These are always at least 128 bits, but may be larger. Currently, they are 128 bits on the
        /// fallback, NEON, WASM, SSE2, and SSE4.2 backends, 256 bits on AVX2, and 512 bits on AVX-512.
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
        ///     # assert_eq!(*result, [6.0, 8.0, 10.0, 12.0]);
        /// });
        /// ```
        pub trait Simd: Sized + Clone + Copy + Send + Sync + Debug + Seal + arch_types::ArchTypes + 'static {
            /// A native-width SIMD vector of [`f32`]s.
            type f32s: SimdFloat<Self, Element = f32, Block = f32x4<Self>, Mask = Self::mask32s, ByteVector = Self::u8s> + SimdCvtFloat<Self::u32s> + SimdCvtFloat<Self::i32s>
                + SimdWiden<Self, Widened = Self::f64s>;
            /// A native-width SIMD vector of [`f64`]s.
            type f64s: SimdFloat<Self, Element = f64, Block = f64x2<Self>, Mask = Self::mask64s, ByteVector = Self::u8s>
                + SimdCvtFloat<Self::u64s> + SimdCvtFloat<Self::i64s>
                + SimdNarrow<Self, Narrowed = Self::f32s>;
            /// A native-width SIMD vector of [`u8`]s.
            type u8s: SimdInt<Self, Element = u8, Block = u8x16<Self>, Mask = Self::mask8s, ByteVector = Self::u8s>
                + SimdWiden<Self, Widened = Self::u16s>;
            /// A native-width SIMD vector of [`i8`]s.
            type i8s: SimdInt<Self, Element = i8, Block = i8x16<Self>, Mask = Self::mask8s, ByteVector = Self::u8s>
                + SimdWiden<Self, Widened = Self::i16s> + core::ops::Neg<Output = Self::i8s>;
            /// A native-width SIMD vector of [`u16`]s.
            type u16s: SimdInt<Self, Element = u16, Block = u16x8<Self>, Mask = Self::mask16s, ByteVector = Self::u8s>
                + SimdNarrow<Self, Narrowed = Self::u8s> + SimdWiden<Self, Widened = Self::u32s>;
            /// A native-width SIMD vector of [`i16`]s.
            type i16s: SimdInt<Self, Element = i16, Block = i16x8<Self>, Mask = Self::mask16s, ByteVector = Self::u8s>
                + SimdNarrow<Self, Narrowed = Self::i8s> + SimdWiden<Self, Widened = Self::i32s>
                + core::ops::Neg<Output = Self::i16s>;
            /// A native-width SIMD vector of [`u32`]s.
            type u32s: SimdInt<Self, Element = u32, Block = u32x4<Self>, Mask = Self::mask32s, ByteVector = Self::u8s>
                + SimdCvtTruncate<Self::f32s> + SimdNarrow<Self, Narrowed = Self::u16s>
                + SimdWiden<Self, Widened = Self::u64s>;
            /// A native-width SIMD vector of [`i32`]s.
            type i32s: SimdInt<Self, Element = i32, Block = i32x4<Self>, Mask = Self::mask32s, ByteVector = Self::u8s> + SimdCvtTruncate<Self::f32s>
                + SimdNarrow<Self, Narrowed = Self::i16s> + SimdWiden<Self, Widened = Self::i64s>
                + core::ops::Neg<Output = Self::i32s>;
            /// A native-width SIMD vector of [`u64`]s.
            type u64s: SimdInt<Self, Element = u64, Block = u64x2<Self>, Mask = Self::mask64s, ByteVector = Self::u8s>
                + SimdCvtTruncate<Self::f64s>
                + SimdNarrow<Self, Narrowed = Self::u32s>;
            /// A native-width SIMD vector of [`i64`]s.
            type i64s: SimdInt<Self, Element = i64, Block = i64x2<Self>, Mask = Self::mask64s, ByteVector = Self::u8s>
                + SimdCvtTruncate<Self::f64s>
                + SimdNarrow<Self, Narrowed = Self::i32s>
                + core::ops::Neg<Output = Self::i64s>;
            /// A native-width SIMD mask with 8-bit lanes.
            type mask8s: SimdMask<Self, Element = i8> + Select<Self::u8s> + Select<Self::i8s> + Select<Self::mask8s>;
            /// A native-width SIMD mask with 16-bit lanes.
            type mask16s: SimdMask<Self, Element = i16> + Select<Self::u16s> + Select<Self::i16s> + Select<Self::mask16s>;
            /// A native-width SIMD mask with 32-bit lanes.
            type mask32s: SimdMask<Self, Element = i32> + Select<Self::f32s> + Select<Self::u32s> + Select<Self::i32s> + Select<Self::mask32s>;
            /// A native-width SIMD mask with 64-bit lanes.
            type mask64s: SimdMask<Self, Element = i64> + Select<Self::f64s> + Select<Self::u64s> + Select<Self::i64s> + Select<Self::mask64s>;

            /// This SIMD token's feature level.
            fn level(self) -> Level;

            /// Call function with SIMD instructions enabled, without forcing [inlining](https://matklad.github.io/2021/07/09/inline-in-rust.html).
            ///
            /// `vectorize()` will set the correct `#[target_feature]` annotations for the SIMD level.
            /// The provided function should be `#[inline(always)]`, otherwise it may not
            /// be able to utilize the best SIMD instructions available.
            /// `vectorize()` itself acts as the function boundary in machine code.
            ///
            /// This is useful when the SIMD implementation has already been selected and you want
            /// to keep a SIMD-generic function outlined instead of forcing the entire function to
            /// be inlined into its caller.
            ///
            /// # Example
            ///
            /// `double_u32s` is deliberately not marked `#[inline(always)]`. Instead, only its
            /// closure is inlined into the target-feature-enabled boundary created by `vectorize()`.
            ///
            /// ```
            /// use fearless_simd::{dispatch, prelude::*, Level};
            ///
            /// fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
            ///     simd.vectorize(
            ///         #[inline(always)]
            ///         || {
            ///             let mut chunks = values.chunks_exact_mut(S::u32s::N);
            ///             for chunk in &mut chunks {
            ///                 let value = S::u32s::from_slice(simd, chunk);
            ///                 (value * 2).store_slice(chunk);
            ///             }
            ///             for value in chunks.into_remainder() {
            ///                 *value *= 2;
            ///             }
            ///         },
            ///     );
            /// }
            ///
            /// let mut values = [1, 2, 3, 4, 5];
            /// let level = Level::new();
            /// dispatch!(level, simd => double_u32s(simd, &mut values));
            /// assert_eq!(values, [2, 4, 6, 8, 10]);
            /// ```
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R;

            /// Call function with SIMD instructions enabled, without forcing [inlining](https://matklad.github.io/2021/07/09/inline-in-rust.html).
            ///
            /// `vectorize()` will set the correct `#[target_feature]` annotations for the SIMD level.
            /// The provided function should be `#[inline(always)]`, otherwise it may not
            /// be able to utilize the best SIMD instructions available.
            /// `vectorize()` itself acts as the function boundary in machine code.
            ///
            /// This is useful when the SIMD implementation has already been selected and you want
            /// to keep a SIMD-generic function outlined instead of forcing the entire function to
            /// be inlined into its caller.
            ///
            /// # Example
            ///
            /// `double_u32s` is deliberately not marked `#[inline(always)]`. Instead, only its
            /// closure is inlined into the target-feature-enabled boundary created by `vectorize()`.
            ///
            /// ```
            /// use fearless_simd::{dispatch, prelude::*, Level};
            ///
            /// fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
            ///     simd.vectorize(
            ///         #[inline(always)]
            ///         || {
            ///             let mut chunks = values.chunks_exact_mut(S::u32s::N);
            ///             for chunk in &mut chunks {
            ///                 let value = S::u32s::from_slice(simd, chunk);
            ///                 (value * 2).store_slice(chunk);
            ///             }
            ///             for value in chunks.into_remainder() {
            ///                 *value *= 2;
            ///             }
            ///         },
            ///     );
            /// }
            ///
            /// let mut values = [1, 2, 3, 4, 5];
            /// let level = Level::new();
            /// dispatch!(level, simd => double_u32s(simd, &mut values));
            /// assert_eq!(values, [2, 4, 6, 8, 10]);
            /// ```
            fn vectorize_inline<F: FnOnce() -> R, R>(self, f: F) -> R;
            #( #methods )*
        }
    };
    code.extend(mk_arch_types());
    code.extend(mk_simd_base());
    code.extend(mk_simd_float());
    code.extend(mk_simd_int());
    code.extend(mk_simd_mask());
    code
}

pub(crate) fn mk_arch_types() -> TokenStream {
    let mut types = vec![];
    let mut mask_array_methods = vec![];
    for vec_ty in SIMD_TYPES {
        let ty_name = vec_ty.rust();
        types.push(quote! {
            type #ty_name: Copy + Send + Sync + SimdPod;
        });
        if vec_ty.scalar == ScalarType::Mask {
            let from_array = format_ident!("{}_from_array", vec_ty.rust_name());
            let to_array = format_ident!("{}_to_array", vec_ty.rust_name());
            let scalar = ScalarType::Int.rust(vec_ty.scalar_bits);
            let len = vec_ty.len;
            mask_array_methods.push(quote! {
                #[inline(always)]
                fn #from_array(self, val: [#scalar; #len]) -> Self::#ty_name {
                    crate::transmute::checked_transmute_copy(&val)
                }

                #[inline(always)]
                fn #to_array(self, val: Self::#ty_name) -> [#scalar; #len] {
                    crate::transmute::checked_transmute_copy(&val)
                }
            });
        }
    }

    quote! {
        pub(crate) mod arch_types {
            use crate::transmute::SimdPod;

            #[expect(
                unnameable_types,
                reason = "The native vector types that back a `Simd` implementation are an internal implementation detail, and intentionally kept private"
            )]
            pub trait ArchTypes: Sized {
                #( #types )*
                #( #mask_array_methods )*
            }
        }
    }
}

fn mk_simd_base() -> TokenStream {
    let mut methods = vec![];
    for op in base_trait_ops() {
        let doc = op.format_docstring(TyFlavor::VecImpl);
        if let Some(method_sig) = op.vec_trait_method_sig() {
            methods.push(quote! {
                #[doc = #doc]
                #method_sig;
            });
        }
    }
    let overloaded_ops = [CoreOpTrait::Add, CoreOpTrait::Sub, CoreOpTrait::Mul];
    let op_traits = overloaded_ops
        .iter()
        .flat_map(|core_op| core_op.trait_bounds());
    let max_lanes = SIMD_TYPES.iter().map(|ty| ty.len).max().unwrap();
    let rotate_left_arms = (0..max_lanes).map(|shift| {
        let shift = Literal::usize_unsuffixed(shift);
        quote! { #shift => self.slide::<#shift>(self) }
    });
    let rotate_right_arms = (1..=max_lanes).map(|shift| {
        let shift = Literal::usize_unsuffixed(shift);
        quote! { #shift => self.slide::<#shift>(self) }
    });
    let shift_left_arms = (0..=max_lanes).map(|shift| {
        let shift = Literal::usize_unsuffixed(shift);
        quote! { #shift => self.slide::<#shift>(padding) }
    });
    let shift_right_arms = (0..=max_lanes).map(|shift| {
        let shift = Literal::usize_unsuffixed(shift);
        quote! { #shift => padding.slide::<#shift>(self) }
    });

    quote! {
        /// Base functionality implemented by all SIMD vectors.
        pub trait SimdBase<S: Simd>:
            Copy + Sync + Send + Debug + 'static
            + Seal
            + Bytes<Bytes = Self::ByteVector> + SimdFrom<Self::Element, S> + SimdFrom<Self::Array, S>
            + core::ops::Index<usize, Output = Self::Element> + core::ops::IndexMut<usize, Output = Self::Element>
            + core::ops::Deref<Target = Self::Array>+ core::ops::DerefMut<Target = Self::Array>
            #(+ #op_traits)*
        {
            /// The type of this vector's elements.
            type Element: SimdElement;
            /// The same-width SIMD vector of `u8` lanes used as the byte representation.
            ///
            /// This is the same type as [`Bytes::Bytes`].
            ///
            /// This associated type exists because expressing the `SimdBase` bound directly on
            /// [`Bytes::Bytes`] creates a trait-solver cycle. The `Bytes<Bytes =
            /// Self::ByteVector>` supertrait bound ensures that the two types are identical.
            /// Generic callers should normally use [`Bytes::Bytes`], not this associated type.
            type ByteVector: SimdBase<S, Element = u8, ByteVector = Self::ByteVector>;
            /// This vector type's lane count. This is useful when you're
            /// working with a native-width vector (e.g. [`Simd::f32s`]) and
            /// want to process data in native-width chunks.
            const N: usize;
            /// A SIMD vector mask with the same number of logical lanes.
            ///
            /// Masks intentionally do not implement [`SimdBase`]. SSE, NEON, WASM, and the
            /// fallback backend currently store masks as all-zero/all-one integer vectors, but
            /// AVX-512/RVV/SVE-style targets use compact predicate registers instead.
            type Mask: SimdMask<S, Element = <Self::Element as SimdElement>::Mask> + Select<Self>;
            /// A 128-bit SIMD vector of the same scalar type.
            type Block: SimdBase<S, Element = Self::Element, Block = Self::Block>;
            /// The array type that this vector type corresponds to, which will
            /// always be `[Self::Element; Self::N]`. It has the same layout as
            /// this vector type, but likely has a lower alignment.
            type Array: Copy
                + Debug
                + IntoIterator<Item = Self::Element>
                + AsRef<[Self::Element]>
                + AsMut<[Self::Element]>
                + From<Self>;
            /// Get the [`Simd`] implementation associated with this type.
            fn witness(&self) -> S;
            fn as_slice(&self) -> &[Self::Element];
            fn as_mut_slice(&mut self) -> &mut [Self::Element];
            /// Create a SIMD vector from a slice.
            ///
            /// The slice must be exactly the size of the SIMD vector.
            fn from_slice(simd: S, slice: &[Self::Element]) -> Self;
            /// Store a SIMD vector into a slice.
            ///
            /// The slice must be exactly the size of the SIMD vector.
            fn store_slice(&self, slice: &mut [Self::Element]);
            /// Create a SIMD vector from its corresponding lane array.
            #[inline(always)]
            fn load_array(simd: S, val: Self::Array) -> Self {
                Self::simd_from(simd, val)
            }
            /// Create a SIMD vector from a reference to its corresponding lane array.
            #[inline(always)]
            fn load_array_ref(simd: S, val: &Self::Array) -> Self {
                Self::load_array(simd, *val)
            }
            /// Convert this SIMD vector to its corresponding lane array.
            #[inline(always)]
            fn to_array(self) -> Self::Array {
                *self
            }
            /// Project this SIMD vector reference to its corresponding lane array reference.
            #[inline(always)]
            fn as_array(&self) -> &Self::Array {
                self
            }
            /// Project this mutable SIMD vector reference to its corresponding mutable lane array reference.
            #[inline(always)]
            fn as_mut_array(&mut self) -> &mut Self::Array {
                self
            }
            /// Store this SIMD vector into its corresponding lane array.
            #[inline(always)]
            fn store_array(self, dest: &mut Self::Array) {
                *dest = self.to_array();
            }
            /// Create a SIMD vector from a 128-bit vector of the same scalar
            /// type, repeated.
            fn block_splat(block: Self::Block) -> Self;
            /// Create a SIMD vector where each element is produced by
            /// calling `f` with that element's lane index (from 0 to
            /// [`SimdBase::N`] - 1).
            fn from_fn(simd: S, f: impl FnMut(usize) -> Self::Element) -> Self;

            /// Rotate the vector elements to the left by `OFFSET`.
            ///
            /// If `OFFSET` is greater than or equal to `Self::N`, it wraps modulo `Self::N`.
            #[inline(always)]
            fn rotate_elements_left<const OFFSET: usize>(self) -> Self {
                match OFFSET % Self::N {
                    #(#rotate_left_arms,)*
                    _ => unreachable!(),
                }
            }

            /// Rotate the vector elements to the right by `OFFSET`.
            ///
            /// If `OFFSET` is greater than or equal to `Self::N`, it wraps modulo `Self::N`.
            #[inline(always)]
            fn rotate_elements_right<const OFFSET: usize>(self) -> Self {
                match Self::N - OFFSET % Self::N {
                    #(#rotate_right_arms,)*
                    _ => unreachable!(),
                }
            }

            /// Shift the vector elements to the left by `OFFSET`, filling in with `padding` from the right.
            ///
            /// If `OFFSET` is greater than or equal to `Self::N`, all lanes are filled with `padding`.
            #[inline(always)]
            fn shift_elements_left<const OFFSET: usize>(self, padding: Self::Element) -> Self {
                match OFFSET.min(Self::N) {
                    #(#shift_left_arms,)*
                    _ => unreachable!(),
                }
            }

            /// Shift the vector elements to the right by `OFFSET`, filling in with `padding` from the left.
            ///
            /// If `OFFSET` is greater than or equal to `Self::N`, all lanes are filled with `padding`.
            #[inline(always)]
            fn shift_elements_right<const OFFSET: usize>(self, padding: Self::Element) -> Self {
                let padding = Self::splat(self.witness(), padding);
                match Self::N.saturating_sub(OFFSET) {
                    #(#shift_right_arms,)*
                    _ => unreachable!(),
                }
            }

            #( #methods )*
        }
    }
}

fn mk_simd_float() -> TokenStream {
    let methods = methods_for_vec_trait(ScalarType::Float);
    let overloaded_ops = overloaded_ops_for(ScalarType::Float);
    let op_traits = overloaded_ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::Overloaded(core_op) => Some(core_op),
            _ => None,
        })
        .filter(|core_op| !is_base_arithmetic(core_op))
        .flat_map(|core_op| core_op.trait_bounds());
    quote! {
        /// Functionality implemented by floating-point SIMD vectors.
        pub trait SimdFloat<S: Simd>: SimdBase<S, Element: SimdFloatElement> + Seal
            #(+ #op_traits)*
        {
            /// Convert this floating-point type to an integer. This is a convenience method that
            /// delegates to [`SimdCvtTruncate::truncate_from`], and can only be called if there
            /// actually exists a target type of the same bit width (`u32`/`i32` for `f32`, or
            /// `u64`/`i64` for `f64`).
            ///
            /// For more information about the semantics of this specific conversion, see the
            /// concrete `SimdCvtTruncate` implementations for integer types.
            #[inline(always)]
            fn to_int<T: SimdCvtTruncate<Self>>(self) -> T { T::truncate_from(self) }

            /// Convert this floating-point type to an integer, saturating on overflow and returning
            /// 0 for NaN. This is a convenience method that delegates to
            /// [`SimdCvtTruncate::truncate_from_precise`], and can only be called if there actually
            /// exists a target type of the same bit width (`u32`/`i32` for `f32`, or `u64`/`i64`
            /// for `f64`).
            ///
            /// For more information about the semantics of this specific conversion, see the
            /// concrete `SimdCvtTruncate` implementations for integer types.
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
        .filter_map(|op| match &op.kind {
            OpKind::Overloaded(core_op) => Some(core_op),
            _ => None,
        })
        .filter(|core_op| !is_base_arithmetic(core_op))
        .flat_map(|core_op| core_op.trait_bounds());
    quote! {
        /// Functionality implemented by (signed and unsigned) integer SIMD vectors.
        pub trait SimdInt<S: Simd>: SimdBase<S, Element: SimdIntElement> + Seal
            #(+ #op_traits)*
        {
            /// Convert this integer type to a floating-point type. This is a convenience method
            /// that delegates to [`SimdCvtFloat::float_from`], and can only be called if there
            /// actually exists a target type of the same bit width (`f32` or `f64`).
            #[inline(always)]
            fn to_float<T: SimdCvtFloat<Self>>(self) -> T { T::float_from(self) }

            #( #methods )*
        }
    }
}

fn is_base_arithmetic(core_op: &CoreOpTrait) -> bool {
    matches!(
        core_op,
        CoreOpTrait::Add | CoreOpTrait::Sub | CoreOpTrait::Mul
    )
}

fn mk_simd_mask() -> TokenStream {
    let methods = methods_for_vec_trait(ScalarType::Mask);
    let overloaded_ops = overloaded_ops_for(ScalarType::Mask);
    let op_traits = overloaded_ops
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::Overloaded(core_op) => Some(core_op),
            _ => None,
        })
        .flat_map(|core_op| {
            let trait_name = Ident::new(core_op.trait_name(), Span::call_site());
            let trait_name_assign = format_ident!("{trait_name}Assign");
            match core_op {
                CoreOpTrait::Not => vec![quote! { core::ops::#trait_name<Output = Self> }],
                _ => vec![
                    quote! { core::ops::#trait_name<Output = Self> },
                    quote! { core::ops::#trait_name_assign },
                ],
            }
        });
    quote! {
        /// Functionality implemented by SIMD masks.
        ///
        /// A mask has one logical boolean lane per SIMD lane. Its storage is intentionally opaque:
        /// current backends may use all-zero/all-one integer vectors internally, while future
        /// predicate-register backends may use a compact representation.
        pub trait SimdMask<S: Simd>:
            Copy + Sync + Send + 'static
            + Seal
            + Select<Self>
            #(+ #op_traits)*
        {
            /// The signed integer type used when converting this mask to and from lane values.
            ///
            /// False lanes are encoded as all zeroes (integer value 0), and true lanes are encoded as all ones
            /// (integer value -1).
            type Element: SimdElement;

            /// This mask type's lane count.
            const N: usize;

            /// Get the [`Simd`] implementation associated with this type.
            fn witness(&self) -> S;

            /// Create a SIMD mask with all lanes set to the given boolean value.
            fn splat(simd: S, val: bool) -> Self;

            /// Create a mask from a compact bitmask.
            ///
            /// Bit `i` maps to lane `i`, with lane 0 in the least significant bit. Bits above
            /// [`Self::N`] are ignored.
            fn from_bitmask(simd: S, bits: u64) -> Self;

            /// Convert this mask to a compact bitmask.
            ///
            /// Bit `i` maps to lane `i`, with lane 0 in the least significant bit. Bits above
            /// [`Self::N`] are cleared.
            fn to_bitmask(self) -> u64;

            /// Test whether one logical lane is set.
            ///
            /// Panics if `index` is greater than or equal to the number of lanes in the mask.
            #[inline(always)]
            fn test(&self, index: usize) -> bool {
                assert!(
                    index < Self::N,
                    "mask lane index {index} is out of bounds for {} lanes",
                    Self::N
                );
                (((*self).to_bitmask() >> index) & 1) != 0
            }

            /// Sets the value of one logical lane.
            ///
            /// Panics if `index` is greater than or equal to the number of lanes in the mask.
            fn set(&mut self, index: usize, value: bool);

            /// Create a SIMD mask from signed integer mask lanes.
            ///
            /// The slice must be exactly the size of the SIMD mask.
            fn from_slice(simd: S, slice: &[Self::Element]) -> Self;

            /// Store this SIMD mask as signed integer mask lanes.
            ///
            /// The slice must be exactly the size of the SIMD mask.
            fn store_slice(&self, slice: &mut [Self::Element]);

            #( #methods )*
        }
    }
}

fn methods_for_vec_trait(scalar: ScalarType) -> Vec<TokenStream> {
    let mut methods = vec![];
    for op in vec_trait_ops_for(scalar) {
        let doc = op.format_docstring(TyFlavor::VecImpl);
        let method_sig = if scalar == ScalarType::Mask && matches!(op.sig, OpSig::Compare) {
            Some(quote! { fn simd_eq(self, rhs: impl SimdInto<Self, S>) -> Self })
        } else {
            op.vec_trait_method_sig()
        };
        if let Some(method_sig) = method_sig {
            methods.push(quote! {
                #[doc = #doc]
                #method_sig;
            });
        }
    }
    methods
}
