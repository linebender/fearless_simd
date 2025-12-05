// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::quote;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ScalarType {
    Float,
    Unsigned,
    Int,
    Mask,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct VecType {
    pub scalar: ScalarType,
    pub scalar_bits: usize,
    pub len: usize,
}

impl ScalarType {
    pub(crate) fn prefix(self) -> &'static str {
        match self {
            Self::Float => "f",
            Self::Unsigned => "u",
            Self::Int | Self::Mask => "i",
        }
    }

    pub(crate) fn rust_name(&self, scalar_bits: usize) -> String {
        format!("{}{}", self.prefix(), scalar_bits)
    }

    pub(crate) fn rust(&self, scalar_bits: usize) -> TokenStream {
        let ident = Ident::new(&self.rust_name(scalar_bits), Span::call_site());
        quote! { #ident }
    }
}

impl VecType {
    pub(crate) const fn new(scalar: ScalarType, scalar_bits: usize, len: usize) -> Self {
        Self {
            scalar,
            scalar_bits,
            len,
        }
    }

    pub(crate) fn n_bits(&self) -> usize {
        self.scalar_bits * self.len
    }

    /// Name of the type, as in `f32x4`
    pub(crate) fn rust_name(&self) -> String {
        let scalar = match self.scalar {
            ScalarType::Float => "f",
            ScalarType::Unsigned => "u",
            ScalarType::Int => "i",
            ScalarType::Mask => "mask",
        };
        format!("{}{}x{}", scalar, self.scalar_bits, self.len)
    }

    /// Returns type without the `<S>`.
    pub(crate) fn rust(&self) -> TokenStream {
        let ident = Ident::new(&self.rust_name(), Span::call_site());
        quote! { #ident }
    }

    pub(crate) fn reinterpret(&self, dst_scalar: ScalarType, dst_scalar_bits: usize) -> Self {
        Self::new(dst_scalar, dst_scalar_bits, self.n_bits() / dst_scalar_bits)
    }

    pub(crate) fn widened(&self) -> Option<Self> {
        if matches!(self.scalar, ScalarType::Mask | ScalarType::Float)
            || self.n_bits() > 256
            || self.scalar_bits != 8
        {
            return None;
        }

        let scalar_bits = self.scalar_bits * 2;
        Some(Self::new(self.scalar, scalar_bits, self.len))
    }

    pub(crate) fn narrowed(&self) -> Option<Self> {
        if matches!(self.scalar, ScalarType::Mask | ScalarType::Float)
            || self.n_bits() < 256
            || self.scalar_bits != 16
        {
            return None;
        }

        let scalar_bits = self.scalar_bits / 2;
        Some(Self::new(self.scalar, scalar_bits, self.len))
    }

    pub(crate) fn mask_ty(&self) -> Self {
        Self::new(ScalarType::Mask, self.scalar_bits, self.len)
    }

    pub(crate) fn block_ty(&self) -> Self {
        Self::new(self.scalar, self.scalar_bits, 128 / self.scalar_bits)
    }

    pub(crate) fn split_operand(&self) -> Option<Self> {
        if self.n_bits() <= 128 {
            return None;
        }
        let n2 = self.len / 2;
        Some(Self::new(self.scalar, self.scalar_bits, n2))
    }

    pub(crate) fn combine_operand(&self) -> Option<Self> {
        if self.n_bits() >= 512 {
            return None;
        }
        let n2 = self.len * 2;
        Some(Self::new(self.scalar, self.scalar_bits, n2))
    }

    pub(crate) fn docstring(&self) -> String {
        let len = self.len;
        if self.scalar == ScalarType::Mask {
            let scalar_bits = self.scalar_bits;
            format!(
                "A SIMD mask of {len} {scalar_bits}-bit elements.\n\n\
                When created from a comparison operation, and as it should be used in a [`Self::select`] operation, each element will be all ones if it's \"true\", and all zeroes if it's \"false\".",
            )
        } else {
            let scalar_name = self.scalar.rust_name(self.scalar_bits);
            let block_ty = self.block_ty();
            let rust_name = self.rust_name();

            let (splat_example, many_example_literals): (String, Vec<String>) = match self.scalar {
                ScalarType::Float => {
                    let start = 1.0;
                    let values = (0..self.len)
                        .map(|n| Literal::f64_unsuffixed(n as f64 + start).to_string())
                        .collect();
                    (Literal::f64_unsuffixed(start).to_string(), values)
                }
                ScalarType::Unsigned | ScalarType::Int => {
                    let start = 1;
                    let values = (0..self.len)
                        .map(|n| Literal::usize_unsuffixed(n + start).to_string())
                        .collect();
                    (Literal::usize_unsuffixed(start).to_string(), values)
                }
                ScalarType::Mask => unreachable!(),
            };
            let many_example = many_example_literals.join(", ");

            let block_example = if &block_ty != self {
                let block_example = many_example_literals[0..block_ty.len].join(", ");
                let block_name = block_ty.rust_name();
                format!(
                    "
    # use fearless_simd::{block_name};
    // From `Self::Block`:
    let f = {rust_name}::block_splat({block_name}::simd_from([{block_example}], simd));"
                )
            } else {
                String::new()
            };

            format!(
                "A SIMD vector of {len} [`{scalar_name}`] elements.\n\n\
                You may construct this vector type using the [`Self::splat`], [`Self::from_slice`], [`Self::simd_from`], [`Self::from_fn`], and [`Self::block_splat`] methods.\n\n\
                ```rust\n\
# use fearless_simd::{{prelude::*, {rust_name}}};
fn construct_simd<S: Simd>(simd: S) {{
    // From a single scalar value:
    let a = {rust_name}::splat(simd, {splat_example});
    let b = {rust_name}::simd_from({splat_example}, simd);

    // From a slice:
    let c = {rust_name}::from_slice(simd, &[{many_example}]);

    // From an array:
    let d = {rust_name}::simd_from([{many_example}], simd);

    // From an element-wise function:
    let e = {rust_name}::from_fn(simd, |i| i as {scalar_name});\
    {block_example}
}}
```")
        }
    }
}

pub(crate) const SIMD_TYPES: &[VecType] = &[
    // 128 bit types
    VecType::new(ScalarType::Float, 32, 4),
    VecType::new(ScalarType::Int, 8, 16),
    VecType::new(ScalarType::Unsigned, 8, 16),
    VecType::new(ScalarType::Mask, 8, 16),
    VecType::new(ScalarType::Int, 16, 8),
    VecType::new(ScalarType::Unsigned, 16, 8),
    VecType::new(ScalarType::Mask, 16, 8),
    VecType::new(ScalarType::Int, 32, 4),
    VecType::new(ScalarType::Unsigned, 32, 4),
    VecType::new(ScalarType::Mask, 32, 4),
    VecType::new(ScalarType::Float, 64, 2),
    VecType::new(ScalarType::Mask, 64, 2),
    // 256 bit types
    VecType::new(ScalarType::Float, 32, 8),
    VecType::new(ScalarType::Int, 8, 32),
    VecType::new(ScalarType::Unsigned, 8, 32),
    VecType::new(ScalarType::Mask, 8, 32),
    VecType::new(ScalarType::Int, 16, 16),
    VecType::new(ScalarType::Unsigned, 16, 16),
    VecType::new(ScalarType::Mask, 16, 16),
    VecType::new(ScalarType::Int, 32, 8),
    VecType::new(ScalarType::Unsigned, 32, 8),
    VecType::new(ScalarType::Mask, 32, 8),
    VecType::new(ScalarType::Float, 64, 4),
    VecType::new(ScalarType::Mask, 64, 4),
    // 512 bit types
    VecType::new(ScalarType::Float, 32, 16),
    VecType::new(ScalarType::Int, 8, 64),
    VecType::new(ScalarType::Unsigned, 8, 64),
    VecType::new(ScalarType::Mask, 8, 64),
    VecType::new(ScalarType::Int, 16, 32),
    VecType::new(ScalarType::Unsigned, 16, 32),
    VecType::new(ScalarType::Mask, 16, 32),
    VecType::new(ScalarType::Int, 32, 16),
    VecType::new(ScalarType::Unsigned, 32, 16),
    VecType::new(ScalarType::Mask, 32, 16),
    VecType::new(ScalarType::Float, 64, 8),
    VecType::new(ScalarType::Mask, 64, 8),
];

pub(crate) fn type_imports() -> TokenStream {
    let mut imports = vec![];
    for ty in SIMD_TYPES {
        let ident = ty.rust();
        imports.push(quote! { #ident });
    }
    quote! {
        use crate::{ #( #imports ),* };
    }
}
