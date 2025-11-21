// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span, TokenStream};
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
