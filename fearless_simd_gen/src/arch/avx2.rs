// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::{Arch, x86_common};
use crate::types::VecType;
use proc_macro2::TokenStream;

pub(crate) struct Avx2;

impl Arch for Avx2 {
    fn arch_ty(&self, ty: &VecType) -> TokenStream {
        x86_common::arch_ty(ty)
    }

    fn expr(&self, op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
        x86_common::expr(op, ty, args)
    }
}
