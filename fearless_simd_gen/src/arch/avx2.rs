// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::{x86_common, Arch};
use crate::types::{ScalarType, VecType};
use crate::x86_common::{op_suffix, set1_intrinsic, simple_intrinsic};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use crate::arch::x86_common::translate_op;

pub(crate) struct Avx2;

impl Arch for Avx2 {
    fn arch_ty(&self, ty: &VecType) -> TokenStream {
        x86_common::arch_ty(ty)
    }

    fn expr(&self, op: &str, ty: &VecType, args: &[TokenStream]) -> TokenStream {
        x86_common::expr(op, ty, args)
    }
}
