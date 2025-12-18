// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span};

pub(crate) trait Level {
    fn name(&self) -> &'static str;

    fn token(&self) -> Ident {
        Ident::new(self.name(), Span::call_site())
    }
}
