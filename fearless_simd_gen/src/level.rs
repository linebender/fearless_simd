// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

use crate::{
    generic::generic_op,
    ops::{Op, ops_for_type},
    types::{SIMD_TYPES, ScalarType, VecType, type_imports},
};

/// Trait implemented by each SIMD level code generator. The methods on top must be provided by each code generator; the
/// others are provided as default trait methods that call into the non-default ones.
pub(crate) trait Level {
    /// The name of this SIMD level token (e.g. `Neon` or `Sse4_2`).
    fn name(&self) -> &'static str;
    /// The highest vector width, in bits, that SIMD instructions can directly operate on. Operations above this width
    /// will be implemented via split/combine.
    fn native_width(&self) -> usize;
    /// The highest bit width available to *store* vector types in. This is usually the same as [`Level::native_width`],
    /// but may differ if the implementation provides wider vector types than most instructions actually operate on. For
    /// instance, NEON provides tuples of vectors like `int32x4x4_t` up to 512 bits, and the fallback implementation
    /// stores everything as arrays but only operates on 128-bit chunks.
    fn max_block_size(&self) -> usize;
    /// The names of the target features to enable within vectorized code. This goes in the
    /// `#[target_feature(enable = "...")]` attribute.
    ///
    /// If this SIMD level is not runtime-toggleable (for instance, the fallback implementation or WASM SIMD128),
    /// returns `None`.
    fn enabled_target_features(&self) -> Option<&'static str>;
    /// A function that takes a given vector type and returns the corresponding native vector type. For instance,
    /// `f32x8` would map to `__m256` on `Avx2`, and to `[f32; 8]` on `Fallback`. This will never be passed a vector
    /// type *larger* than [`Level::max_block_size`], since [`VecType::aligned_wrapper_ty`] will split those up into
    /// smaller blocks.
    fn arch_ty(&self, vec_ty: &VecType) -> TokenStream;
    /// The docstring for this SIMD level token.
    fn token_doc(&self) -> &'static str;
    /// The full path to the `core_arch` token wrapped by this SIMD level token.
    fn token_inner(&self) -> TokenStream;

    /// Whether to generate From/SimdFrom conversions for arch types at this level.
    /// This should return false if another level already provides these conversions
    /// (e.g., AVX-512 with native_width=256 should return false since AVX2 provides them).
    fn generate_arch_conversions(&self) -> bool {
        true
    }

    /// Any additional imports or supporting code necessary for the module (for instance, importing
    /// implementation-specific functions from `core::arch`).
    fn make_module_prelude(&self) -> TokenStream;
    /// The body of the SIMD token's inherent `impl` block. By convention, this contains an unsafe `new_unchecked`
    /// method for constructing a SIMD token that may not be supported on current hardware, or a safe `new` method for
    /// constructing a SIMD token that is statically known to be supported.
    fn make_impl_body(&self) -> TokenStream;
    /// Generate a single operation's method on the `Simd` implementation.
    fn make_method(&self, op: Op, vec_ty: &VecType) -> TokenStream;

    fn token(&self) -> Ident {
        Ident::new(self.name(), Span::call_site())
    }

    fn impl_arch_types(&self) -> TokenStream {
        let mut assoc_types = vec![];
        for vec_ty in SIMD_TYPES {
            let ty_ident = vec_ty.rust();
            let wrapper_ty =
                vec_ty.aligned_wrapper_ty(|vec_ty| self.arch_ty(vec_ty), self.max_block_size());
            assoc_types.push(quote! {
                type #ty_ident = #wrapper_ty;
            });
        }
        let level_tok = self.token();

        quote! {
            impl ArchTypes for #level_tok {
                #( #assoc_types )*
            }
        }
    }

    /// Any additional supporting code necessary for the module's implementation, but placed *after* the `Simd`
    /// implementation itself.
    fn make_module_footer(&self) -> TokenStream {
        TokenStream::new()
    }

    /// The body of the `Simd::level` function. This can be overridden, e.g. to return `Level::baseline()` if we know a
    /// higher SIMD level is statically enabled.
    fn make_level_body(&self) -> TokenStream {
        let level_tok = self.token();

        quote! {
            Level::#level_tok(self)
        }
    }

    fn make_simd_impl(&self) -> TokenStream {
        let level_tok = self.token();
        let native_width = self.native_width();
        let mut methods = vec![];
        for vec_ty in SIMD_TYPES {
            for op in ops_for_type(vec_ty) {
                if op.sig.should_use_generic_op(vec_ty, native_width) {
                    methods.push(generic_op(&op, vec_ty));
                    continue;
                }

                let method = self.make_method(op, vec_ty);
                methods.push(method);
            }
        }

        let vectorize_body = if let Some(target_features) = self.enabled_target_features() {
            let vectorize = format_ident!("vectorize_{}", self.name().to_ascii_lowercase());
            quote! {
                // This function is deliberately not marked #[inline]:
                // The closure passed to it is already required to be #[inline(always)],
                // so this wrapper is the only opportunity for the compiler to make inlining decisions.
                #[target_feature(enable = #target_features)]
                unsafe fn #vectorize<F: FnOnce() -> R, R>(f: F) -> R {
                    f()
                }
                unsafe { #vectorize(f) }
            }
        } else {
            // This SIMD level doesn't do runtime feature detection/enabling, so we could just call the passed closure as-is.
            //
            // But the inner function is required to be annotated `#[inline(always)]`,
            // so we wrap it in a function that isn't `#[inline(always)]`
            // to let the compiler make its own inlining decisions, as opposed to forcing it to inline everything.
            quote! {
                fn vectorize_inner<F: FnOnce() -> R, R>(f: F) -> R {
                    f()
                }
                vectorize_inner(f)
            }
        };

        let level_body = self.make_level_body();

        let mut assoc_types = vec![];
        for (scalar, scalar_bits) in [
            (ScalarType::Float, 32),
            (ScalarType::Float, 64),
            (ScalarType::Unsigned, 8),
            (ScalarType::Int, 8),
            (ScalarType::Unsigned, 16),
            (ScalarType::Int, 16),
            (ScalarType::Unsigned, 32),
            (ScalarType::Int, 32),
            (ScalarType::Mask, 8),
            (ScalarType::Mask, 16),
            (ScalarType::Mask, 32),
            (ScalarType::Mask, 64),
        ] {
            let native_width_ty = VecType::new(scalar, scalar_bits, native_width / scalar_bits);
            let name = native_width_ty.rust();
            let native_width_name = scalar.native_width_name(scalar_bits);
            assoc_types.push(quote! {
                type #native_width_name = #name<Self>;
            });
        }

        quote! {
            impl Simd for #level_tok {
                #( #assoc_types )*

                #[inline(always)]
                fn level(self) -> Level {
                    #level_body
                }

                #[inline]
                fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                    #vectorize_body
                }

                #(
                    #[inline(always)]
                    #methods
                )*
            }
        }
    }

    fn make_type_impl(&self) -> TokenStream {
        // Skip generating conversions if another level already provides them
        if !self.generate_arch_conversions() {
            return quote! {};
        }
        let native_width = self.native_width();
        let max_block_size = self.max_block_size();
        let mut result = vec![];
        for ty in SIMD_TYPES {
            let n_bits = ty.n_bits();
            // If n_bits is below our native width (e.g. 128 bits for AVX2), another module will have already
            // implemented the conversion.
            if n_bits > max_block_size || n_bits < native_width {
                continue;
            }
            let simd = ty.rust();
            let arch = self.arch_ty(ty);
            result.push(quote! {
                impl<S: Simd> SimdFrom<#arch, S> for #simd<S> {
                    #[inline(always)]
                    fn simd_from(simd: S, arch: #arch) -> Self {
                        Self {
                            val: unsafe { core::mem::transmute_copy(&arch) },
                            simd
                        }
                    }
                }
                impl<S: Simd> From<#simd<S>> for #arch {
                    #[inline(always)]
                    fn from(value: #simd<S>) -> Self {
                        unsafe { core::mem::transmute_copy(&value.val) }
                    }
                }
            });
        }
        quote! {
            #( #result )*
        }
    }

    fn make_module(&self) -> TokenStream {
        let level_tok = self.token();
        let token_doc = self.token_doc();
        let field_name = Ident::new(&self.name().to_ascii_lowercase(), Span::call_site());
        let token_inner = self.token_inner();
        let imports = type_imports();
        let module_prelude = self.make_module_prelude();
        let impl_body = self.make_impl_body();
        let arch_types_impl = self.impl_arch_types();
        let simd_impl = self.make_simd_impl();
        let ty_impl = self.make_type_impl();
        let footer = self.make_module_footer();

        quote! {
            use crate::{prelude::*, seal::Seal, arch_types::ArchTypes, Level};

            #imports

            #module_prelude

            #[doc = #token_doc]
            #[derive(Clone, Copy, Debug)]
            pub struct #level_tok {
                pub #field_name: #token_inner,
            }

            impl #level_tok {
                #impl_body
            }

            impl Seal for #level_tok {}

            #arch_types_impl

            #simd_impl

            #ty_impl

            #footer
        }
    }
}
