// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Generate `core_arch` wrapper modules from parsed stdarch intrinsics.

use std::{
    fs::{self, File},
    path::Path,
};

use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, quote};

use crate::util::write_code;

/// Target architecture for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetArch {
    X86,
    Aarch64,
    Wasm32,
}

const ALL_ARCHS: &[TargetArch] = &[TargetArch::X86, TargetArch::Aarch64, TargetArch::Wasm32];

impl TargetArch {
    fn name(self) -> &'static str {
        match self {
            Self::X86 => "x86",
            Self::Aarch64 => "aarch64",
            Self::Wasm32 => "wasm32",
        }
    }

    fn parse_config(self) -> &'static crate::parse_stdarch::ArchConfig {
        match self {
            Self::X86 => &crate::parse_stdarch::X86_CONFIG,
            Self::Aarch64 => &crate::parse_stdarch::AARCH64_CONFIG,
            Self::Wasm32 => &crate::parse_stdarch::WASM32_CONFIG,
        }
    }

    fn feature_module_config(self, feature: &str) -> FeatureModuleConfig {
        FeatureModuleConfig {
            arch: self,
            struct_name: if let Self::Wasm32 = self {
                format!("Wasm{}", feature_to_struct_name(feature))
            } else {
                feature_to_struct_name(feature)
            },
            file_name: feature_to_file_name(feature),
        }
    }

    /// Generate the arch import statements for this target.
    fn gen_arch_imports(self) -> TokenStream {
        match self {
            Self::X86 => quote! {
                #[cfg(target_arch = "x86")]
                use core::arch::x86 as arch;
                #[cfg(target_arch = "x86_64")]
                use core::arch::x86_64 as arch;

                use arch::*;
            },
            Self::Aarch64 => quote! {
                use core::arch::aarch64 as arch;
                use arch::*;
            },
            Self::Wasm32 => quote! {
                use core::arch::wasm32 as arch;
                use arch::*;
            },
        }
    }

    /// Generate the doc string for struct tokens on this architecture.
    fn arch_doc_suffix(self) -> &'static str {
        match self {
            Self::X86 => "`x86` and `x86_64`",
            Self::Aarch64 => "`aarch64`",
            Self::Wasm32 => "`wasm32`",
        }
    }

    /// Generate the constructor for this architecture.
    ///
    /// x86 and aarch64 require runtime feature detection, so the constructor is unsafe.
    /// wasm32 has static feature detection, so the constructor is safe.
    fn gen_constructor(self) -> TokenStream {
        match self {
            Self::X86 | Self::Aarch64 => quote! {
                /// Create a SIMD token.
                ///
                /// # Safety
                ///
                /// The required CPU features must be available.
                #[inline]
                pub const unsafe fn new_unchecked() -> Self {
                    Self { _private: () }
                }
            },
            Self::Wasm32 => quote! {
                /// Create a SIMD token.
                #[inline]
                #[expect(
                    clippy::new_without_default,
                    reason = "other architectures have unsafe `new_unchecked` constructors and cannot implement `Default`; for symmetry, we do not do so either"
                )]
                pub const fn new() -> Self {
                    Self { _private: () }
                }
            },
        }
    }

    fn gen_arch_mod(&self, configs: &[FeatureModuleConfig]) -> TokenStream {
        let arch_doc_suffix = self.arch_doc_suffix();
        let doc = format!("Access to intrinsics on {arch_doc_suffix}.");

        let mod_decls: Vec<TokenStream> = configs
            .iter()
            .map(|config| {
                let mod_name = Ident::new(&config.file_name, Span::call_site());
                quote! { mod #mod_name; }
            })
            .collect();

        let pub_uses: Vec<TokenStream> = configs
            .iter()
            .map(|config| {
                let mod_name = Ident::new(&config.file_name, Span::call_site());
                let struct_name = Ident::new(&config.struct_name, Span::call_site());
                quote! { pub use #mod_name::#struct_name; }
            })
            .collect();

        quote! {
            #![doc = #doc]

            #(#mod_decls)*

            #(#pub_uses)*
        }
    }
}

/// Configuration for generating a target feature module.
#[derive(Debug, Clone)]
pub(crate) struct FeatureModuleConfig {
    /// The target architecture.
    pub arch: TargetArch,
    /// The struct name (e.g., `"Sse4_1"`, `"Neon"`).
    pub struct_name: String,
    /// The module file name (e.g., `"sse4_1"`, `"neon"`).
    pub file_name: String,
}

impl FeatureModuleConfig {
    /// Generate a complete module for a target feature.
    pub(crate) fn gen_feature_module(&self, intrinsics: &[syn::ItemFn]) -> TokenStream {
        let struct_name = Ident::new(&self.struct_name, Span::call_site());
        let struct_name_str = &self.struct_name;
        let arch_doc_suffix = self.arch.arch_doc_suffix();

        // Generate method implementations
        let methods: Vec<TokenStream> = intrinsics.iter().map(gen_method).collect();

        // Architecture-specific imports
        let arch_imports = self.arch.gen_arch_imports();

        // Architecture-specific constructor
        let constructor = self.arch.gen_constructor();

        let doc = format!("A token for `{struct_name_str}` intrinsics on {arch_doc_suffix}.");

        quote! {
            #arch_imports

            #[doc = #doc]
            #[derive(Clone, Copy, Debug)]
            pub struct #struct_name {
                _private: (),
            }

            #[allow(
                clippy::missing_safety_doc,
                reason = "The underlying functions have their own safety docs"
            )]
            impl #struct_name {
                #constructor

                #(#methods)*
            }
        }
    }
}

/// Convert a feature name to a struct name (e.g., `"sse4.1"` -> `"Sse4_1"`).
fn feature_to_struct_name(feature: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;

    for c in feature.chars() {
        if c == '.' || c == '_' || c == '-' {
            result.push('_');
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }

    result
}

/// Convert a feature name to a file name (e.g., `"sse4.1"` -> `"sse4_1"`).
fn feature_to_file_name(feature: &str) -> String {
    feature.replace(['.', '-'], "_")
}

/// Generate a forwarding method for an intrinsic.
fn gen_method(item: &syn::ItemFn) -> TokenStream {
    let forwarding_params: Vec<&Ident> = item
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| match p {
            syn::GenericParam::Lifetime(_) => None,
            syn::GenericParam::Type(type_param) => Some(&type_param.ident),
            syn::GenericParam::Const(const_param) => Some(&const_param.ident),
        })
        .collect();
    let turbofish = if forwarding_params.is_empty() {
        quote! {}
    } else {
        quote! { ::<#(#forwarding_params),*> }
    };

    // Parameters for the call
    let mut param_names = Vec::new();
    for arg in &item.sig.inputs {
        match arg {
            syn::FnArg::Typed(pat) => match &*pat.pat {
                syn::Pat::Ident(syn::PatIdent { ident, .. }) => {
                    param_names.push(ident);
                }
                _ => panic!(
                    "Unsupported non-ident parameter pattern in intrinsic wrapper: {}",
                    pat.pat.to_token_stream()
                ),
            },
            syn::FnArg::Receiver(receiver) => {
                panic!(
                    "Unexpected receiver argument in intrinsic wrapper: {}",
                    receiver.to_token_stream()
                );
            }
        }
    }

    let name = &item.sig.ident;
    let unsafe_mod = &item.sig.unsafety;
    let inputs = &item.sig.inputs;
    let return_type = &item.sig.output;

    // Attributes that should be copied to the generated wrapper
    let extra_attrs = item.attrs.iter().filter(|attr| {
        attr.path().is_ident("expect")
            || attr.path().is_ident("allow")
            || attr.path().is_ident("deprecated")
    });

    // Doc comment linking to the underlying intrinsic
    let doc = format!("See [`arch::{name}`].");

    let (impl_generics, _, where_clause) = item.sig.generics.split_for_impl();
    quote! {
        #[doc = #doc]
        #(#extra_attrs)*
        #[inline(always)]
        pub #unsafe_mod fn #name #impl_generics (self, #inputs) #return_type #where_clause {
            unsafe { #name #turbofish (#(#param_names,)*) }
        }
    }
}

/// Generate all `core_arch` modules for all supported architectures.
pub(crate) fn generate_all_modules(stdarch_root: &Path, output_base: &Path) {
    for &arch in ALL_ARCHS {
        let arch_name = arch.name();
        let intrinsics_by_feature =
            crate::parse_stdarch::parse_arch(stdarch_root, arch.parse_config()).unwrap();

        let output_dir = output_base.join(arch_name);

        // Ensure the output directory exists
        fs::create_dir_all(&output_dir).unwrap_or_else(|_| panic!("error creating {output_dir:?}"));

        let mut present_modules = Vec::new();
        for (feature, config) in arch
            .parse_config()
            .supported_features
            .iter()
            .map(|feature| (*feature, arch.feature_module_config(feature)))
        {
            let Some(intrinsics) = intrinsics_by_feature
                .get(feature)
                .filter(|intrinsics| !intrinsics.is_empty())
            else {
                panic!("No {feature} intrinsics found");
            };

            let code = config.gen_feature_module(intrinsics);
            let path = output_dir.join(format!("{}.rs", config.file_name));
            let file = File::create(&path).unwrap_or_else(|_| panic!("error creating {path:?}"));
            write_code(code, file);
            present_modules.push(config.clone());
        }

        let mod_code = arch.gen_arch_mod(&present_modules);
        let mod_path = output_dir.join("mod.rs");
        let mod_file =
            File::create(&mod_path).unwrap_or_else(|_| panic!("error creating {mod_path:?}"));
        write_code(mod_code, mod_file);
    }
}
