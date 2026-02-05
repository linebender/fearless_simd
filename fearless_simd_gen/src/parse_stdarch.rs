// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Parse stdarch source files to extract intrinsic function signatures.

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

use anyhow::{Context, Result};
use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use syn::{ItemFn, LitStr, visit::Visit};

/// Recursively check if a token stream contains a specific literal string.
fn contains_literal(tokens: &TokenStream, target: &str) -> bool {
    for token in tokens.clone() {
        match token {
            TokenTree::Literal(lit) if lit.to_string() == target => return true,
            TokenTree::Group(group) => {
                if contains_literal(&group.stream(), target) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Recursively check if a token stream contains a specific identifier.
fn contains_ident(tokens: &TokenStream, target: &str) -> bool {
    for token in tokens.clone() {
        match token {
            TokenTree::Ident(ident) if ident == target => return true,
            TokenTree::Group(group) => {
                if contains_ident(&group.stream(), target) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Recursively check if a token stream contains `not(target_arch = "arm")`.
fn contains_not_target_arch_arm(tokens: &TokenStream) -> bool {
    let tokens: Vec<_> = tokens.clone().into_iter().collect();
    for (i, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Ident(ident) if ident == "not" => {
                // Check the next token is a group with target_arch = "arm"
                if let Some(TokenTree::Group(group)) = tokens.get(i + 1) {
                    let inner = group.stream();
                    if contains_ident(&inner, "target_arch") && contains_literal(&inner, "\"arm\"")
                    {
                        return true;
                    }
                }
            }
            TokenTree::Group(group) => {
                if contains_not_target_arch_arm(&group.stream()) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// A visitor that extracts intrinsic functions from a Rust file.
struct IntrinsicVisitor {
    intrinsics: Vec<ItemFn>,
    /// Set of all previously-visited intrinsic names. Some modules contain duplicate intrinsics (e.g. NEON sometimes
    /// defines intrinsics once per endianness).
    visited: HashSet<String>,
    /// The target feature for the file/module being parsed.
    module_feature: String,
}

impl IntrinsicVisitor {
    fn new(module_feature: String) -> Self {
        Self {
            intrinsics: Vec::new(),
            visited: HashSet::new(),
            module_feature,
        }
    }
}

impl<'ast> Visit<'ast> for IntrinsicVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        // Skip non-public functions
        if !matches!(node.vis, syn::Visibility::Public(_)) {
            return;
        }

        let name = node.sig.ident.to_string();
        if !self.visited.insert(name) {
            return;
        }

        // Skip functions that don't enable the target feature we're looking for. This should filter out non-intrinsics.
        let mut target_features = Vec::new();
        for attr in &node.attrs {
            // Skip directly unstable intrinsics
            if attr.path().is_ident("unstable") {
                return;
            }

            // Skip intrinsics that aren't usable on stable Rust yet.
            // Check for `since = "CURRENT_RUSTC_VERSION"` in both direct #[stable] and #[cfg_attr(..., stable(...))]
            if (attr.path().is_ident("stable") || attr.path().is_ident("cfg_attr"))
                && contains_literal(&attr.meta.to_token_stream(), "\"CURRENT_RUSTC_VERSION\"") {
                    return;
                }

            // Skip intrinsics that are unstable on non-ARM32 platforms:
            // #[cfg_attr(not(target_arch = "arm"), unstable(...))]
            // But keep intrinsics that are only unstable on ARM32:
            // #[cfg_attr(target_arch = "arm", unstable(...))]
            if attr.path().is_ident("cfg_attr") {
                let tokens = attr.meta.to_token_stream();
                if contains_not_target_arch_arm(&tokens) && contains_ident(&tokens, "unstable") {
                    return;
                }
            }

            if !attr.path().is_ident("target_feature") {
                continue;
            }

            if let Err(err) = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("enable") {
                    let value = meta.value()?;
                    let lit: LitStr = value.parse()?;
                    target_features.extend(lit.value().split(',').map(str::trim).map(String::from));
                }
                Ok(())
            }) {
                panic!(
                    "Failed to parse #[target_feature] attribute on {}: {err}",
                    attr.path().to_token_stream()
                );
            }
        }

        if target_features.is_empty()
            || !target_features
                .iter()
                .any(|feature| feature == &self.module_feature)
        {
            return;
        }

        self.intrinsics.push(node.clone());

        // Continue visiting
        syn::visit::visit_item_fn(self, node);
    }
}

/// Parse a single Rust source file and extract intrinsics.
pub(crate) fn parse_file(path: &Path, module_feature: &str) -> Result<Vec<ItemFn>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    let syntax = syn::parse_file(&content)
        .with_context(|| format!("Failed to parse file: {}", path.display()))?;

    let mut visitor = IntrinsicVisitor::new(module_feature.to_string());
    visitor.visit_file(&syntax);

    Ok(visitor.intrinsics)
}

/// Architecture definition with its source directories and target features.
#[derive(Debug, Clone)]
pub(crate) struct ArchConfig {
    pub source_dirs: &'static [&'static str],
    pub module_feature: fn(&str) -> &str,
    pub supported_features: &'static [&'static str],
}

pub(crate) const X86_CONFIG: ArchConfig = ArchConfig {
    source_dirs: &["crates/core_arch/src/x86"],
    module_feature: |file_stem| match file_stem {
        "sse41" => "sse4.1",
        "sse42" => "sse4.2",
        _ => file_stem,
    },
    supported_features: &[
        "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "avx", "avx2", "fma",
    ],
};

/// aarch64 architecture configuration.
pub(crate) const AARCH64_CONFIG: ArchConfig = ArchConfig {
    source_dirs: &[
        "crates/core_arch/src/aarch64/neon",
        "crates/core_arch/src/arm_shared/neon",
    ],
    module_feature: |_file_stem| "neon",
    supported_features: &["neon"],
};

/// wasm32 architecture configuration.
pub(crate) const WASM32_CONFIG: ArchConfig = ArchConfig {
    source_dirs: &["crates/core_arch/src/wasm32"],
    module_feature: |_file_stem| "simd128",
    supported_features: &["simd128"],
};

pub(crate) type IntrinsicsByFeature = BTreeMap<String, Vec<ItemFn>>;

fn module_file_stem(path: &Path) -> Option<&str> {
    if !path.is_file() || path.extension().is_none_or(|ext| ext != "rs") {
        return None;
    }

    let file_stem = path.file_stem()?.to_str()?;
    if file_stem == "mod"
        || file_stem == "test"
        || file_stem.starts_with("test_")
        || file_stem == "macros"
    {
        return None;
    }

    Some(file_stem)
}

/// Parse all intrinsics from a stdarch directory for a given architecture.
pub(crate) fn parse_arch(stdarch_root: &Path, config: &ArchConfig) -> Result<IntrinsicsByFeature> {
    let mut result: IntrinsicsByFeature = BTreeMap::new();
    let supported_set = HashSet::<_>::from_iter(config.supported_features.iter().copied());

    for source_dir in config.source_dirs {
        let dir_path = stdarch_root.join(source_dir);
        if !dir_path.exists() {
            continue;
        }

        let mut entries: Vec<_> = fs::read_dir(&dir_path)
            .unwrap()
            .map(|entry| entry.unwrap())
            .collect();
        // Nothing we're doing *should* vary based on file order, but just in case...
        entries.sort_by_cached_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let Some(file_stem) = module_file_stem(&path) else {
                continue;
            };

            let module_feature = (config.module_feature)(file_stem);
            if !supported_set.contains(module_feature) {
                continue;
            }

            let intrinsics = parse_file(&path, module_feature)?;
            match result.get_mut(module_feature) {
                Some(existing_intrinsics) => {
                    existing_intrinsics.extend(intrinsics);
                }
                None => {
                    result.insert(module_feature.to_string(), intrinsics);
                }
            }
        }
    }

    Ok(result)
}
