// Copyright 2026 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![doc = include_str!("../README.md")]

use core::mem;
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{AttrStyle, Attribute, FnArg, ItemFn, Pat, PatIdent, Result};

/// Run a SIMD-generic function body with the token's target features enabled.
///
/// The first typed parameter after an optional `self` receiver is used as the
/// SIMD token. See the [crate-level documentation](crate) for the complete
/// expansion, supported function forms, and semantic caveats.
#[proc_macro_attribute]
pub fn simd(args: TokenStream, item: TokenStream) -> TokenStream {
    expand(args.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(args: TokenStream2, item: TokenStream2) -> Result<TokenStream2> {
    if !args.is_empty() {
        return Err(syn::Error::new_spanned(
            args,
            "`#[simd]` does not accept arguments",
        ));
    }

    // ItemFn's signature grammar also accepts body-bearing inherent and trait
    // methods. Parsing the item rejects bodyless and specialization methods.
    let mut function: ItemFn = syn::parse2(item)?;

    function.modifiers.require_empty()?;
    reject_unsupported_signature(&function)?;
    reject_unsupported_attributes(&function.attrs)?;

    let token = simd_token(&mut function)?;
    let original_statements = mem::take(&mut function.block.stmts);

    // Inner function attributes are held in function.attrs by Syn. Leaving
    // them there keeps them at the beginning of the outer function body,
    // rather than changing their scope by moving them into this closure.
    // Keep generated wrapper tokens on their normal macro-expansion spans.
    // Giving the entire call the token parameter's source span makes Clippy's
    // `semicolon_if_nothing_returned` lint fire on unit-returning functions.
    let vectorize_call: syn::Expr = syn::parse_quote! {
        #token.vectorize(
            #[inline(always)]
            || { #(#original_statements)* }
        )
    };
    function
        .block
        .stmts
        .push(syn::Stmt::Expr(vectorize_call, None));

    Ok(quote!(#function))
}

fn reject_unsupported_signature(function: &ItemFn) -> Result<()> {
    if let Some(asyncness) = &function.sig.asyncness {
        return Err(syn::Error::new(
            asyncness.span,
            "`#[simd]` does not support async functions",
        ));
    }
    if let Some(constness) = &function.sig.constness {
        return Err(syn::Error::new(
            constness.span,
            "`#[simd]` does not support const functions",
        ));
    }
    if let Some(variadic) = &function.sig.variadic {
        return Err(syn::Error::new_spanned(
            variadic,
            "`#[simd]` does not support variadic functions",
        ));
    }
    Ok(())
}

fn reject_unsupported_attributes(attrs: &[Attribute]) -> Result<()> {
    for attr in attrs {
        if !matches!(attr.style, AttrStyle::Outer) {
            continue;
        }

        let reason = if is_attribute(attr, "track_caller") {
            Some("`#[simd]` cannot preserve `#[track_caller]` through its closure")
        } else if is_attribute(attr, "naked") {
            Some("`#[simd]` cannot be used on a naked function")
        } else if is_attribute(attr, "instruction_set") {
            Some("`#[simd]` cannot be combined with `#[instruction_set]`")
        } else {
            None
        };

        if let Some(reason) = reason {
            return Err(syn::Error::new_spanned(attr, reason));
        }
    }
    Ok(())
}

fn is_attribute(attr: &Attribute, name: &str) -> bool {
    if attr.path().is_ident(name) {
        return true;
    }

    // Attributes with safety obligations use `#[unsafe(attribute)]` syntax.
    // Naked functions require this form on supported Rust releases.
    attr.path().is_ident("unsafe")
        && attr
            .parse_args::<syn::Path>()
            .is_ok_and(|path| path.is_ident(name))
}

fn simd_token(function: &mut ItemFn) -> Result<Ident> {
    let Some(argument) = function
        .sig
        .inputs
        .iter_mut()
        .find_map(|argument| match argument {
            FnArg::Receiver(_) => None,
            FnArg::Typed(argument) => Some(argument),
        })
    else {
        return Err(syn::Error::new_spanned(
            &function.sig.inputs,
            "`#[simd]` requires a SIMD token parameter after any receiver",
        ));
    };

    reject_conditional_attributes(&argument.attrs)?;

    match &mut *argument.pat {
        Pat::Ident(pattern) => {
            reject_conditional_attributes(&pattern.attrs)?;
            if let Some(by_ref) = &pattern.by_ref {
                return Err(syn::Error::new(
                    by_ref.span,
                    "the SIMD token parameter must be bound by value, not `ref`",
                ));
            }
            if let Some((at, _)) = &pattern.subpat {
                return Err(syn::Error::new(
                    at.span,
                    "the SIMD token parameter cannot use an `@` subpattern",
                ));
            }
            Ok(pattern.ident.clone())
        }
        Pat::Wild(pattern) => {
            reject_conditional_attributes(&pattern.attrs)?;
            let token = Ident::new("__fearless_simd_token", Span::mixed_site());
            let attrs = mem::take(&mut pattern.attrs);
            *argument.pat = Pat::Ident(PatIdent {
                attrs,
                by_ref: None,
                mutability: None,
                ident: token.clone(),
                subpat: None,
            });
            Ok(token)
        }
        pattern => Err(syn::Error::new_spanned(
            pattern,
            "the SIMD token parameter must be an identifier or `_`",
        )),
    }
}

fn reject_conditional_attributes(attrs: &[Attribute]) -> Result<()> {
    if let Some(attr) = attrs
        .iter()
        .find(|attr| attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr"))
    {
        return Err(syn::Error::new_spanned(
            attr,
            "the SIMD token parameter cannot be conditional",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::expand;
    use quote::{ToTokens, quote};
    use syn::{AttrStyle, Expr, ItemFn, Stmt};

    fn expand_ok(item: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        expand(proc_macro2::TokenStream::new(), item).expect("macro expansion should succeed")
    }

    fn expand_err(item: proc_macro2::TokenStream) -> String {
        expand(proc_macro2::TokenStream::new(), item)
            .expect_err("macro expansion should fail")
            .to_string()
    }

    #[test]
    fn expands_body_as_an_attributed_tail_closure() {
        let expanded = expand_ok(quote! {
            fn add<S: Simd>(simd: S, lhs: u32, rhs: u32) -> u32 {
                let sum = lhs + rhs;
                sum
            }
        });
        let parsed: ItemFn = syn::parse2(expanded).expect("expanded function parses");
        let Some(Stmt::Expr(Expr::MethodCall(call), None)) = parsed.block.stmts.last() else {
            panic!("function tail should be a method call");
        };
        let Some(Expr::Closure(closure)) = call.args.first() else {
            panic!("vectorize argument should be a closure");
        };
        let Expr::Block(body) = &*closure.body else {
            panic!("closure body should be a block");
        };

        assert_eq!(call.method, "vectorize");
        assert_eq!(call.args.len(), 1);
        assert!(closure.capture.is_none());
        assert_eq!(closure.attrs.len(), 1);
        assert!(closure.attrs[0].path().is_ident("inline"));
        let inline_kind: proc_macro2::Ident = closure.attrs[0]
            .parse_args()
            .expect("inline attribute has one identifier argument");
        assert_eq!(inline_kind, "always");
        assert_eq!(body.block.stmts.len(), 2);
        assert!(
            parsed
                .attrs
                .iter()
                .all(|attr| !attr.path().is_ident("inline"))
        );
    }

    #[test]
    fn unit_returns_remain_tail_expressions() {
        for item in [
            quote! {
                fn implicit_unit<S: Simd>(simd: S) { let _ = simd.level(); }
            },
            quote! {
                fn explicit_unit<S: Simd>(simd: S) -> () { let _ = simd.level(); }
            },
        ] {
            let expanded = expand_ok(item);
            let parsed: ItemFn = syn::parse2(expanded).expect("expanded function parses");
            let Some(Stmt::Expr(Expr::MethodCall(call), None)) = parsed.block.stmts.last() else {
                panic!("unit function tail should be a method call");
            };
            let Some(Expr::Closure(closure)) = call.args.first() else {
                panic!("vectorize argument should be a closure");
            };

            assert!(matches!(closure.output, syn::ReturnType::Default));
        }
    }

    #[test]
    fn non_unit_return_remains_a_tail_expression() {
        let expanded = expand_ok(quote! {
            fn non_unit<S: Simd>(simd: S) -> u32 { 42 }
        });
        let parsed: ItemFn = syn::parse2(expanded).expect("expanded function parses");

        assert!(matches!(
            parsed.block.stmts.last(),
            Some(Stmt::Expr(Expr::MethodCall(_), None))
        ));
    }

    #[test]
    fn preserves_signature_attributes_and_inner_attributes() {
        let expanded = expand_ok(quote! {
            #[doc = "docs"]
            #[inline(never)]
            #[target_feature(enable = "sse2")]
            unsafe extern "C" fn operation<'a, S, T>(simd: S, value: &'a T) -> &'a T
            where
                S: Simd,
            {
                #![allow(unused_unsafe)]
                unsafe { value }
            }
        });
        let parsed: ItemFn = syn::parse2(expanded.clone()).expect("expanded function parses");

        assert!(matches!(parsed.sig.safety, syn::Safety::Unsafe(_)));
        assert!(parsed.sig.abi.is_some());
        assert!(parsed.sig.generics.where_clause.is_some());
        assert_eq!(
            parsed
                .attrs
                .iter()
                .filter(|attr| matches!(attr.style, AttrStyle::Inner(_)))
                .count(),
            1
        );

        let text = expanded.to_string();
        let inner_attr = text
            .find("# ! [allow")
            .expect("inner attribute is retained");
        let call = text
            .find("simd . vectorize")
            .expect("vectorize call exists");
        assert!(inner_attr < call);
        assert_eq!(text.matches("inline (never)").count(), 1);
        assert_eq!(text.matches("inline (always)").count(), 1);
        assert!(text.contains("target_feature"));
    }

    #[test]
    fn selects_first_typed_parameter_after_receiver() {
        let expanded = expand_ok(quote! {
            fn method<S: Simd>(&self, mut backend: S, value: u32) -> u32 {
                backend.level();
                value
            }
        });
        let text = expanded.to_string();

        assert!(text.contains("mut backend : S"));
        assert!(text.contains("backend . vectorize"));
    }

    #[test]
    fn gives_a_wildcard_token_a_private_binding() {
        let expanded = expand_ok(quote! {
            fn operation<S: Simd>(_: S, value: u32) -> u32 { value }
        });
        let text = expanded.to_string();

        assert_eq!(text.matches("__fearless_simd_token").count(), 2);
        assert!(text.contains("__fearless_simd_token : S"));
        assert!(text.contains("__fearless_simd_token . vectorize"));
    }

    #[test]
    fn wildcard_binding_does_not_rename_a_user_binding_with_the_same_spelling() {
        let expanded = expand_ok(quote! {
            fn operation<S: Simd>(_: S, __fearless_simd_token: u32) -> u32 {
                __fearless_simd_token
            }
        });
        let parsed: ItemFn = syn::parse2(expanded).expect("expanded function parses");

        assert_eq!(parsed.sig.inputs.len(), 2);
        assert_eq!(
            parsed
                .to_token_stream()
                .to_string()
                .matches("__fearless_simd_token")
                .count(),
            4
        );
    }

    #[test]
    fn accepts_default_trait_method_syntax() {
        let expanded = expand_ok(quote! {
            fn operation<S: Simd>(&self, simd: S) -> u32 { 42 }
        });
        assert!(expanded.to_string().contains("simd . vectorize"));
    }

    #[test]
    fn rejects_attribute_arguments() {
        let error = expand(
            quote!(token = simd),
            quote!(
                fn f<S: Simd>(simd: S) {}
            ),
        )
        .expect_err("arguments should be rejected")
        .to_string();
        assert_eq!(error, "`#[simd]` does not accept arguments");
    }

    #[test]
    fn rejects_unsupported_signatures() {
        assert!(
            expand_err(quote!(
                async fn f<S: Simd>(simd: S) {}
            ))
            .contains("async functions")
        );
        assert!(
            expand_err(quote!(
                const fn f<S: Simd>(simd: S) {}
            ))
            .contains("const functions")
        );
        assert!(
            expand_err(quote!(
                unsafe extern "C" fn f<S: Simd>(simd: S, ...) {}
            ))
            .contains("variadic functions")
        );
        assert!(expand(quote!(), quote!(default fn f<S: Simd>(simd: S) {})).is_err());
    }

    #[test]
    fn rejects_unsupported_function_attributes() {
        assert!(
            expand_err(quote!(
                #[track_caller]
                fn f<S: Simd>(simd: S) {}
            ))
            .contains("cannot preserve")
        );
        assert!(
            expand_err(quote!(
                #[naked]
                fn f<S: Simd>(simd: S) {}
            ))
            .contains("naked")
        );
        assert!(
            expand_err(quote!(
                #[unsafe(naked)]
                fn f<S: Simd>(simd: S) {}
            ))
            .contains("naked")
        );
        assert!(
            expand_err(quote!(
                #[instruction_set(arm::a32)]
                fn f<S: Simd>(simd: S) {}
            ))
            .contains("instruction_set")
        );
    }

    #[test]
    fn rejects_missing_or_unsupported_token_patterns() {
        assert!(
            expand_err(quote!(
                fn f() {}
            ))
            .contains("requires a SIMD token")
        );
        assert!(
            expand_err(quote!(
                fn f<S: Simd>(ref simd: S) {}
            ))
            .contains("bound by value")
        );
        assert!(
            expand_err(quote!(
                fn f<S: Simd>(simd @ _: S) {}
            ))
            .contains("subpattern")
        );
        assert!(
            expand_err(quote!(
                fn f<S: Simd>((simd, _): (S, u32)) {}
            ))
            .contains("identifier or `_`")
        );
    }

    #[test]
    fn rejects_conditional_token_parameters() {
        assert!(
            expand_err(quote!(
                fn f<S: Simd>(#[cfg(any())] simd: S) {}
            ))
            .contains("cannot be conditional")
        );
        assert!(
            expand_err(quote!(
                fn f<S: Simd>(#[cfg_attr(any(), allow(unused))] simd: S) {}
            ))
            .contains("cannot be conditional")
        );
    }

    #[test]
    fn rejects_non_functions_and_bodyless_functions() {
        assert!(
            expand_err(quote!(
                struct NotAFunction;
            ))
            .contains("expected")
        );
        assert!(
            expand_err(quote!(
                fn bodyless<S: Simd>(simd: S);
            ))
            .contains("expected")
        );
    }
}
