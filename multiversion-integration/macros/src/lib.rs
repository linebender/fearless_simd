//! Deliberately restricted research prototype, not a production macro.
use std::collections::HashSet;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Expr, FnArg, ItemFn, Pat, parse_quote};

#[proc_macro_attribute]
pub fn simd(attr: TokenStream, input: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new(Span::call_site(), "prototype takes no arguments")
            .into_compile_error()
            .into();
    }
    expand(syn::parse_macro_input!(input as ItemFn))
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(function: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let signature = &function.sig;
    if signature.asyncness.is_some()
        || signature.constness.is_some()
        || signature.unsafety.is_some()
        || signature.abi.is_some()
        || signature.variadic.is_some()
        || !signature.generics.params.is_empty()
        || signature.generics.where_clause.is_some()
    {
        return Err(syn::Error::new_spanned(
            signature,
            "prototype supports safe synchronous non-generic Rust free functions",
        ));
    }
    for attr in &function.attrs {
        if !attr.path().is_ident("inline") && !attr.path().is_ident("doc") {
            return Err(syn::Error::new_spanned(
                attr,
                "prototype only supports inline and doc attributes",
            ));
        }
    }
    let mut arguments = Vec::new();
    for arg in &signature.inputs {
        let FnArg::Typed(arg) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "prototype supports free functions",
            ));
        };
        let Pat::Ident(pat) = &*arg.pat else {
            return Err(syn::Error::new_spanned(
                arg,
                "prototype requires identifier parameters",
            ));
        };
        if pat.by_ref.is_some() || pat.subpat.is_some() {
            return Err(syn::Error::new_spanned(
                pat,
                "prototype requires by-value parameters",
            ));
        }
        arguments.push(pat.ident.clone());
    }

    // Conservatively exclude any name bound anywhere in the function. This can
    // miss specialization, but avoids confusing local callables with imports.
    let mut bindings = Bindings::default();
    bindings.visit_item_fn(&function);
    let name = &signature.ident;
    let vis = &function.vis;
    let implementation = format_ident!("__simd_research_{}_with", name);
    let witness = format_ident!("__simd_research_witness", span = Span::mixed_site());
    let mut body = *function.block.clone();
    Rewrite {
        entry: format_ident!("__call"),
        witness: witness.clone(),
        locals: &bindings.names,
    }
    .visit_block_mut(&mut body);

    let mut sig = signature.clone();
    sig.ident = implementation.clone();
    sig.generics.params.push(parse_quote!(__SimdResearchBackend:
        ::simd_research_runtime::fearless_simd::Simd));
    sig.generics
        .params
        .push(parse_quote!(__SimdResearchOriginal));
    sig.inputs
        .insert(0, parse_quote!(_: __SimdResearchOriginal));
    sig.inputs
        .insert(0, parse_quote!(#witness: __SimdResearchBackend));
    let attrs = &function.attrs;
    let inline = if attrs.iter().any(|a| a.path().is_ident("inline")) {
        quote!()
    } else {
        quote!(#[inline])
    };
    let mut wrapper = function.clone();
    wrapper.block = Box::new(parse_quote!({
        ::simd_research_runtime::fearless_simd::dispatch!(
            ::simd_research_runtime::selected_level(),
            #witness => #implementation(#witness, (), #(#arguments),*)
        )
    }));
    Ok(quote! {
        #wrapper
        #[doc(hidden)]
        #inline
        #(#attrs)*
        pub #sig {
            ::simd_research_runtime::fearless_simd::Simd::vectorize(
                #witness,
                #[inline(always)]
                move || #body,
            )
        }
        #[doc(hidden)]
        #vis mod #name {
            pub use super::#implementation as __call;
        }
    })
}

#[derive(Default)]
struct Bindings {
    names: HashSet<String>,
}
impl<'ast> Visit<'ast> for Bindings {
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.names.insert(pat.ident.to_string());
        syn::visit::visit_pat_ident(self, pat);
    }
    fn visit_item(&mut self, _: &'ast syn::Item) {}
}

struct Rewrite<'a> {
    entry: syn::Ident,
    witness: syn::Ident,
    locals: &'a HashSet<String>,
}
impl VisitMut for Rewrite<'_> {
    // A nested item or deferred closure is a separate execution boundary.
    fn visit_item_mut(&mut self, _: &mut syn::Item) {}
    fn visit_expr_closure_mut(&mut self, _: &mut syn::ExprClosure) {}
    fn visit_expr_async_mut(&mut self, _: &mut syn::ExprAsync) {}
    fn visit_expr_mut(&mut self, expression: &mut Expr) {
        syn::visit_mut::visit_expr_mut(self, expression);
        let Expr::Call(call) = expression else { return };
        let Expr::Path(path) = &*call.func else {
            return;
        };
        // Importable single names suffice for the cross-crate alias proof.
        // Leave qualified/associated/generic calls unchanged in this prototype.
        if path.qself.is_some() || path.path.segments.len() != 1 {
            return;
        }
        let segment = &path.path.segments[0];
        if !segment.arguments.is_empty()
            || self.locals.contains(&segment.ident.to_string())
            || segment
                .ident
                .to_string()
                .chars()
                .next()
                .is_some_and(char::is_uppercase)
        {
            return;
        }
        let callee = &path.path;
        let entry = &self.entry;
        let witness = &self.witness;
        let args = &call.args;
        let ids: Vec<_> = (0..args.len())
            .map(|i| format_ident!("__arg_{i}", span = Span::mixed_site()))
            .collect();
        let types: Vec<_> = (0..args.len()).map(|i| format_ident!("__Arg{i}")).collect();
        let attrs = &call.attrs;
        *expression = parse_quote! {
            #(#attrs)*
            {
                mod __simd_fallback {
                    pub mod __simd_callee {
                        #[inline(always)]
                        pub fn #entry<__Witness, __F, __R, #(#types),*>(_: __Witness, f: __F, #(#ids: #types),*) -> __R
                        where __F: FnOnce(#(#types),*) -> __R {
                            f(#(#ids),*)
                        }
                    }
                }
                #[allow(unused_imports)]
                use __simd_fallback::*;
                #[allow(unused_imports)]
                use #callee as __simd_callee;
                // A safe call preserves the original argument expressions,
                // including coercions, reborrows and their unsafe context.
                __simd_callee::#entry(#witness, __simd_callee, #args)
            }
        };
    }
}
