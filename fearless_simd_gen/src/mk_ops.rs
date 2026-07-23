// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

use crate::{
    generic::generic_op_name,
    ops::{CoreOpTrait, OpKind, OpSig, TyFlavor, overloaded_ops_for},
    types::{SIMD_TYPES, ScalarType, VecType, type_imports},
};

pub(crate) fn mk_ops() -> TokenStream {
    let imports = type_imports();

    let mut impls = vec![];

    for ty in SIMD_TYPES {
        let simd = ty.rust();
        for op in overloaded_ops_for(ty.scalar) {
            let OpKind::Overloaded(core_op) = op.kind else {
                continue;
            };
            let opfn = core_op.op_fn();
            let trait_name = core_op.trait_name();
            let simd_name = core_op.simd_name();
            let op_assign_fn = format_ident!("{opfn}_assign");
            let trait_id = Ident::new(trait_name, Span::call_site());
            let trait_assign_id = format_ident!("{trait_name}Assign");
            let simd_fn = generic_op_name(simd_name, ty);
            let opfn = Ident::new(opfn, Span::call_site());
            let doc = op.format_docstring(TyFlavor::VecImpl);
            let method_doc = if core_op == CoreOpTrait::Add {
                format!("{doc}\n\n{}", add_doctest(ty))
            } else {
                doc.clone()
            };
            let doc_attrs = make_doc_attrs(&doc);
            let method_doc_attrs = make_doc_attrs(&method_doc);

            match core_op {
                CoreOpTrait::ShlVectored | CoreOpTrait::ShrVectored => {
                    impls.push(quote! {
                        impl<S: Simd> core::ops::#trait_id for #simd<S> {
                            type Output = Self;
                            #method_doc_attrs
                            #[inline(always)]
                            fn #opfn(self, rhs: Self) -> Self::Output {
                                self.simd.#simd_fn(self, rhs)
                            }
                        }

                        impl<S: Simd> core::ops::#trait_assign_id for #simd<S> {
                            #doc_attrs
                            #[inline(always)]
                            fn #op_assign_fn(&mut self, rhs: Self) {
                                *self = self.simd.#simd_fn(*self, rhs);
                            }
                        }
                    });
                }
                CoreOpTrait::Shl | CoreOpTrait::Shr => {
                    impls.push(quote! {
                        impl<S: Simd> core::ops::#trait_id<u32> for #simd<S> {
                            type Output = Self;
                            #method_doc_attrs
                            #[inline(always)]
                            fn #opfn(self, rhs: u32) -> Self::Output {
                                self.simd.#simd_fn(self, rhs)
                            }
                        }

                        impl<S: Simd> core::ops::#trait_assign_id<u32> for #simd<S> {
                            #[inline(always)]
                            fn #op_assign_fn(&mut self, rhs: u32) {
                                *self = self.simd.#simd_fn(*self, rhs);
                            }
                        }
                    });
                }
                _ if matches!(op.sig, OpSig::Unary) => {
                    impls.push(quote! {
                        impl<S: Simd> core::ops::#trait_id for #simd<S> {
                            type Output = Self;
                            #method_doc_attrs
                            #[inline(always)]
                            fn #opfn(self) -> Self::Output {
                                self.simd.#simd_fn(self)
                            }
                        }
                    });
                }
                _ => {
                    let scalar = ty.scalar.rust(ty.scalar_bits);
                    let scalar_overloads = (ty.scalar != ScalarType::Mask).then(|| {
                        quote! {
                            impl<S: Simd> core::ops::#trait_id<#scalar> for #simd<S> {
                                type Output = Self;
                                #[inline(always)]
                                fn #opfn(self, rhs: #scalar) -> Self::Output {
                                    self.simd.#simd_fn(self, rhs.simd_into(self.simd))
                                }
                            }

                            impl<S: Simd> core::ops::#trait_assign_id<#scalar> for #simd<S> {
                                #[inline(always)]
                                fn #op_assign_fn(&mut self, rhs: #scalar) {
                                    *self = self.simd.#simd_fn(*self, rhs.simd_into(self.simd));
                                }
                            }

                            impl<S: Simd> core::ops::#trait_id<#simd<S>> for #scalar {
                                type Output = #simd<S>;
                                #[inline(always)]
                                fn #opfn(self, rhs: #simd<S>) -> Self::Output {
                                    rhs.simd.#simd_fn(self.simd_into(rhs.simd), rhs)
                                }
                            }
                        }
                    });
                    impls.push(quote! {
                        impl<S: Simd> core::ops::#trait_id for #simd<S> {
                            type Output = Self;
                            #method_doc_attrs
                            #[inline(always)]
                            fn #opfn(self, rhs: Self) -> Self::Output {
                                self.simd.#simd_fn(self, rhs)
                            }
                        }

                        impl<S: Simd> core::ops::#trait_assign_id for #simd<S> {
                            #doc_attrs
                            #[inline(always)]
                            fn #op_assign_fn(&mut self, rhs: Self) {
                                *self = self.simd.#simd_fn(*self, rhs);
                            }
                        }

                        #scalar_overloads
                    });
                }
            }
        }
    }

    quote! {
        use crate::{Simd, SimdInto};
        #imports
        #( #impls )*
    }
}

fn make_doc_attrs(doc: &str) -> TokenStream {
    let lines = doc.lines();
    quote! {
        #(#[doc = #lines])*
    }
}

fn add_doctest(ty: &VecType) -> String {
    let rust_name = ty.rust_name();
    let fn_name = format!("add_{rust_name}");
    let scalar_name = ty.scalar.rust_name(ty.scalar_bits);
    let (lhs, rhs, expected) = add_doctest_values(ty);
    let lhs = format_array(&lhs);
    let rhs = format_array(&rhs);
    let expected = format_array(&expected);
    let len = ty.len;

    format!(
        r#"```rust
# use fearless_simd::{{prelude::*, {rust_name}}};
# fearless_simd::__simd_doctest! {{ {fn_name},
#[inline(always)]
fn {fn_name}<S: Simd>(simd: S) {{
    let a = {rust_name}::simd_from(
        simd,
        {lhs},
    );
    let b = {rust_name}::simd_from(
        simd,
        {rhs},
    );

    assert_eq!(
        <[{scalar_name}; {len}]>::from(a + b),
        {expected},
    );
}}
# }}
```"#
    )
}

fn add_doctest_values(ty: &VecType) -> (Vec<String>, Vec<String>, Vec<String>) {
    match ty.scalar {
        ScalarType::Float | ScalarType::Unsigned => {
            let lhs = (0..ty.len).map(|i| format_literal(2 * i, ty.scalar));
            let rhs = (0..ty.len).map(|i| format_literal(2 * i + 1, ty.scalar));
            let expected = (0..ty.len).map(|i| format_literal(4 * i + 1, ty.scalar));
            (lhs.collect(), rhs.collect(), expected.collect())
        }
        ScalarType::Int => {
            let value = |lane: usize, offset: usize| {
                let magnitude = if lane % 2 == 0 {
                    lane + offset
                } else {
                    lane + offset - 1
                };
                if lane % 2 == 0 {
                    magnitude.to_string()
                } else {
                    format!("-{magnitude}")
                }
            };
            let lhs = (0..ty.len).map(|i| value(i, 1));
            let rhs = (0..ty.len).map(|i| value(i, 2));
            let expected = (0..ty.len).map(|i| {
                if i % 2 == 0 {
                    (2 * i + 3).to_string()
                } else {
                    format!("-{}", 2 * i + 1)
                }
            });
            (lhs.collect(), rhs.collect(), expected.collect())
        }
        ScalarType::Mask => unreachable!("masks do not implement Add"),
    }
}

fn format_literal(value: usize, scalar: ScalarType) -> String {
    if scalar == ScalarType::Float {
        format!("{value}.0")
    } else {
        value.to_string()
    }
}

fn format_array(values: &[String]) -> String {
    let mut array = String::from("[\n");
    for chunk in values.chunks(8) {
        array.push_str("            ");
        array.push_str(&chunk.join(", "));
        array.push_str(",\n");
    }
    array.push_str("        ]");
    array
}
