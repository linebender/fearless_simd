// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

use crate::types::{SIMD_TYPES, ScalarType, type_imports};

pub(crate) fn mk_ops() -> TokenStream {
    let imports = type_imports();

    let mut impls = vec![];

    for ty in SIMD_TYPES {
        let ops = match ty.scalar {
            ScalarType::Float => &["neg", "add", "sub", "mul", "div"][..],
            ScalarType::Int => &["neg", "add", "sub", "mul", "and", "or", "xor", "not"],
            ScalarType::Unsigned => &["add", "sub", "mul", "and", "or", "xor", "not"],
            ScalarType::Mask => &["and", "or", "xor", "not"],
        };
        let simd = ty.rust();
        for op in ops {
            let is_bit = matches!(*op, "and" | "or" | "xor");
            let is_unary = matches!(*op, "neg" | "not");
            let mut trait_name = format!("{}{}", op[0..1].to_ascii_uppercase(), &op[1..]);
            let opfn = if is_bit {
                trait_name = format!("Bit{trait_name}");
                Ident::new(&format!("bit{op}"), Span::call_site())
            } else {
                Ident::new(op, Span::call_site())
            };
            let op_assign_fn = format_ident!("{opfn}_assign");
            let trait_id = Ident::new(&trait_name, Span::call_site());
            let trait_assign_id = format_ident!("{trait_name}Assign");
            let simd_fn_name = format!("{op}_{}", ty.rust_name());
            let simd_fn = Ident::new(&simd_fn_name, Span::call_site());
            if is_unary {
                impls.push(quote! {
                    impl<S: Simd> core::ops::#trait_id for #simd<S> {
                        type Output = Self;
                        #[inline(always)]
                        fn #opfn(self) -> Self::Output {
                            self.simd.#simd_fn(self)
                        }
                    }
                });
            } else {
                let scalar = ty.scalar.rust(ty.scalar_bits);
                impls.push(quote! {
                    impl<S: Simd> core::ops::#trait_id for #simd<S> {
                        type Output = Self;
                        #[inline(always)]
                        fn #opfn(self, rhs: Self) -> Self::Output {
                            self.simd.#simd_fn(self, rhs)
                        }
                    }

                    impl<S: Simd> core::ops::#trait_assign_id for #simd<S> {
                        #[inline(always)]
                        fn #op_assign_fn(&mut self, rhs: Self) {
                            *self = self.simd.#simd_fn(*self, rhs);
                        }
                    }

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
                });
            }
        }

        // Shifts
        if matches!(ty.scalar, ScalarType::Int | ScalarType::Unsigned) {
            let shift_left_fn = format_ident!("shl_{}", ty.rust_name());
            let shift_right_fn = format_ident!("shr_{}", ty.rust_name());
            let shift_right_vectored_fn = format_ident!("shrv_{}", ty.rust_name());
            impls.push(quote! {
                impl<S: Simd> core::ops::Shl<u32> for #simd<S> {
                    type Output = Self;
                    #[inline(always)]
                    fn shl(self, rhs: u32) -> Self::Output {
                        self.simd.#shift_left_fn(self, rhs)
                    }
                }

                impl<S: Simd> core::ops::Shr<u32> for #simd<S> {
                    type Output = Self;
                    #[inline(always)]
                    fn shr(self, rhs: u32) -> Self::Output {
                        self.simd.#shift_right_fn(self, rhs)
                    }
                }

                impl<S: Simd> core::ops::ShlAssign<u32> for #simd<S> {
                    #[inline(always)]
                    fn shl_assign(&mut self, rhs: u32) {
                        *self = self.simd.#shift_left_fn(*self, rhs);
                    }
                }

                impl<S: Simd> core::ops::ShrAssign<u32> for #simd<S> {
                    #[inline(always)]
                    fn shr_assign(&mut self, rhs: u32) {
                        *self = self.simd.#shift_right_fn(*self, rhs);
                    }
                }

                impl<S: Simd> core::ops::Shr for #simd<S> {
                    type Output = Self;
                    #[inline(always)]
                    fn shr(self, rhs: Self) -> Self::Output {
                        self.simd.#shift_right_vectored_fn(self, rhs)
                    }
                }

                impl<S: Simd> core::ops::ShrAssign for #simd<S> {
                    #[inline(always)]
                    fn shr_assign(&mut self, rhs: Self) {
                        *self = self.simd.#shift_right_vectored_fn(*self, rhs);
                    }
                }
            });
        }
    }

    quote! {
        use crate::{Simd, SimdInto};
        #imports
        #( #impls )*
    }
}
