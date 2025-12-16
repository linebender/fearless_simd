// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

use crate::{
    ops::{OpSig, RefKind},
    types::{SIMD_TYPES, ScalarType, VecType},
};

pub(crate) fn generic_op_name(op: &str, ty: &VecType) -> Ident {
    Ident::new(&format!("{op}_{}", ty.rust_name()), Span::call_site())
}

/// Implementation based on split/combine
///
/// Only suitable for lane-wise and block-wise operations
pub(crate) fn generic_op(op: &str, sig: OpSig, ty: &VecType) -> TokenStream {
    let name = format!("{op}_{}", ty.rust_name());
    let split = Ident::new(&format!("split_{}", ty.rust_name()), Span::call_site());
    let half = VecType::new(ty.scalar, ty.scalar_bits, ty.len / 2);
    let combine = Ident::new(&format!("combine_{}", half.rust_name()), Span::call_site());
    let do_half = Ident::new(&format!("{op}_{}", half.rust_name()), Span::call_site());
    let method_sig = sig.simd_trait_method_sig(ty, &name);
    let method_sig = quote! {
        #[inline(always)]
        #method_sig
    };
    match sig {
        OpSig::Splat => {
            quote! {
                #method_sig {
                    let half = self.#do_half(val);
                    self.#combine(half, half)
                }
            }
        }
        OpSig::Unary => {
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::Binary => {
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    self.#combine(self.#do_half(a0, b0), self.#do_half(a1, b1))
                }
            }
        }
        OpSig::Shift => {
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0, shift), self.#do_half(a1, shift))
                }
            }
        }
        OpSig::Ternary => {
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    let (c0, c1) = self.#split(c);
                    self.#combine(self.#do_half(a0, b0, c0), self.#do_half(a1, b1, c1))
                }
            }
        }
        OpSig::Compare => {
            let half_mask = VecType::new(ScalarType::Mask, ty.scalar_bits, ty.len / 2);
            let combine_mask = Ident::new(
                &format!("combine_{}", half_mask.rust_name()),
                Span::call_site(),
            );
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    self.#combine_mask(self.#do_half(a0, b0), self.#do_half(a1, b1))
                }
            }
        }
        OpSig::Select => {
            let mask_ty = VecType::new(ScalarType::Mask, ty.scalar_bits, ty.len);
            let split_mask =
                Ident::new(&format!("split_{}", mask_ty.rust_name()), Span::call_site());
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split_mask(a);
                    let (b0, b1) = self.#split(b);
                    let (c0, c1) = self.#split(c);
                    self.#combine(self.#do_half(a0, b0, c0), self.#do_half(a1, b1, c1))
                }
            }
        }
        OpSig::Zip { select_low } => {
            let (e1, e2, e3) = if select_low {
                (
                    quote! {
                        (a0, _)
                    },
                    quote! {
                        (b0, _)
                    },
                    quote! {
                        a0, b0
                    },
                )
            } else {
                (
                    quote! {
                        (_, a1)
                    },
                    quote! {
                        (_, b1)
                    },
                    quote! {
                        a1, b1
                    },
                )
            };

            let zip_low_half =
                Ident::new(&format!("zip_low_{}", half.rust_name()), Span::call_site());
            let zip_high_half =
                Ident::new(&format!("zip_high_{}", half.rust_name()), Span::call_site());

            quote! {
                #method_sig {
                    let #e1 = self.#split(a);
                    let #e2 = self.#split(b);
                    self.#combine(self.#zip_low_half(#e3), self.#zip_high_half(#e3))
                }
            }
        }
        OpSig::Unzip { .. } => {
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    let (b0, b1) = self.#split(b);
                    self.#combine(self.#do_half(a0, a1), self.#do_half(b0, b1))
                }
            }
        }
        OpSig::Cvt {
            target_ty,
            scalar_bits,
            ..
        } => {
            let half = VecType::new(target_ty, scalar_bits, ty.len / 2);
            let combine = Ident::new(&format!("combine_{}", half.rust_name()), Span::call_site());
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::Reinterpret {
            target_ty,
            scalar_bits,
        } => {
            let mut half = ty.reinterpret(target_ty, scalar_bits);
            half.len /= 2;
            let combine = Ident::new(&format!("combine_{}", half.rust_name()), Span::call_site());
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::WidenNarrow { mut target_ty } => {
            target_ty.len /= 2;
            let combine = Ident::new(
                &format!("combine_{}", target_ty.rust_name()),
                Span::call_site(),
            );
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#combine(self.#do_half(a0), self.#do_half(a1))
                }
            }
        }
        OpSig::MaskReduce { quantifier, .. } => {
            let combine_op = quantifier.bool_op();
            quote! {
                #method_sig {
                    let (a0, a1) = self.#split(a);
                    self.#do_half(a0) #combine_op self.#do_half(a1)
                }
            }
        }
        OpSig::LoadInterleaved {
            block_size,
            block_count,
        } => {
            let split_len = (block_size * block_count) as usize / (ty.scalar_bits * 2);
            let delegate = format_ident!(
                "{op}_{}",
                VecType {
                    len: ty.len / 2,
                    ..*ty
                }
                .rust_name()
            );
            quote! {
                #method_sig {
                    let (chunks, _) = src.as_chunks::<#split_len>();
                    unsafe {
                        core::mem::transmute([self.#delegate(&chunks[0]), self.#delegate(&chunks[1])])
                    }
                }
            }
        }
        OpSig::StoreInterleaved { .. } => {
            quote! {
                #method_sig {
                    todo!()
                }
            }
        }
        OpSig::Split { .. }
        | OpSig::Combine { .. }
        | OpSig::AsArray { .. }
        | OpSig::FromArray { .. } => {
            panic!("These operations require more information about the target platform");
        }
        OpSig::FromBytes => generic_from_bytes(method_sig, ty),
        OpSig::ToBytes => generic_to_bytes(method_sig, ty),
    }
}

pub(crate) fn scalar_binary(name: &Ident, f: TokenStream, ty: &VecType) -> TokenStream {
    let ty_rust = ty.rust();
    quote! {
        #[inline(always)]
        fn #name(self, a: #ty_rust<Self>, b: #ty_rust<Self>) -> #ty_rust<Self> {
            core::array::from_fn(|i| #f(a[i], b[i])).simd_into(self)
        }
    }
}

pub(crate) fn generic_block_split(
    method_sig: TokenStream,
    half_ty: &VecType,
    max_block_size: usize,
) -> TokenStream {
    let split_arch_ty = half_ty.aligned_wrapper();
    let half_rust = half_ty.rust();
    let expr = match (half_ty.n_bits(), max_block_size) {
        (256, 128) => quote! {
            (
                #half_rust { val: #split_arch_ty([a.val.0[0], a.val.0[1]]), simd: self },
                #half_rust { val: #split_arch_ty([a.val.0[2], a.val.0[3]]), simd: self },
            )
        },
        (128, 128) | (256, 256) => quote! {
            (
                #half_rust { val: #split_arch_ty(a.val.0[0]), simd: self },
                #half_rust { val: #split_arch_ty(a.val.0[1]), simd: self },
            )
        },
        _ => unimplemented!(),
    };
    quote! {
        #method_sig {
            #expr
        }
    }
}

pub(crate) fn generic_block_combine(
    method_sig: TokenStream,
    combined_ty: &VecType,
    max_block_size: usize,
) -> TokenStream {
    let combined_arch_ty = combined_ty.aligned_wrapper();
    let combined_rust = combined_ty.rust();
    let expr = match (combined_ty.n_bits(), max_block_size) {
        (512, 128) => quote! {
            #combined_rust {val: #combined_arch_ty([a.val.0[0], a.val.0[1], b.val.0[0], b.val.0[1]]), simd: self }
        },
        (256, 128) | (512, 256) => quote! {
            #combined_rust {val: #combined_arch_ty([a.val.0, b.val.0]), simd: self }
        },
        _ => unimplemented!(),
    };
    quote! {
        #method_sig {
            #expr
        }
    }
}

pub(crate) fn generic_from_array(
    method_sig: TokenStream,
    vec_ty: &VecType,
    _kind: RefKind,
    max_block_size: usize,
    load_unaligned_block: impl Fn(&VecType) -> Ident,
) -> TokenStream {
    let block_size = max_block_size.min(vec_ty.n_bits());
    let block_count = vec_ty.n_bits() / block_size;
    let num_scalars_per_block = vec_ty.len / block_count;

    let native_block_ty = VecType::new(
        vec_ty.scalar,
        vec_ty.scalar_bits,
        block_size / vec_ty.scalar_bits,
    );

    let wrapper_ty = vec_ty.aligned_wrapper();
    let load_unaligned = load_unaligned_block(&native_block_ty);
    let expr = if block_count == 1 {
        quote! {
            unsafe { #wrapper_ty(#load_unaligned(val.as_ptr() as *const _)) }
        }
    } else {
        let blocks = (0..block_count).map(|n| n * num_scalars_per_block);
        quote! {
            unsafe { #wrapper_ty([
                #(#load_unaligned(val.as_ptr().add(#blocks) as *const _)),*
            ]) }
        }
    };
    let vec_rust = vec_ty.rust();

    quote! {
        #method_sig {
            #vec_rust { val: #expr, simd: self }
        }
    }
}

pub(crate) fn generic_as_array(
    method_sig: TokenStream,
    vec_ty: &VecType,
    kind: RefKind,
    max_block_size: usize,
    arch_ty: impl Fn(&VecType) -> Ident,
) -> TokenStream {
    let rust_scalar = vec_ty.scalar.rust(vec_ty.scalar_bits);
    let num_scalars = vec_ty.len;

    let ref_tok = kind.token();
    let native_ty = vec_ty.wrapped_native_ty(arch_ty, max_block_size);

    quote! {
        #method_sig {
            unsafe {
                // Safety: The native vector type backing any implementation will be:
                // - A `#[repr(simd)]` type, which has the same layout as an array of scalars
                // - An array of `#[repr(simd)]` types
                // - For AArch64 specifically, a `#[repr(C)]` tuple of `#[repr(simd)]` types
                //
                // Not only do these all have the same layout as a flat array of the corresponding scalars, but they
                // wrap primitives where all bit patterns are valid (ints and floats).
                core::mem::transmute::<#ref_tok #native_ty, #ref_tok [#rust_scalar; #num_scalars]>(#ref_tok a.val.0)
            }
        }
    }
}

pub(crate) fn generic_to_bytes(method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
    let bytes_ty = vec_ty.reinterpret(ScalarType::Unsigned, 8).rust();
    quote! {
        #method_sig {
            unsafe {
                #bytes_ty { val: core::mem::transmute(a.val), simd: self }
            }
        }
    }
}

pub(crate) fn generic_from_bytes(method_sig: TokenStream, vec_ty: &VecType) -> TokenStream {
    let ty = vec_ty.rust();
    quote! {
        #method_sig {
            unsafe {
                // Safety: All values are wrapped in alignment wrappers (`Aligned128`, `Aligned256`, `Aligned512`), so
                // we're transmuting between types with all valid bit patterns and the same size and alignment.
                #ty { val: core::mem::transmute(a.val), simd: self }
            }
        }
    }
}

pub(crate) fn impl_arch_types(
    level_name: &str,
    max_block_size: usize,
    arch_ty: impl Fn(&VecType) -> Ident,
) -> TokenStream {
    let mut assoc_types = vec![];
    for vec_ty in SIMD_TYPES {
        let ty_ident = vec_ty.rust();
        let wrapper_ty = vec_ty.aligned_wrapper_ty(&arch_ty, max_block_size);
        assoc_types.push(quote! {
            type #ty_ident = #wrapper_ty;
        });
    }
    let level_tok = Ident::new(level_name, Span::call_site());

    quote! {
        impl ArchTypes for #level_tok {
            #( #assoc_types )*
        }
    }
}
