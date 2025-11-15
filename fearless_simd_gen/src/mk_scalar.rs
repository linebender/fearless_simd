use proc_macro2::TokenStream;
use quote::quote;

use crate::{mk_fallback::fallback_methods, types::type_imports};

pub(crate) fn mk_scalar_impl() -> TokenStream {
    let imports = type_imports();
    let methods = fallback_methods();

    quote! {
        use core::ops::*;
        use crate::{seal::Seal, Level, Simd, SimdInto};

        #imports

        #[derive(Debug, Copy, Clone)]
        pub struct Scalar;

        impl Scalar {
            #[inline]
            pub const fn new() -> Self {
                Scalar
            }
        }

        impl Simd for Scalar {
            type f32s = f32;
            type u8s = u8;
            type i8s = i8;
            type u16s = u16;
            type i16s = i16;
            type u32s = u32;
            type i32s = i32;
            type mask8s = i8;
            type mask16s = i16;
            type mask32s = i32;

            #[inline(always)]
            fn level(self) -> Level {
                Level::Scalar(self)
            }

            #[inline]
            fn vectorize<F: FnOnce() -> R, R>(self, f: F) -> R {
                f()
            }

            #( #methods )*
        }

        impl Seal for Scalar {}
    }
}
