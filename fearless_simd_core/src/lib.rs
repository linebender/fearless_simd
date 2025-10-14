//! Tooling for Rust's target features.

// LINEBENDER LINT SET - lib.rs - v4
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]

// TODO: Do we want both an `x86` and `x86_64` module?
#[cfg(any(target_arch = "x86", target_arch = "x86_64", doc))]
pub mod x86;

pub mod trampoline;

#[cfg(feature = "std")]
extern crate std;

/// Token that a set of target feature is available.
///
/// Note that this trait is only meaningful when there are values of this type.
/// That is, to enable the target features in `FEATURES`, you *must* have a value
/// of this type.
///
/// Values which implement this trait are used in the second argument to [`trampoline!`],
/// which is a safe abstraction over enabling target features.
///
/// # Safety
///
/// To construct a value of a type implementing this trait, you must have proven that each
/// target feature in `FEATURES` is available.
pub unsafe trait TargetFeatureToken: Copy {
    /// The set of target features which are enabled for this run, if
    /// you have a value of this type.
    const FEATURES: &[&str];

    /// Enable the target features in `FEATURES` for a single run of `f`, and run it.
    ///
    /// `f` must be marked `#[inline(always)]` for this to work.
    ///
    /// Note that this does *not* enable the target features on the Rust side (e.g. for calling).
    /// To do so, you should instead use [`trampoline!`] directly - this is a convenience wrapper around `trampoline`
    /// for cases where the dispatch of simd values is handled elsewhere.
    fn vectorize<R>(self, f: impl FnOnce() -> R) -> R;
}

/// Run an operation in a context with specific target features enabled, validated with [`TargetFeatureToken`] values.
///
/// This is effectively a stable implementation of the "Struct Target Features" Rust feature,
/// which at the time of writing is neither in stable or nightly Rust.
/// This macro can be used to make SIMD dispatch safe in addition to make explicit SIMD, both safely.
///
/// # Reference
///
/// These reference examples presume that you have (values in brackets are the "variables"):
///
/// - An expression (`token`) of a type (`Token`) which is `TargetFeatureToken` for some target features (`"f1,f2,f3"`);
/// - A function (signature `fn uses_simd(val: [f32; 4]) -> [f32; 4]`) which is safe but enables a subset of those target features (`"f1,f2"`);
/// - Local values of types corresponding to the argument types (`a` of type `[f32; 4]`)
///
/// ```rust,ignore
/// trampoline!(Token = token => "f1,f2", uses_simd(a: [f32; 4]) -> [f32; 4])
/// ```
///
/// Multiple tokens are also supported by providing them in a sequence in square brackets:
///
/// ```rust,ignore
/// trampoline!([Token = token, Sse = my_sse] => "f1,f2,sse", uses_simd(a: [f32; 4]) -> [f32; 4])
/// ```
///
/// A more advanced syntax is available if you need to use generics.
/// That syntax is explained in comments around the macro's definition, which can be seen above.
/// For reference, the implementation used to implement [`vectorize`](TargetFeatureToken::vectorize) for `"sse"` is:
///
/// ```rust,ignore
/// trampoline!([Self = self] => "sse", <(R)> fn<(R)>(f: impl FnOnce() -> R = f) -> R { f() })
/// ```
///
/// There is also support for where clauses after the return type.
///
/// # Motivation
///
/// In Fearless SIMD, this macro has two primary use cases:
///
/// 1) To dispatch to a specialised SIMD implementation of a function using target specific
///    instructions which will be more efficient than generic version written using the portable subset.
/// 2) To implement the portable subset of SIMD operations.
///
/// To expand on use case 1, when using Fearless SIMD you will often be writing functions which are
/// instantiated for multiple different SIMD levels (using generics).
/// However, for certain SIMD levels, there may be specific instructions which solve your problem more
/// efficiently than using the generic implementations (as an example, consider SHA256 hashing, which has
/// built-in instructions on several architectures).
/// However, in such generic implementations, the Rust type system doesn't know which target features are enabled,
/// so it would ordinarily require writing code to:
///
/// - detect whether a specific target feature is supported.
/// - unsafely, enter a context where the target feature is enabled in a way which makes the type system aware of this.
///
/// This macro provides a way to do the second safely once you have completed the first.
///
/// # Example
///
/// This expands upon the example in the reference, written out completely.
///
/// ```rust,ignore
/// // Just once, acquire a token.
/// let token = Token::try_new();
/// // Later, dispatch based on whether that token is available, potentially multiple times:
///
/// /// Perform some computation using SIMD.
/// #[target_feature(enable = "f1,f2")]
/// fn uses_simd(val: [f32; 4]) -> [f32; 4] {
///     // ...
/// }
///
/// let a = [1., 2., 3., 4.];
/// let Some(token) = token else { return scalar_fallback(a) };
///
/// trampoline!(Token = token => "f1,f2", uses_simd(a: [f32; 4]) -> [f32; 4])
/// ```
///
/// Note that a function only operating on 128 bytes is probably too small for checking
/// whether a token exists just for it is worthwhile.
/// However, if you have amorphised the cost of that check between many function calls,
/// the `trampoline!` macro itself compiles down to a function call.
/// (This would be the case when this macro is being used to implement the portable subset of SIMD operations)
///
// TODO: We could write an example for each of ARM, x86, and conditionally compile it in?
/// Note that our examples are all ignored as there is no target feature which is available on every platform,
/// but we need these docs to compile for users on any platform.
///
/// # Soundness
///
/// This macro is designed to be sound, i.e. no input to this macro can lead to undefined behaviour
/// without using the `unsafe` keyword.
///
/// The operation provided will only ever be immediately called once on the same thread as the macro caller,
/// so safety justifications within the operation can rely on the context of the call site of this macro.
/// The shorthand format does not allow calling unsafe functions.
#[macro_export]
macro_rules! trampoline {
    // [Sse = sse] for "sse", <(u32)> fn<(T: Int)>(a: [T; 4]) -> T where (...) {...}
    (
        // The token types, with an expression to get a value of that token kind.
        [$($token_type: path = $token: expr),+$(,)?]
        // The target feature to enable. Must be a string literal.
        => $to_enable: literal,
        // The generic arguments to instantiate the call to the generated function with.
        // Note the inner brackets, needed because we can't write a parser for this in macros.
        $(<($($generic_instantiation: tt)+)>)?
        // The generic parameters to give the inner generated function.
        // Brackets needed as above.
        fn$(<($($generic_args: tt)*)>)?
        // The arguments to the function, with provided explicit values, plus return type and where clause.
        ($($arg_name: ident: $arg_type: ty = $arg_value: expr),*$(,)?) $(-> $ret: ty)?
            // The where clause of the generated function.
            // Note the inner brackets after `where`, needed as above.
            $(where ($($where: tt)*))?
        // The operation to run inside the context with the target feature enabled.
        $op: block
    ) => {{
        #[target_feature(enable = $to_enable)]
        #[inline]
        // TODO: Do we want any other attributes here?
        // Soundness: We wrap the $op in a wrapping block, to ensure that any inner attributes don't apply to the function.
        // This ensures that the user can't add `#![target_feature(enable = "xxx")]` to their block.
        // Soundness: Either of generic_args and `$where` could be used to exit the function item early, so aren't
        // inside an unsafe block.
        fn trampoline_impl$(<$($generic_args)*>)?($($arg_name: $arg_type),*) $(-> $ret)? $(where $($where)*)? { $op }

        $(
            // We validate that we actually have a token of each claimed type.
            let _: $token_type = $token;
        )+
        const {
            // And that the claimed types justify enabling the enabled target features.
            $crate::trampoline::is_feature_subset($to_enable, [$(<$token_type as $crate::TargetFeatureToken>::FEATURES),+])
                // TODO: Better failure message here (i.e. at least concatting the set of requested features)
                .unwrap();
        }

        $(
            // Soundness: We use `arg_value` outside of the macro body to ensure it doesn't
            // accidentally gain an unsafe capability.
            #[allow(clippy::redundant_locals, reason="Required for consistency/safety.")]
            let $arg_name = $arg_value;
        )*
        // Safety: We have validated that the target features enabled in `trampoline_impl` are enabled,
        // because we have values of token types which implement $crate::TargetFeatureToken
        // Soundness: `$generic_args` could be used to exit the path expression early. As `<>` are
        // not treated as "real" brackets by macros, this isn't practical to detect and avoid statically.
        // To try and ensure that this can't turn into unsoundess, the
        // `trampoline_impl::<$generic_instantiation>` is evaluated outside of an unsafe block.
        // In theory, if a user could make the value of `func` be an `unsafe` fn pointer or
        // item type, this would still be unsound.
        // However, we haven't found a way for this to compile given the trailing `>`,
        // so aren't aware of any actual unsoundess. But note that this hasn't been rigorously proven,
        // and new Rust features could open this up wider.
        let func = trampoline_impl$(::<$($generic_instantiation)*>)?;
        unsafe { func($($arg_name),*) }
    }};
    // Sse = sse => "sse", sse_do_x(a: [f32; 4], b: [f32; 4]) -> [f32; 4]
    ($token_type: path = $token: expr => $to_enable: literal, $function: ident($($arg_name: ident: $arg_type: ty),*$(,)?) $(-> $ret: ty)?) => {
        $crate::trampoline!(
            [$token_type = $token]
            => $to_enable,
            $function($($arg_name: $arg_type),*) $(-> $ret)?
        )
    };
    // [Sse = sse] => "sse", sse_do_x(a: [f32; 4], b: [f32; 4]) -> [f32; 4]
    ([$($token_type: path = $token: expr),+$(,)?] => $to_enable: literal, $function: ident($($arg_name: ident: $arg_type: ty),*$(,)?) $(-> $ret: ty)?) => {
        $crate::trampoline!(
            [$($token_type = $token),+]
            => $to_enable,
            fn($($arg_name: $arg_type = $arg_name),*) $(-> $ret)? { $function($($arg_name),*) }
        )
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(test)]
mod example_expansion {
    use core::arch::x86_64::{__m128, _mm_mul_ps};

    use crate::x86::{self, v1::Sse};

    #[target_feature(enable = "sse")]
    fn sse_mul_f32s(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        let a: __m128 = bytemuck::must_cast(a);
        let b: __m128 = bytemuck::must_cast(b);
        bytemuck::must_cast(_mm_mul_ps(a, b))
    }

    #[test]
    // This is a test so that it is runnable
    fn example_output() {
        let Some(sse) = x86::v1::Sse::try_new() else {
            panic!("Example code")
        };
        let a = [10_f32, 20_f32, 30_f32, 40_f32];
        let b = [4_f32, 5_f32, 6_f32, 7_f32];

        // Both of these example expansions, the former using the shorthand form:
        let res =
            trampoline!(Sse = sse => "sse", sse_mul_f32s(a: [f32; 4], b: [f32; 4]) -> [f32; 4]);
        assert_eq!(res, [40_f32, 100_f32, 180_f32, 280_f32]);
        let res = trampoline!([Sse = sse] => "sse", fn(a: [f32; 4] = a, b: [f32; 4] = b) -> [f32; 4] { sse_mul_f32s(a, b)});
        assert_eq!(res, [40_f32, 100_f32, 180_f32, 280_f32]);
        // will expand to:
        #[expect(unused_braces, reason = "Required for macro soundness.")]
        // Start expansion:
        let res = {
            #[target_feature(enable = "sse")]
            #[inline]
            fn trampoline_impl(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
                { sse_mul_f32s(a, b) }
            }
            let _: Sse = sse;
            const {
                crate::trampoline::is_feature_subset(
                    "sse",
                    [<Sse as crate::TargetFeatureToken>::FEATURES],
                )
                .unwrap();
            }
            #[allow(clippy::redundant_locals, reason = "Required for consistency/safety.")]
            let a = a;
            #[allow(clippy::redundant_locals, reason = "Required for consistency/safety.")]
            let b = b;
            let func = trampoline_impl;
            unsafe { func(a, b) }
        };
        // End expansion
        assert_eq!(res, [40_f32, 100_f32, 180_f32, 280_f32]);
    }
}
