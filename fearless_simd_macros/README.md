# Fearless SIMD macros

This crate provides the experimental `#[simd]` attribute for
[`fearless_simd`](https://crates.io/crates/fearless_simd). It is versioned
separately so that the macro can evolve without adding a procedural-macro
dependency to `fearless_simd` itself.

The macro currently targets the inline-friendly `vectorize` implementation on
the main branch. Until the next `fearless_simd` release containing that
implementation and the first macro release are published, use both packages
from Git:

```toml
[dependencies]
fearless_simd = { git = "https://github.com/linebender/fearless_simd" }
fearless_simd_macros = { git = "https://github.com/linebender/fearless_simd" }
```

`fearless_simd_macros` 0.1.0 must be published only after that core release;
this example should then use the released core version and
`fearless_simd_macros = "0.1"`.

Then apply `#[simd]` to a function whose first ordinary parameter is its SIMD
token:

```rust,ignore
use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

#[simd]
fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
    let mut chunks = values.chunks_exact_mut(S::u32s::N);
    for chunk in &mut chunks {
        let value = S::u32s::from_slice(simd, chunk);
        (value * 2).store_slice(chunk);
    }
    for value in chunks.into_remainder() {
        *value *= 2;
    }
}
```

Conceptually, the macro expands the body to:

```rust,ignore
fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
    simd.vectorize(
        #[inline(always)]
        || {
            // Original body.
        },
    )
}
```

The attributed closure ensures that the original body is inlined into the
target-feature-enabled function provided by `Simd::vectorize`. The macro does
not add an `#[inline]` attribute to the annotated function. Any existing
`#[inline]`, documentation, lint, conditional-compilation, or other function
attributes remain on that function unchanged.
Code-placement attributes such as `#[cold]` therefore continue to describe the
outer wrapper; their effects are not transferred to the generated closure or
the target-feature helper that executes it.

The closure and `vectorize` call are tail expressions, so the original body's
value and type checking are preserved.

## Accepted functions

The first typed parameter after an optional `self` receiver is treated as the
SIMD token. It must be a by-value identifier, such as `simd: S` or
`mut simd: S`. An unused token may be written as `_: S`; the macro gives it a
private hygienic binding. Destructured, `ref`, and `binding @ pattern`
parameters are not supported for the token. Neither `#[cfg]` nor `#[cfg_attr]`
may be placed on that parameter.

The macro accepts synchronous free functions, inherent methods, trait
implementation methods, and default trait methods. Generic parameters, `where`
clauses, return types, `unsafe`, and non-variadic `extern` ABIs are preserved.
Attribute arguments are not supported: write `#[simd]`, not `#[simd(...)]`.

`async`, `const`, variadic, bodyless, and specialization `default fn`
functions are rejected. The attributes `#[track_caller]`, `#[unsafe(naked)]`,
and `#[instruction_set]` are also rejected because moving the body into a
closure would invalidate their semantics or body requirements.
`#[target_feature]` and other attributes are preserved.

A trait's `#[track_caller]` attribute is inherited by its implementations, but
is not present in the implementation method's token stream when this macro
runs. Applying `#[simd]` to an implementation of a trait method declared with
`#[track_caller]` is therefore unsupported even though the macro cannot
diagnose it.

## Execution boundaries and captures

Only work performed while the function body is executing is covered by
`vectorize`. Code inside a returned future, closure, or lazy iterator runs
later and is not covered. Named helper functions do not inherit the enabled
target features; make them inlineable or annotate their own SIMD-generic body.
Recursive calls enter `vectorize` again.

The original body becomes a non-`move` `FnOnce` closure. Rust infers each
capture mode from how the body uses its parameters. As with any closure
conversion, the destruction order of captured values is not a stable
substitute for function-parameter destruction order. Avoid relying on the
relative drop order of by-value parameters with observable destructors in a
`#[simd]` function.

This procedural macro intentionally has no dependency on `fearless_simd`.
The selected token's type must therefore make the `vectorize` trait method
available, normally through an `S: Simd` bound.

## Minimum supported Rust version

This version of `fearless_simd_macros` has been verified to compile with Rust
1.89 and later. Future versions may increase this requirement.
