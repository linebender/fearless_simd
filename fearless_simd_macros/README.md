# Fearless SIMD macros

This crate provides the `#[simd]` attribute for
[`fearless_simd`](https://crates.io/crates/fearless_simd). It is versioned
separately so that the macro can evolve without adding a procedural-macro
dependency to `fearless_simd` itself.

Add both packages from crates.io to your `Cargo.toml`:

```toml
[dependencies]
fearless_simd = "1.0"
fearless_simd_macros = "0.1"
```

The library must be in scope as `fearless_simd` in the module containing the
annotated function. The dependency declaration above makes that name available
automatically. For a renamed Cargo dependency, add an alias in that module:

```rust,ignore
use simd_backend as fearless_simd;
```

A library re-export can be imported with
`use my_facade::simd_backend as fearless_simd;`. For use inside the
`fearless_simd` library itself, write `use crate as fearless_simd;`.
The macro does not inspect Cargo manifests to discover dependency names.

The library version must provide the internal `__fearless_simd_dispatch!`
helper. `Simd::vectorize` or `__fearless_simd_kernel_target_fn!` alone is not
sufficient for compatibility with older library versions.

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

Conceptually, the macro passes the body and its arguments to a generated
dispatcher:

```rust,ignore
fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
    dispatcher.call(
        simd,
        simd,
        values,
        #[inline(always)]
        |simd, values| {
            // Original body.
        },
    )
}
```

Here `dispatcher` represents generated helpers that select the token's backend
and enable its target features. The first token selects the backend; the
remaining values become arguments to the body. Each argument has a separate
helper parameter, allowing the compiler to pass it in registers even when the
helper remains out of line. There is no fixed argument-count limit.

The attributed closure ensures that the original body is inlined into the
target-feature-enabled helper. The helpers use ordinary `#[inline]` so large
bodies can remain shared between callers. The macro does not add an `#[inline]`
attribute to the annotated function. Any existing `#[inline]`, documentation,
lint, conditional-compilation, or other function attributes remain on that
function unchanged.
Code-placement attributes such as `#[cold]` therefore continue to describe the
outer wrapper; their effects are not transferred to the generated closure or
the target-feature helper that executes it.

The closure and dispatcher call are tail expressions, so the original body's
value is preserved. The closure also receives the function's declared return
type to preserve return-value coercions. Any `impl Trait` within that annotation
is replaced with `_` for inference; the function's argument and return types
remain unchanged. Ordinary parameter bindings and their lint attributes move
into the closure, and the outer parameters receive private names.

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

Only work performed while the function body is executing is covered by the
SIMD context. Code inside a returned future, closure, or lazy iterator runs
later and is not covered. Named helper functions do not inherit the enabled
target features; make them inlineable or annotate their own SIMD-generic body.
Recursive calls enter the dispatcher again.

The original body becomes an always-inline `FnOnce` closure with explicit
parameters. Receivers and parameters carrying `#[cfg]` or `#[cfg_attr]` remain
captures, preserving their existing semantics and uses inside nested macros.
These captures can still require memory when a helper remains out of line.
Rust infers their capture modes from how the body uses them. As with any closure
conversion, the destruction order of captured values is not a stable
substitute for function-parameter destruction order. Avoid relying on the
relative drop order of by-value parameters with observable destructors in a
`#[simd]` function.

The selected token must implement the library's `Simd` trait, normally through
an `S: Simd` bound. The procedural macro invokes
`fearless_simd::__fearless_simd_dispatch!`; the procedural-macro crate itself
does not depend on `fearless_simd`. The library helper owns the unsafe calls
and resolves all proof types through `$crate`, so a lookalike module cannot
substitute counterfeit proof tokens when re-exporting that helper.
Unknown future backends retain the existing `Simd::vectorize` path until the
library helper is updated to generate entries for them.

## Minimum supported Rust version

This version of `fearless_simd_macros` has been verified to compile with Rust
1.89 and later. Future versions may increase this requirement.
