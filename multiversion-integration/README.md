# Building annotation-only multiversioning on Fearless SIMD 1.0

The current public API is sufficient to build this as a separate macro/runtime
crate. No change to `fearless_simd`, its sealed traits, or its generated code is
needed for the integration demonstrated here. This work does not provide a
reason to delay the core 1.0 release.

This isolated workspace adapts the companion-module research prototype to the
unmodified `fearless_simd` **1.0.0-rc.1**, commit
`dfae6d52c0ac339f50b84cfad75b7284236494db`. Its runtime depends on the local core
crate through a path dependency. The original research repository is separate;
this directory contains the integration experiment.

## Architecture

An annotated source function retains its ordinary signature:

```rust
#[simd]
fn mix(x: u32) -> u32 {
    x.wrapping_mul(0x9e3779b9).rotate_left(7)
}
```

The macro generates a dispatching public entry and a hidden generic entry.
Conceptually, omitting the original-function marker used by the call protocol:

```rust
fn mix(x: u32) -> u32 {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    dispatch!(level, simd => mix::__call(simd, x))
}

mod mix {
    #[inline]
    pub fn __call<S: Simd>(simd: S, x: u32) -> u32 {
        simd.vectorize(#[inline(always)] move || {
            x.wrapping_mul(0x9e3779b9).rotate_left(7)
        })
    }
}
```

The actual implementation keeps bodies in their original lexical scope and
reexports hidden entries through companion modules. A participating caller
transforms `mix(x)` into a call to the generic entry with its own hidden token.
The companion-module/fallback-import protocol still lets Rust's resolver choose
between annotated and ordinary callees, including imports from another crate.

The macro generates the `S: Simd` parameters and token arguments; users do not
write them. Rust monomorphizes the hidden entries for the selected backend.
`Simd::vectorize` establishes the matching CPU feature context, and only the
body closure is forced to inline into that context. Internal calls propagate
the token instead of detecting or selecting a backend again.

Using `move` for the generated closure avoids borrowing the zero-sized token
into its capture environment. In these experiments, that removed unnecessary
stack marshalling at an outer feature boundary. The usual closure-conversion
limitations, including observable destruction order, remain frontend design work.

## Public APIs used

| Public API | Role |
| --- | --- |
| `Simd` | Bound on the generated, hidden backend type |
| `Simd::vectorize` | Correct target-feature boundary for each monomorphization |
| `dispatch!` | Runtime selection and architecture/configuration handling |
| `Level::try_detect`, `Level::baseline` | Selection policy that also compiles without `std` |
| Existing vector types and traits | Explicit SIMD in the separate generic bridge example |
| `kernel!`, `Level::as_avx2` | Optional concrete backend entry points, used here for codegen comparisons |

The generic macro and its runtime contain **no architecture list, target-feature
list, unsafe block, or direct reference to hidden core APIs**. They consume the
existing `Simd` implementations, so sealing does not obstruct the design.

`dispatch!` deliberately exposes an opaque `impl Simd` token rather than its
concrete type. Accepting any `S: Simd` works with that contract. Requiring an
additional external trait implemented only for named concrete tokens would not;
the generic protocol avoids that requirement.

Do not directly call the core's `__fearless_simd_*` helpers or
`Level::__dispatch_target`. They are implementation details of public macros,
and this integration does not depend on them directly.

## Feature detection and no_std

The standalone research witness only promised the `avx2` feature. Fearless SIMD's
`Avx2` token promises the complete x86-64-v3 feature set, including FMA and other
extensions. Replacing the old witness with `Avx2::assume_supported()` after an
AVX2-only check would be unsound. Using `dispatch!` and `vectorize` delegates both
the proof and the feature boundary to the core.

The facade uses `Level::try_detect()` and falls back to `Level::baseline()` when
runtime detection is unavailable. This is an explicit frontend policy. A frontend
can instead accept a caller-supplied `Level`, as the bridge example demonstrates,
without requiring a new core API. On bare-metal x86 this proof compiles with
`no_std` and `libm`; it does not add runtime CPU detection there.

## Verification

On Rust 1.97.1 / LLVM 22.1.6:

- The attribute-macro workspace passes all 17 tests in debug and release,
  including cross-crate calls, ordinary-call fallback, borrow/coercion semantics,
  and unsafe-operation rejection.
- Its 12 runtime tests pass on AArch64/NEON through QEMU and WASM SIMD128 through
  Wasmtime. The five host compiler probes are not run on those targets.
- The actual macro-generated demo compiles for `x86_64-unknown-none` with `no_std`
  and `libm`.
- Library tests also pass with Rust 1.89.0, the core crate's minimum supported
  version.
- With LTO disabled, the automatic AVX2 feature loop matches the handwritten
  `kernel!` loop after assembly-label normalization, with actual constants
  compared. It contains no calls or CPU detection. Both public `kernel!` entry
  wrappers have the same single forwarding tail jump.

The [codegen report](codegen-results.txt) distinguishes inlined and outlined
calls. A deliberately non-inlined generic helper retains a fixed forwarding
jump into its `vectorize` feature body. It performs no runtime backend selection,
but it is not instruction-for-instruction equivalent to a directly emitted
`#[target_feature]` helper with no wrapper. The opaque-loop comparison uses the
same generic bridge in automatic and manual code and explicitly records that
limitation. Further tuning of the frontend's outlining strategy is possible
without changing the core API: an external layer can emit concrete feature
functions authorized by the existing tokens, as well as generic bridges.

The [hand-expanded bridge](bridge/README.md) independently composes a cross-crate
scalar-source helper with existing native-width explicit SIMD operations. It
also has a no-LTO assembly comparison and a caller-supplied-level entry point.
The `alternatives/` directory contains the original independent design probes;
those do not use Fearless SIMD and are not integration evidence.

These are observations for the tested functions and compiler. Public dispatch
boundaries still have a cost, and the prototype is not a complete transformation
of arbitrary Rust syntax.

## Release implications

A separately versioned facade can depend on `fearless_simd = "1"` after the core
release and on its own proc-macro package. The core need not depend on the facade,
reexport its attribute, adopt a new feature flag, add trait methods, or reserve
specialization hooks before 1.0. New frontend options and syntax can evolve in
that separate package while the core API remains stable.

The frontend still needs production work on generics, methods, attributes,
callbacks, macro-generated calls, hygiene, diagnostics, and its supported syntax.
Those are frontend limitations, not missing core capabilities.

There are two separate compatibility questions:

- **Core compatibility:** this integration uses the existing documented APIs;
  no breaking core change was needed.
- **Frontend and downstream compatibility:** generated companion modules occupy
  the type namespace, so adding the annotation to an existing public downstream
  API can introduce name conflicts. Cross-crate generated entry names and token
  type identity also require a versioned protocol. The callable-marker design
  changes `Fn` and function-pointer behavior more substantially. These decisions
  belong to the optional frontend's release and adoption policy.

The recommendation is to ship the core according to its existing release
criteria and develop this frontend independently. No pre-1.0 API blocker was
identified for this approach.

## Reproduce

From the Fearless SIMD repository root (offline builds require cached dependencies):

```sh
cargo test --manifest-path multiversion-integration/Cargo.toml --offline
cargo test --manifest-path multiversion-integration/Cargo.toml --offline --release
python3 multiversion-integration/check_codegen.py --static
cargo +1.89.0 test --manifest-path multiversion-integration/Cargo.toml --offline --lib
cargo check --manifest-path multiversion-integration/Cargo.toml --offline --target x86_64-unknown-none -p simd-research-demo --no-default-features --features libm --lib
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER="qemu-aarch64 -cpu cortex-a53" cargo test --manifest-path multiversion-integration/Cargo.toml --offline --target aarch64-unknown-linux-musl -p simd-research-demo --lib --test semantics
cargo test --manifest-path multiversion-integration/Cargo.toml --offline --target wasm32-wasip1 -p simd-research-demo --lib --test semantics --config 'target.wasm32-wasip1.rustflags = "-Ctarget-feature=+simd128"' --config 'target.wasm32-wasip1.runner = "wasmtime"'
```

The source-level facade is in [demo/src/lib.rs](demo/src/lib.rs); the attribute
implementation is in [macros/src/lib.rs](macros/src/lib.rs), and the minimal
runtime glue is in [runtime/src/lib.rs](runtime/src/lib.rs).
