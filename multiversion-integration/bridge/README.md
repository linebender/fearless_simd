# Public-API generic bridge experiment

This is a hand expansion of an attribute macro that could hide Fearless SIMD's
generic token argument and propagate it through an annotated call graph.
It depends on an unmodified local `fearless_simd` 1.0.0-rc.1 checkout at
`dfae6d5` and does not use any hidden APIs.

The generated protocol only needs `Simd`, `Simd::vectorize`, `dispatch!`, and
`Level`; the explicit vector example also uses existing vector traits and types.
It does not enumerate backends or target features. The AVX2 names in the root
crate are code generation inspection entry points, not part of the protocol.

The `bridge-kernels` dependency represents a separately compiled crate. Its
`mix::__simd<S: Simd>` companion forwards the hidden generic witness. The root
crate's `process::__simd` calls it in a loop. Both wrap their bodies in
`simd.vectorize(#[inline(always)] move || { ... })` and mark the outer adapter
`#[inline(always)]`. The explicit vector example similarly composes a native
width vector kernel and the scalar-source loop.

The `move` matters for code generation: borrowing the zero-sized hidden token
into the closure can introduce unnecessary entry-point stack marshalling.

Verified on Rust 1.97.1 / LLVM 22.1.6, with release LTO disabled:

- Three correctness tests pass on x86-64 in debug and release.
- Three tests pass on AArch64/NEON under QEMU in release.
- Three tests pass on WASM SIMD128 under Wasmtime in release.
- `cargo check --offline --no-default-features --target x86_64-unknown-none`
  passes, using `libm` and an explicit `Level` at the public boundary.
- `python3 check_codegen.py` checks that the AVX2 body is identical to a
  handwritten loop after assembly-label normalization, uses YMM vectors, and
  contains no calls.

This is a generic-protocol feasibility experiment, not a finished attribute
macro. The tests use ordinary function calls to the hand-expanded companion
namespace. A production macro must additionally handle call rewriting, hygiene,
supported syntax, and its own cross-crate generated protocol versioning.

`Level::new` requires `std` outside WASM; a `no_std` frontend should accept a
supplied `Level` or explicitly choose `Level::baseline`. That is an existing
runtime detection policy limitation, not a required API change.

Reproduce:

```sh
cargo test --offline
cargo test --offline --release
python3 check_codegen.py
cargo check --offline --no-default-features --target x86_64-unknown-none
cargo test --offline --release --target aarch64-unknown-linux-musl --config 'target.aarch64-unknown-linux-musl.runner = "qemu-aarch64 -cpu cortex-a53"'
cargo test --offline --release --target wasm32-wasip1 --config 'target.wasm32-wasip1.rustflags = "-Ctarget-feature=+simd128"' --config 'target.wasm32-wasip1.rustdocflags = "-Ctarget-feature=+simd128"' --config 'target.wasm32-wasip1.runner = "wasmtime"'
```
