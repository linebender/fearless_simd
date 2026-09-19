<div align="center">

# Fearless SIMD Tests

</div>

This is a development-only crate for testing `fearless_simd`.

### Testing the SIMD attribute

The UI suite runs explicitly on native x86-64 Linux, including on the minimum
supported Rust version:

```sh
cargo test -p fearless_simd_tests --locked --test simd_ui --target x86_64-unknown-linux-gnu -- --ignored
cargo +1.89 test -p fearless_simd_tests --locked --test simd_ui --target x86_64-unknown-linux-gnu -- --ignored
```

Fixtures in `tests/ui/simd/fail` use `ui_test` annotations to check an error at
the annotated source line. For diagnostics produced by the attribute, match a
focused part of the primary message:

```rust,ignore
#[simd(token = simd)] //~ ERROR: `#[simd]` does not accept arguments
```

For compiler diagnostics, match the error code, with a comment explaining what
the fixture protects:

```rust,ignore
#[simd] //~ E0277
```

An annotation matches either a message or a code, not both. Missing expected
errors and unexpected additional errors fail the test. Compiler notes, help,
warnings, and rendered output are not compared, and rustfix checks are disabled.
There are no `.stderr` snapshots to update. Use `//~^` to refer to the preceding
line when an inline annotation is inconvenient.

Fixtures in `tests/ui/simd/pass` use `//@run` so their runtime assertions execute
as well as compiling. The separate renamed-dependency fixture checks a renamed
import and a library re-export without the standard library or forced fallback.


### Testing WebAssembly +simd128

To run the WebAssembly tests, first install a WebAssembly runtime such as [wasmtime](https://docs.wasmtime.dev/introduction.html):

```sh
cargo install --locked wasmtime-cli
```

Or [wasmi](https://github.com/wasmi-labs/wasmi):

```sh
cargo install --locked --features simd wasmi_cli
```

Run WebAssembly tests with:

```sh
cargo test --target wasm32-wasip1 \
    --config 'target.wasm32-wasip1.rustflags = "-Ctarget-feature=+simd128,+relaxed-simd"' \
    --config 'target.wasm32-wasip1.rustdocflags = "-Ctarget-feature=+simd128,+relaxed-simd"' \
    --config 'target.wasm32-wasip1.runner = "wasmtime"' # or "wasmi_cli" if you installed that
```
