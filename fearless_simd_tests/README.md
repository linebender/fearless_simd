<div align="center">

# Fearless SIMD Tests

</div>

This is a development-only crate for testing `fearless_simd`.


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
    --config 'target.wasm32-wasip1.rustflags = "-Ctarget-feature=+simd128"' \
    --config 'target.wasm32-wasip1.rustdocflags = "-Ctarget-feature=+simd128"' \
    --config 'target.wasm32-wasip1.runner = "wasmtime"' # or "wasmi_cli" if you installed that
```
