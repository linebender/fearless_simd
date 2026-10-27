# Exact-width SVE2 prototype

Enable the `experimental_sve2` Cargo feature and use nightly Rust on little-endian
AArch64. The prototype was developed with `rustc 1.101.0-nightly (12433c307
2026-10-24)`. The default build and MSRV are unchanged.

`Level::new()` detects SVE2 availability, reads the calling thread's active vector
length with `svcntb()`, and selects `Sve2_128`, `Sve2_256`, or `Sve2_512`. These
are **exact** lengths, not minimum lengths. Other lengths, or absence of SVE2,
fall back to the existing baseline. CPU feature detection may be cached by the
standard library; the active vector length is read on every call.

`Level::detect_sve2()` exposes the same check as an `Option<Level>`. Detection
requires `std`. With `no_std` + `libm`, callers can construct tokens through
`unsafe assume_supported()` after establishing the feature and exact length.
`Level::baseline()` does not infer a width from `target_feature = "sve2"`.

## Known soundness hole: Send and Sync

This is deliberately an unsound research prototype. `Simd` requires `Send + Sync`,
and public vectors carry their token. SVE vector length is per-thread on Linux.
The safe detection API establishes a width on the detecting thread only:

1. Thread A has VL=256 and obtains a `Sve2_256` token through `Level::new()`.
2. Safe Rust transfers that token, a copy, a containing `Level`, or a vector to
   thread B, whose VL is 512 (or 128).
3. Safe vector operations can now access beyond their fixed-size storage or
   read incompletely initialized results. Rearrangements can also return wrong
   lanes because they use physical register boundaries.

There is no runtime defense against this in the prototype. `Sync` also allows
sharing through references. Thread-local caching would not fix escaped tokens.
The safe `Level::new()` API and existing trait bounds are retained intentionally
to evaluate the backend before deciding how to resolve this issue.

Keep tokens and their vectors on the detecting thread, or independently ensure
that every receiving thread has the same active width. Tests detect within each
test thread and never change its width. Explicit per-thread vector-length
changes are assumed to require unsafe code and are out of scope. Streaming SME
mode is also outside this experiment.

Relevant interfaces: [Linux SVE thread state and vector-length controls](https://www.kernel.org/doc/html/latest/arch/arm64/sve.html),
[rustix's unsafe vector-length setter](https://docs.rs/rustix/latest/src/rustix/thread/prctl.rs.html).

## Implementation and operation coverage

The source is [mk_sve2.rs](../fearless_simd_gen/src/mk_sve2.rs). Regenerate with
`cargo run -p fearless_simd_gen`; do not edit `src/generated/sve2_*.rs` directly.

All public storage uses the existing aligned arrays, including masks. Associated
native vector types have fixed lane counts: for example, `Sve2_256::f32s` is
`f32x8<Sve2_256>`. Scalable SVE types exist only as locals inside target-feature
kernels. No fixed-VL LLVM compiler flags or C helpers are used.

| Operations | SVE implementation |
| --- | --- |
| Integer and float add, subtract, multiply | All element sizes, logical widths up to the selected native width |
| Float divide, square root, fused multiply-add | f32 and f64, logical widths up to the selected native width |
| Integer `reduce_sum` | Predicated horizontal sum, truncated for wrapping lane-width semantics |
| `reverse`, zip/unzip, interleave/deinterleave | Native logical width only: physical boundaries must match logical boundaries |
| Whole-vector byte swizzles, including precise variants | Native-width `TBL`; out-of-range indices produce zero |
| Concatenated byte swizzles, including precise variants | Native-width SVE2 two-register `TBL` |
| Swizzles within 128-bit blocks | Native-width `TBL`, adding each block's base to the low four index bits |

Typed swizzles continue to route through byte vectors. For the within-block
operation, out-of-range indices wrap within their block, which is permitted by
its implementation-defined contract.

Operations not listed use the existing scalar/generic implementations. Smaller
logical rearrangements deliberately avoid applying `REV`, `ZIP2`, or `UZP` to
the full physical register. Wider-than-native operations use generic fallback
composition; this prototype does not optimize all split/combine paths.

Float reductions retain the crate's existing fixed reduction tree. Directly
substituting SVE `FADDV` could change the promised bitwise results.

For exact native-width operations, the load/store predicate is all true.
For smaller logical vectors, a fixed-lane predicate bounds the memory access.
Results are stored into `MaybeUninit` arrays and then assumed initialized;
the width requirement is essential to the validity of that assumption.

## Correctness checks

The existing `#[simd_test]` harness has an additional SVE2 instance for every
operation test when the test crate's `experimental_sve2` feature is enabled.
The existing explicit test bodies cover arithmetic, wrapping, float edge cases,
out-of-range swizzles, cross-block shuffles, and smaller/wider logical vectors.
Additional ordinary tests check detection, dispatch, array layout, and operations
outside `vectorize()`.

Run the suite at each length, without changing the length of a running thread:

```sh
SVE_TEST_BITS=128 CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER="qemu-aarch64 -cpu max,sve-max-vq=1" cargo +nightly test -p fearless_simd_tests --features experimental_sve2 --target aarch64-unknown-linux-musl --test tests
SVE_TEST_BITS=256 CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER="qemu-aarch64 -cpu max,sve-max-vq=2" cargo +nightly test -p fearless_simd_tests --features experimental_sve2 --target aarch64-unknown-linux-musl --test tests
SVE_TEST_BITS=512 CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER="qemu-aarch64 -cpu max,sve-max-vq=4" cargo +nightly test -p fearless_simd_tests --features experimental_sve2 --target aarch64-unknown-linux-musl --test tests
```

Add `--release` to exercise optimized code. `SVE_TEST_BITS` makes the detection
test assert the expected backend, preventing a run without SVE from silently
passing the optional SVE operation tests. Set it to `0` when checking NEON
fallback, for example with `-cpu cortex-a53`, or an unsupported 1024-bit length
using `-cpu max,sve-max-vq=8,sve-default-vector-length=128` (the latter is bytes).

The optimized suite passed at all three supported widths: 7,962 tests passed
and 90 were ignored per run. Both fallback detection checks passed. The host
workspace tests and the AArch64 `no_std` + `libm` build also passed.

## Assembly experiments

[examples/sve2_codegen.rs](examples/sve2_codegen.rs) contains named fixtures for
arithmetic chains at all three widths, table swizzles, concatenation,
within-block swizzles, reverse/zip/unzip, interleave, FMA/sqrt, integer reduction,
and a smaller logical vector on a wider backend.

```sh
cargo +nightly rustc -p fearless_simd --example sve2_codegen --features experimental_sve2 --target aarch64-unknown-linux-musl --release -- --emit=asm
rg --files target/aarch64-unknown-linux-musl/release | rg 'sve2_codegen.*\.s$'
```

Inspect both the exported `sve2_*` symbols and their `__fearless_simd_kernel`
helpers. The exported functions have a small argument-shuffling tail-call wrapper
because they cross from the baseline feature context. Within the SVE context,
array-backed intermediate vectors optimize away in the native-width samples.

Observed with the nightly above, without globally enabling SVE2:

* The 128/256/512-bit arithmetic chains use three SVE loads, add/multiply/subtract/
  divide, and one SVE store; there are no intermediate stack accesses.
* Whole-vector and concatenated swizzles use one `TBL` each.
* The native reverse/zip-high/unzip-low chain uses `REV`, `ZIP2`, and `UZP1`, with
  no intermediate memory traffic.
* Interleave uses `ZIP1` and `ZIP2` and two output stores.
* FMA/sqrt and integer multiply/reduce-sum likewise retain intermediate values
  in registers.
* The 128-bit logical vector on `Sve2_512` mixes SVE arithmetic with fallback
  rearrangement and retains a 16-byte stack temporary. Array storage alone
  therefore does not guarantee good code for every composition.

An earlier bounded-predicate version used `PTRUE ... VL<n>` even at the exact
native width. LLVM retained redundant `SEL` lane-clearing operations between
array-backed operations. Using the exact-width guarantee to choose `PTRUE ALL`
removed them. This is a concrete code-quality benefit of exact-width levels.

These are code-generation observations and QEMU correctness tests, not native
hardware throughput measurements. Mask storage/conversion costs, broader
intrinsic coverage, and the thread-transfer contract remain open work.
