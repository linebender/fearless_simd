# Exact-width SVE2 prototype with bounded memory accesses

Enable the `experimental_sve2` Cargo feature and use nightly Rust on little-endian
AArch64. The prototype was developed with `rustc 1.101.0-nightly (12433c307
2026-10-24)`. The default build and MSRV are unchanged.

`Level::new()` detects SVE2 availability, reads the calling thread's active vector
length with `svcntb()`, and selects `Sve2_128`, `Sve2_256`, or `Sve2_512`
only when the active length matches exactly. Absence of SVE2 or any other
active length (including 1024 and 2048 bits) falls back to the existing baseline.
CPU feature detection may be cached by the standard library; the active vector
length is read on every call.

`Level::detect_sve2()` exposes the same check as an `Option<Level>`. Detection
requires `std`. With `no_std` + `libm`, callers can construct tokens through
`unsafe assume_supported()` after establishing the feature and exact length.
`Level::baseline()` does not infer a width from `target_feature = "sve2"`.

`Level::as_sve2_128()`, `as_sve2_256()`, and `as_sve2_512()` extract only the
matching token. There are no conversions between SVE2 levels: an exact 256-bit
length cannot establish an exact 128-bit length. Smaller logical vectors still
work with the same token; for example, `f32x4<Sve2_256>` uses bounded SVE
implementations on a thread whose active length is 256 bits. The generator
shares these smaller-width implementations and specializes native-width
rearrangements to use direct permutations.

## Known soundness hole: Send and Sync

This is deliberately an unsound research prototype. `Simd` requires `Send + Sync`,
and public vectors carry their token. SVE vector length is per-thread on Linux.
The safe detection API establishes a width on the detecting thread only:

1. Thread A has VL=256 and obtains a `Sve2_256` token through `Level::new()`.
2. Safe Rust transfers that token, a copy, a containing `Level`, or a vector to
   thread B, whose VL is only 128.
3. The bounded predicates cannot activate the required 256 bits. Safe vector
   operations can consequently read incompletely initialized results.

Every receiving thread must have the same active width, including for
`Sve2_128`. Transfer to a wider thread also violates the token contract:
bounded memory accesses do not protect permutations that depend on physical
register boundaries. They can produce incorrect results even though the
loads and stores stay within the logical arrays.

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
| `reverse`, zip/unzip, interleave/deinterleave | Direct permutations at native width; bounded table-based permutations for smaller logical vectors |
| Whole-vector byte swizzles, including precise variants | Bounded `TBL`; zero-filled inactive input lanes preserve precise out-of-range behavior |
| Concatenated byte swizzles, including precise variants | `TBL2` at native width; two single-register lookups combined with OR for smaller logical vectors |
| Swizzles within 128-bit blocks | Bounded `TBL`, adding each block's base to the low four index bits |

Typed swizzles continue to route through byte vectors. For the within-block
operation, out-of-range indices wrap within their block, which is permitted by
its implementation-defined contract.

Operations not listed use the existing scalar/generic implementations.
Wider-than-level operations use generic fallback composition; this prototype
does not optimize all split/combine paths.

At native width, the exact-length contract permits `REV`, `ZIP1`/`ZIP2`, and
`UZP1`/`UZP2` directly. These instructions use physical register boundaries,
which coincide with the logical vector boundaries only at native width.
For smaller logical vectors, reverse uses a descending table index; upper-half
zip gathers the logical upper halves before `ZIP1`; and unzip gathers even or
odd lanes from both inputs and selects between them at the logical midpoint.

Similarly, native-width concatenation uses `TBL2`, whose second table starts
at the physical vector length. Smaller concatenation instead looks up
`a[index]` and `b[index - logical_length]` with two `TBL` instructions and ORs
the results. Zeroing loads make entries outside each logical table zero,
including wrapped subtraction indices, preserving precise out-of-range semantics.

Float reductions retain the crate's existing fixed reduction tree. Directly
substituting SVE `FADDV` could change the promised bitwise results.

Every load/store uses a fixed-lane predicate that bounds memory access to the
logical vector. `SV_ALL` is never used, even at native width.
Results are stored into `MaybeUninit` arrays and then assumed initialized;
the width contract guarantees that all logical lanes are initialized.

## Correctness checks

The existing `#[simd_test]` harness has an additional SVE2 instance for every
operation test when the test crate's `experimental_sve2` feature is enabled.
Each such instance uses the detected exact-width level on the detecting thread.
Tests of smaller vectors retain that token rather than constructing a lower-level
token. For example, a run at 512 bits covers 128-bit and 256-bit operations on
`Sve2_512`, as well as native-width operations.
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
fallback, for example with `-cpu cortex-a53`. Also set it to `0` for unsupported
1024-bit (`-cpu max,sve-max-vq=8,sve-default-vector-length=128`) and 2048-bit
(`-cpu max,sve-max-vq=16,sve-default-vector-length=256`) lengths. QEMU's default
vector-length property is in bytes; `SVE_TEST_BITS` is the expected selected
level in bits. These wider configurations check fallback detection, not SVE2
operation coverage.

All three supported widths passed the full suite in debug (7,978 passed,
90 ignored) and release (7,969 passed, 90 ignored). Fallback detection passed
at 1024 and 2048 bits and on a NEON-only Cortex-A53. Host workspace tests and
the AArch64 `no_std` + `libm` check also passed.

## Assembly experiments

[examples/sve2_codegen.rs](examples/sve2_codegen.rs) contains named fixtures for
arithmetic chains at all three widths, table swizzles, concatenation,
within-block swizzles, reverse/zip/unzip, interleave, FMA/sqrt, integer reduction,
and the same 128-bit operation chain on all three levels.

```sh
cargo +nightly rustc -p fearless_simd --example sve2_codegen --features experimental_sve2 --target aarch64-unknown-linux-musl --release -- --emit=asm
rg --files target/aarch64-unknown-linux-musl/release | rg 'sve2_codegen.*\.s$'
```

Inspect both the exported `sve2_*` symbols and their `__fearless_simd_kernel`
helpers. The exported functions have a small argument-shuffling tail-call wrapper
because they cross from the baseline feature context.

The load/store predicate and the token's exact-length guarantee serve different
purposes. Fixed-lane predicates expose the fixed memory-access size to LLVM,
helping it eliminate intermediate array storage. The exact-length guarantee
allows native-width permutations to use the register's physical boundaries.
LLVM does not know that the predicate covers the entire physical register,
so some redundant lane-clearing instructions remain between operations.

The following kernel-body instruction counts include loads, stores, setup,
and return, but exclude the exported wrappers. They are not cycle estimates.

| Probe | Original exact width with `SV_ALL` | Bounded levels | Exact width with bounded memory accesses |
| --- | ---: | ---: | ---: |
| Native-width arithmetic chain | 10 | 14 | 14 |
| 256-bit reverse/zip/unzip | 7 | 21 | 11 |
| 512-bit interleave | 23 | 12 | 9 |
| 256-bit concatenated swizzle | 6 | 11 | 7 |

In particular, interleave now uses bounded loads, `ZIP1` and `ZIP2`, and two
stores directly into the return buffer. The original `SV_ALL` formulation
materialized two 64-byte stack arrays and copied them to the return buffer.
The current sampled kernels eliminate intermediate array stores and stack spills.

The 128-bit operation chains on `Sve2_256` and `Sve2_512` still share their
bounded implementation. The chain on `Sve2_128` instead uses direct native-width
permutations, so it no longer aliases those two wider-level fixtures.

These are code-generation observations and QEMU correctness tests, not native
hardware throughput measurements. Mask storage/conversion costs, broader
intrinsic coverage, and the thread-transfer contract remain open work.
