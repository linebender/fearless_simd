# Bounded-width SVE2 prototype

Enable the `experimental_sve2` Cargo feature and use nightly Rust on little-endian
AArch64. The prototype was developed with `rustc 1.101.0-nightly (12433c307
2026-10-24)`. The default build and MSRV are unchanged.

`Level::new()` detects SVE2 availability, reads the calling thread's active vector
length with `svcntb()`, and selects the largest of `Sve2_128`, `Sve2_256`, or
`Sve2_512` that fits. These are **minimum** hardware widths and fixed logical
vector widths. For example, 1024-bit hardware selects `Sve2_512`. Absence of
SVE2 falls back to the existing baseline. CPU feature detection may be cached
by the standard library; the active vector length is read on every call.

`Level::detect_sve2()` exposes the same check as an `Option<Level>`. Detection
requires `std`. With `no_std` + `libm`, callers can construct tokens through
`unsafe assume_supported()` after establishing the feature and minimum length.
`Level::baseline()` does not infer a width from `target_feature = "sve2"`.

`Level::as_sve2_128()`, `as_sve2_256()`, and `as_sve2_512()` expose any supported
lower level. A 128-bit token can be used on 256-bit or wider hardware without
changing the thread's vector length. The generator shares one implementation
per logical width across levels: 128-bit operations on a `Sve2_256` token use
the same bounded SVE formulas as the operations on `Sve2_128`.

## Known soundness hole: Send and Sync

This is deliberately an unsound research prototype. `Simd` requires `Send + Sync`,
and public vectors carry their token. SVE vector length is per-thread on Linux.
The safe detection API establishes a width on the detecting thread only:

1. Thread A has VL=256 and obtains a `Sve2_256` token through `Level::new()`.
2. Safe Rust transfers that token, a copy, a containing `Level`, or a vector to
   thread B, whose VL is only 128.
3. The bounded predicates cannot activate the required 256 bits. Safe vector
   operations can consequently read incompletely initialized results.

Transfer to a thread with equal or greater width is now valid. `Sve2_128` has no
additional width-transfer restriction on SVE2-capable threads, since SVE always
provides at least 128 bits. The remaining width-related hole concerns the
256-bit and 512-bit levels.

There is no runtime defense against this in the prototype. `Sync` also allows
sharing through references. Thread-local caching would not fix escaped tokens.
The safe `Level::new()` API and existing trait bounds are retained intentionally
to evaluate the backend before deciding how to resolve this issue.

Keep tokens and their vectors on the detecting thread, or independently ensure
that every receiving thread has sufficient active width. Tests detect within each
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
| `reverse`, zip/unzip, interleave/deinterleave | All logical widths up to the selected level; table indices and predicates use logical boundaries |
| Whole-vector byte swizzles, including precise variants | Bounded `TBL`; zero-filled inactive input lanes preserve precise out-of-range behavior |
| Concatenated byte swizzles, including precise variants | Two single-register `TBL` lookups with a logical-width offset, combined with OR |
| Swizzles within 128-bit blocks | Bounded `TBL`, adding each block's base to the low four index bits |

Typed swizzles continue to route through byte vectors. For the within-block
operation, out-of-range indices wrap within their block, which is permitted by
its implementation-defined contract.

Operations not listed use the existing scalar/generic implementations.
Wider-than-level operations use generic fallback composition; this prototype
does not optimize all split/combine paths.

The bounded rearrangements do not use `REV`, `ZIP2`, or `UZP` directly, because
those instructions depend on physical register boundaries. Reverse uses a
descending logical-lane table index. Upper-half zip gathers the logical upper
halves before using `ZIP1`. Unzip gathers even or odd logical lanes from both
inputs and selects between them at the logical midpoint.

Similarly, `TBL2` puts its second table at the physical vector length. Bounded
concatenation instead looks up `a[index]` and `b[index - logical_length]` with
two `TBL` instructions. Inputs are loaded with zeroing predicates, so entries
outside each logical table are zero, including wrapped subtraction indices.
OR combines the two lookups and preserves precise out-of-range semantics.

Float reductions retain the crate's existing fixed reduction tree. Directly
substituting SVE `FADDV` could change the promised bitwise results.

Every load/store uses a fixed-lane predicate that bounds memory access to the
logical vector. `SV_ALL` is never used, even for a level's preferred width.
Results are stored into `MaybeUninit` arrays and then assumed initialized;
the minimum-width requirement is essential to the validity of that assumption.

## Correctness checks

The existing `#[simd_test]` harness has an additional SVE2 instance for every
operation test when the test crate's `experimental_sve2` feature is enabled.
Each such instance exercises **every supported SVE2 level**, not just the
detected preferred level. On 512-bit or wider hardware this means running each
test with 128-bit, 256-bit, and 512-bit tokens on the same thread.
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
fallback, for example with `-cpu cortex-a53`. Set it to `512` for 1024-bit
hardware (`-cpu max,sve-max-vq=8,sve-default-vector-length=128`) or 2048-bit
hardware (`-cpu max,sve-max-vq=16,sve-default-vector-length=256`). QEMU's default
vector-length property is in bytes; `SVE_TEST_BITS` is the expected selected
level in bits. These wider runs exercise all three bounded levels, including
the maximum physical byte-table size where every u8 index fits in a register.

The release suite passed at all five physical widths (128, 256, 512, 1024,
and 2048 bits): 7,968 tests passed and 90 were ignored in each run. Detection
also passed on a NEON-only Cortex-A53 configuration. The 2048-bit debug run
passed 7,977 tests with 90 ignored. Host workspace tests and the AArch64
`no_std` + `libm` check also passed.

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
because they cross from the baseline feature context. Within the SVE context,
the code must work at any physical width satisfying the level's minimum.

On the nightly compiler above, all three `sve2_subwidth_*` fixtures compile to
the same code: the 256-bit and 512-bit symbols are aliases of the 128-bit symbol.
The kernel uses `ptrue p0.s, vl4`, two bounded loads, an add, table-based reverse
and upper-half zip, and one bounded store. There are no intermediate array
stores or stack spills. The other sampled kernels likewise eliminate the
intermediate arrays. Whole-vector swizzles use one `TBL`; concatenated swizzles
use two `TBL` instructions, subtraction, and OR.

Compared with the former exact-width prototype, bounded predicates can retain
`SEL` lane-clearing operations between array-backed operations. These preserve
the zero-filled upper lanes needed by subsequent table lookups. Rearrangements
also require more instructions to respect logical boundaries. The benefit is
that these implementations can be reused at smaller widths and on wider hardware
without requiring a physical-width-specific version.

These are code-generation observations and QEMU correctness tests, not native
hardware throughput measurements. Mask storage/conversion costs, broader
intrinsic coverage, and the thread-transfer contract remain open work.
