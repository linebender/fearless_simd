<div align="center">

# Fearless SIMD

**Safer and easier SIMD**

[![Latest published version.](https://img.shields.io/crates/v/fearless_simd.svg)](https://crates.io/crates/fearless_simd)
[![Documentation build status.](https://img.shields.io/docsrs/fearless_simd.svg)](https://docs.rs/fearless_simd)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip, #simd channel.](https://img.shields.io/badge/Linebender-%23simd-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/514230-simd)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/fearless_simd/ci.yml?logo=github&label=CI)](https://github.com/linebender/fearless_simd/actions)
[![Dependency staleness status.](https://deps.rs/crate/fearless_simd/latest/status.svg)](https://deps.rs/crate/fearless_simd/)

</div>

<!-- We use cargo-rdme to update the README with the contents of lib.rs.
To edit the following section, update it in lib.rs, then run:
cargo rdme --workspace-project=fearless_simd
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here. 
See https://linebender.org/blog/doc-include/ for related discussion. -->

[libm]: https://crates.io/crates/libm
[`f32x4`]: https://docs.rs/fearless_simd/latest/fearless_simd/generated/simd_types/struct.f32x4.html
[`Simd`]: https://docs.rs/fearless_simd/latest/fearless_simd/generated/simd_trait/trait.Simd.html
[`SimdFrom`]: https://docs.rs/fearless_simd/latest/fearless_simd/traits/trait.SimdFrom.html
[SimdBase::from_slice]: https://docs.rs/fearless_simd/latest/fearless_simd/generated/simd_trait/trait.SimdBase.html#tymethod.from_slice
[`dispatch`]: https://docs.rs/fearless_simd/latest/fearless_simd/macro.dispatch.html
[`Level`]: https://docs.rs/fearless_simd/latest/fearless_simd/enum.Level.html
[`Level::new`]: https://docs.rs/fearless_simd/latest/fearless_simd/enum.Level.html#method.new
[`Level::dispatch`]: https://docs.rs/fearless_simd/latest/fearless_simd/enum.Level.html#method.dispatch
[`std::simd`]: https://doc.rust-lang.org/std/simd/index.html
[kernel]: https://docs.rs/fearless_simd/latest/fearless_simd/macro.kernel.html
[Simd::vectorize]: https://docs.rs/fearless_simd/latest/fearless_simd/trait.Simd.html#tymethod.vectorize

<!-- cargo-rdme start -->

`fearless_simd` takes `unsafe` out of SIMD.

No matter what level of abstraction you're after, be it autovectorization and multiversioning, or portable SIMD, or safe access to raw
intrinsics and nothing more, `fearless_simd` has you covered!

Zero dependencies, from-scratch build time under 1 second, safe public APIs, and [very little](https://gist.github.com/Shnatsel/61fc294987a1e051ce3835c97dc0fc19) `unsafe` under the hood.

## Automatic vectorization

Put the code to vectorize in an `#[inline(always)]` function generic over [`Simd`].

This will generate several implementations for different SIMD levels and select the best one at runtime:

```rust
use fearless_simd::{dispatch, Level, Simd};

#[inline(always)]
fn double_u32s<S: Simd>(_: S, values: &mut [u32]) {
    for value in values {
        *value = *value * 2;
    }
}

let mut values = [1, 2, 3, 4, 5];
let level = Level::new(); // Detect SIMD available on the CPU. Expensive, so do it once.
dispatch!(level, simd => double_u32s(simd, &mut values));
assert_eq!(values, [2, 4, 6, 8, 10]);
```

## Portable SIMD

Use the vector types for explicit lane-wise operations while staying generic over the SIMD level:

```rust
use fearless_simd::{dispatch, prelude::*, Level};

#[inline(always)]
fn double_u32s<S: Simd>(simd: S, values: &mut [u32]) {
    let mut chunks = values.chunks_exact_mut(S::u32s::N); // the CPU's native SIMD width
    for chunk in &mut chunks {
        let v = S::u32s::from_slice(simd, chunk);
        (v * 2).store_slice(chunk);
    }
    for value in chunks.into_remainder() {
        *value = *value * 2;
    }
}

let mut values = [1, 2, 3, 4, 5];
let level = Level::new(); // Detect SIMD available on the CPU. Expensive, so do it once.
dispatch!(level, simd => double_u32s(simd, &mut values));
assert_eq!(values, [2, 4, 6, 8, 10]);
```

You can also use fixed-size types such as [u32x8] instead of using the hardware's native SIMD width.

## Explicit intrinsics

If you need access to raw intrinsics, [`kernel!`][kernel] creates a function where they can be called safely:

```rust
use fearless_simd::{prelude::*, Level, u32x4};

fearless_simd::kernel!(
    fn double_u32s_neon(neon: Neon, values: &mut [u32]) {
        use core::arch::aarch64::*;

        let mut chunks = values.chunks_exact_mut(4);
        for chunk in &mut chunks {
            let v: uint32x4_t = u32x4::from_slice(neon, chunk).into(); // safe load
            let doubled = vmulq_u32(v, vdupq_n_u32(2)); // safe access to a NEON intrinsic
            let doubled: u32x4<_> = doubled.simd_into(neon);
            doubled.store_slice(chunk);
        }
        for value in chunks.into_remainder() {
            *value = *value * 2;
        }
    }
);

#[cfg(target_arch = "aarch64")]
{
    let level = Level::new(); // Detect SIMD available on the CPU. Expensive, so do it once.
    if let Some(neon) = level.as_neon() {
        let mut values = [1, 2, 3, 4, 5];
        double_u32s_neon(neon, &mut values);
        assert_eq!(values, [2, 4, 6, 8, 10]);
    }
}
```

You can also [mix and match](https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/srgb.rs)
intrinsics with the other approaches, using high-level code most of the time and dropping down to
hardware-specific intrinsics only when necessary.

## Inlining

Fearless SIMD relies heavily on Rust's inlining support to create functions which have the given target features enabled.

As a rule of thumb:

- All SIMD functions need `#[inline(always)]`.
- Use [`dispatch`] when calling SIMD code from non-SIMD code.
- Use [`vectorize()`][Simd::vectorize] when calling SIMD from SIMD if you don't want to force inlining.

[The article describing the design](https://gist.github.com/Shnatsel/61fc294987a1e051ce3835c97dc0fc19#the-abi-would-like-a-word) covers why this is the
case. There's also Q&A on [Zulip](https://xi.zulipchat.com/#narrow/channel/514230-simd/topic/inlining/with/546913433).

## Instruction set support

- x86/x86-64: [v2](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) (SSE4.2), [v3](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) (AVX2), [Ice Lake](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512) (AVX-512, avoiding early slow implementations)
- Aarch64: Baseline [NEON](https://en.wikipedia.org/wiki/Arm_architecture_family#Advanced_SIMD_(Neon))
- WebAssembly: [128-bit packed SIMD](https://github.com/WebAssembly/spec/blob/main/proposals/simd/SIMD.md), [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)

A scalar fallback is also provided for platforms, so your code still works even if SIMD is not available.

## WebAssembly

WASM SIMD doesn't have feature detection, and so you need to compile two versions of your bundle for WASM, one with SIMD and one without,
then select the appropriate one for your user's browser. This can be done via [the `wasm-feature-detect`
library](https://github.com/GoogleChromeLabs/wasm-feature-detect).

You can compile WebAssembly with the SIMD128 feature enabled via the `RUSTFLAGS` environment variable
(`RUSTFLAGS="-Ctarget-feature=+simd128"`), or by adding the compiler flags in your [Cargo
config.toml](https://doc.rust-lang.org/cargo/reference/config.html):

```toml
[target.'cfg(target_arch = "wasm32")']
rustflags = ["-Ctarget-feature=+simd128"]
rustdocflags = ["-Ctarget-feature=+simd128"]
```

If you want to compile both SIMD and non-SIMD versions of your WebAssembly library, your best option right now is to create a shell script
that builds it once with the `RUSTFLAGS` specified, and once without. [Cargo currently does not allow specifying compiler flags
per-profile.](https://github.com/rust-lang/cargo/issues/10271)

### Relaxed SIMD

Fearless SIMD can make use of the [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)
WebAssembly instructions, if the requisite target feature is enabled. These instructions can return implementation-dependent results
depending on what is fastest on the underlying hardware. They are only used for operations where we already give hardware-dependent results.

At the time of writing, relaxed SIMD is only supported in Chrome. To make use of it, you'll need to build two versions of your library, one
with relaxed SIMD enabled (`RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd"`) and one with it disabled, and then feature-detect at
runtime.

## Multiversioning on x86

x86 CPUs are not guaranteed to have any SIMD particular instruction set, so `fearless_simd` compiles a version
of each function generic over [`Simd`] for each instruction set, and [`dispatch`] selects the best one at runtime.

This is necessary to take advantage of SIMD, but results in an increased binary size on x86.
If binary size is a concern, the increase can be partially mitigated by setting
[`codegen-units=1`](https://nnethercote.github.io/perf-book/build-configuration.html#codegen-units)
or [`lto=true`](https://nnethercote.github.io/perf-book/build-configuration.html#link-time-optimization) in your Cargo.toml,
at the cost of longer build times.

As a last resort, you can turn off multiversioning for specific SIMD instruction sets by passing
`--cfg disable_dispatch_sse4_2`, `--cfg disable_dispatch_avx2`, or `--cfg disable_dispatch_avx512` in `RUSTFLAGS`.
These configuration flags only control automatic multiversioning. Disabling one does not remove its token type, its
[`Simd`] implementation, or explicit [`kernel`] support; for example, an `Avx2` token can still be used to call an
AVX2 kernel when the CPU supports it.

Note that later extensions can be beneficial even if you are only using 128-bit vectors:
AVX2 and AVX-512 provide more efficient instructions for some operations,
and AVX-512 also more than doubles the number of vector registers of all sizes.

You can also [disable certain instruction sets for select functions](https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/sigmoid.rs)
without disabling them globally.

## Feature Flags

The following crate [feature flags](https://doc.rust-lang.org/cargo/reference/features.html#dependency-features) are available:

- `std` (enabled by default): Get floating point functions from the standard library (likely using your target's libc).
  Also allows using [`Level::new`] on all platforms, to detect which target features are enabled.
- `libm`: Use floating point implementations from [libm]. Useful for `#[no_std]`.
- `force_support_fallback`: Force scalar fallback, to be supported, even if your compilation target has a better baseline.

At least one of `std` and `libm` is required; `std` overrides `libm`.

## Credits

This crate was inspired by [`pulp`], [`std::simd`], among others in the Rust ecosystem, though makes many decisions differently.
It benefited from conversations with Luca Versari, though he is not responsible for any of the mistakes or bad decisions.

[`pulp`]: https://crates.io/crates/pulp

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Fearless SIMD has been verified to compile with **Rust 1.89** and later.

Future versions of Fearless SIMD might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

## Community

[![Linebender Zulip, #simd channel.](https://img.shields.io/badge/Linebender-%23simd-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/514230-simd)

Discussion of Fearless SIMD development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically in [#simd](https://xi.zulipchat.com/#narrow/channel/514230-simd).
All public content can be read without logging in.

Contributions are welcome by pull request.
The [Rust code of conduct] applies.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[Rust Code of Conduct]: https://www.rust-lang.org/policies/code-of-conduct
