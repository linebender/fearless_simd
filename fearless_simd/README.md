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

> [!CAUTION]
> Fearless SIMD is in extremely early experimental development. As such, there are no stability
> guarantees, APIs are incomplete, and architectures have missing implementations. Fearless SIMD is
> being developed in conjunction with the [Vello Sparse
> Strips](https://github.com/linebender/vello/) renderer.

<!-- We use cargo-rdme to update the README with the contents of lib.rs.
To edit the following section, update it in lib.rs, then run:
cargo rdme --workspace-project=fearless_simd --heading-base-level=0
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here. 
See https://linebender.org/blog/doc-include/ for related discussion. -->

[libm]: https://crates.io/crates/libm
[`f32x4`]: https://docs.rs/fearless_simd/latest/fearless_simd/generated/simd_types/struct.f32x4.html
[`Simd`]: https://docs.rs/fearless_simd/0.2.0/fearless_simd/generated/simd_trait/trait.Simd.html
[`SimdFrom`]: https://docs.rs/fearless_simd/0.2.0/fearless_simd/traits/trait.SimdFrom.html
[SimdBase::from_slice]: https://docs.rs/fearless_simd/0.2.0/fearless_simd/generated/simd_trait/trait.SimdBase.html#tymethod.from_slice
[`dispatch`]: https://docs.rs/fearless_simd/0.2.0/fearless_simd/macros/macro.dispatch.html
[`Level`]: https://docs.rs/fearless_simd/0.2.0/fearless_simd/enum.Level.html
[`Level::new`]: https://docs.rs/fearless_simd/0.2.0/fearless_simd/enum.Level.html#method.new
[`std::simd`]: https://doc.rust-lang.org/std/simd/index.html
<!-- cargo-rdme start -->

A helper library to make SIMD more friendly.

Fearless SIMD exposes safe SIMD with ergonomic multi-versioning in Rust.

Fearless SIMD uses "marker values" which serve as proofs of which target features are available on the current CPU.
These each implement the [`Simd`] trait, which exposes a core set of SIMD operations which are implemented as
efficiently as possible on each target platform.

Additionally, there are types for packed vectors of a specific width and element type (such as [`f32x4`]).
Fearless SIMD does not currently support vectors of less than 128 bits.
These vector types implement some standard arithmetic traits (i.e. they can be added together using
`+`, multiplied by a scalar using `*`, among others), which are implemented as efficiently
as possible using SIMD instructions.
These can be created in a SIMD context using the [`SimdFrom`] trait, or the
[`from_slice`][SimdBase::from_slice] associated function.

To call a function with the best available target features and get the associated `Simd`
implementation, use the [`dispatch!()`] macro:

```rust
use fearless_simd::{Level, Simd, dispatch};

#[inline(always)]
fn sigmoid<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) { /* ... */ }

// The stored level, which you should only construct once in your application.
let level = Level::new();

dispatch!(level, simd => sigmoid(simd, &[/*...*/], &mut [/*...*/]));
```

A few things to note:

1) `sigmoid` is generic over any `Simd` type.
2) The [`dispatch`] macro is used to invoke the given function with the target features associated with the supplied [`Level`].
3) The function or closure passed to [`dispatch!()`] should be `#[inline(always)]`.
   The performance of the SIMD implementation may be poor if that isn't the case. See [the section on inlining for details](#inlining)

The first parameter to [`dispatch!()`] is the [`Level`].
If you are writing an application, you should create this once (using [`Level::new`]), and pass it to any function which wants to use SIMD.
This type stores which instruction sets are available for the current process, which is used
in the macro to dispatch to the most optimal variant of the supplied function for this process.

# Inlining

Fearless SIMD relies heavily on Rust's inlining support to create functions which have the
given target features enabled.
As such, most functions which you write when using Fearless SIMD should have the `#[inline(always)]` attribute.

There is a rule of thumb for how to achieve things in Fearless SIMD:

- All SIMD functions need `#[inline(always)]`.
- Use [`dispatch!`] when calling SIMD code from non-SIMD code.
- Use [`vectorize()`](Simd::vectorize) when calling SIMD from SIMD if you don't want to force inlining.

We currently don't have docs explaining why this is the case.
You can read [this Zulip conversation](https://xi.zulipchat.com/#narrow/channel/514230-simd/topic/inlining/with/546913433)
for some train of thought explanation.

<!--
TODO: Also have concrete examples of each of these.

TODO: This is a really subtle point, and we do need there to be a well-written explanation available.
E.g. We might want names for these, e.g.:

# Kernels vs not kernels

TODO: Talk about writing versions of functions which can be called in other `S: Simd` functions.
-->

# WebAssembly

WASM SIMD doesn't have feature detection, and so you need to compile two versions of your bundle for WASM, one with SIMD and one without,
then select the appropriate one for your user's browser. This can be done via [the `wasm-feature-detect` library](https://github.com/GoogleChromeLabs/wasm-feature-detect).

You can compile WebAssembly with the SIMD128 feature enabled via the `RUSTFLAGS` environment variable (`RUSTFLAGS="-Ctarget-feature=+simd128"`), or by adding the compiler flags in your [Cargo config.toml](https://doc.rust-lang.org/cargo/reference/config.html):

```toml
[target.'cfg(target_arch = "wasm32")']
rustflags = ["-Ctarget-feature=+simd128"]
rustdocflags = ["-Ctarget-feature=+simd128"]
```

If you want to compile both SIMD and non-SIMD versions of your WebAssembly library, your best option right now is to create a shell script that builds it once with the `RUSTFLAGS` specified, and once without. [Cargo currently does not allow specifying compiler flags per-profile.](https://github.com/rust-lang/cargo/issues/10271)

## Relaxed SIMD

Fearless SIMD can make use of the [relaxed SIMD](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md) WebAssembly instructions, if the requisite target feature is enabled. These instructions can return implementation-dependent results depending on what is fastest on the underlying hardware. They are only used for operations where we already give hardware-dependent results.

At the time of writing, relaxed SIMD is only supported in Chrome. To make use of it, you'll need to build two versions of your library, one with relaxed SIMD enabled (`RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd"`) and one with it disabled, and then feature-detect at runtime.

# Credits

This crate was inspired by [`pulp`], [`std::simd`], among others in the Rust ecosystem, though makes many decisions differently.
It benefited from conversations with Luca Versari, though he is not responsible for any of the mistakes or bad decisions.

# Feature Flags

The following crate [feature flags](https://doc.rust-lang.org/cargo/reference/features.html#dependency-features) are available:

- `std` (enabled by default): Get floating point functions from the standard library (likely using your target's libc).
  Also allows using [`Level::new`] on all platforms, to detect which target features are enabled.
- `libm`: Use floating point implementations from [libm].
- `safe_wrappers`: Include safe wrappers for (some) target feature specific intrinsics,
  beyond the basic SIMD operations abstracted on all platforms.
- `force_support_fallback`: Force scalar fallback, to be supported, even if your compilation target has a better baseline.

At least one of `std` and `libm` is required; `std` overrides `libm`.

[`pulp`]: https://crates.io/crates/pulp

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Fearless SIMD has been verified to compile with **Rust 1.88** and later.

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
