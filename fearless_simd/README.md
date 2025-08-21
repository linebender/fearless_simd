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
[`simd_dispatch`]: https://docs.rs/fearless_simd/0.2.0/fearless_simd/macros/macro.simd_dispatch.html
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

To create a function which SIMD and can be multiversioned, it will have a signature like:

```rust
use fearless_simd::{Simd, simd_dispatch};

#[inline(always)]
fn sigmoid_impl<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) { /* ... */ }

simd_dispatch!(fn sigmoid(level, x: &[f32], out: &mut [f32]) = sigmoid_impl);
```

A few things to note:

1) This is generic over any `Simd` type.
2) The [`simd_dispatch`] macro is used to create a multi-versioned version of the given function.
3) The `_impl` suffix is used by convention to indicate the version of a function which will be dispatched to.
4) The `impl` function *must* be `#[inline(always)]`.
   The performance of the SIMD implementation will be poor if that isn't the case. See [the section on inlining for details](#inlining)

The signature of the generated function will be:

```rust
use fearless_simd::Level;
fn sigmoid(level: Level, x: &[f32], out: &mut [f32]) { /* ... */ }
```

The first parameter to this function is the [`Level`].
If you are writing an application, you should create this once (using [`Level::new`]), and pass it to any function which wants to use SIMD.
This type stores which instruction sets are available for the current process, which is used
in the (generated) `sigmoid` function to dispatch to the most optimal variant of the function for this process.

# Inlining

Fearless SIMD relies heavily on Rust's inlining support to create functions which have the
given target features enabled.
As such, most functions which you write when using Fearless SIMD should have the `#[inline(always)]` attribute.
This is required because in LLVM, functions with different target features cannot.

<!--
# Kernels vs not kernels

TODO: Talk about writing versions of functions which can be called in other `S: Simd` functions.
I think this pattern can also have a macro.
-->

# Webassembly

WASM SIMD doesn't have feature detection, and so you need to compile two versions of your bundle for WASM, one with SIMD and one without,
then select the appropriate one for your user's browser.
TODO: Expand on this.

## Credits

This crate was inspired by [`pulp`], [`std::simd`], among others in the Rust ecosystem, though makes many decisions differently.
It benefited from conversations with Luca Versari, though he is not responsible for any of the mistakes or bad decisions.

# Feature Flags

The following crate [feature flags](https://doc.rust-lang.org/cargo/reference/features.html#dependency-features) are available:

- `std` (enabled by default): Get floating point functions from the standard library (likely using your target's libc).
  Also allows using [`Level::new`] on all platforms, to detect which target features are enabled.
- `libm`: Use floating point implementations from [libm].
- `safe_wrappers`: Include safe wrappers for (some) target feature specific intrinsics,
  beyond the basic SIMD operations abstracted on all platforms.

At least one of `std` and `libm` is required; `std` overrides `libm`.

[`pulp`]: https://crates.io/crates/pulp

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Fearless SIMD has been verified to compile with **Rust 1.86** and later.

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
