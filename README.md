<!--

This repo-level readme needs restructuring, pending some Linebender templating decisions.
https://xi.zulipchat.com/#narrow/channel/419691-linebender/topic/Bikeshedding.20badges/with/452312397

For now, prefer updating the package-level readmes, e.g. fearless_simd/README.md.

-->

<div align="center">

# Fearless SIMD

**Safer and easier SIMD**

[![Latest published version.](https://img.shields.io/crates/v/fearless_simd.svg)](https://crates.io/crates/fearless_simd)
[![Documentation build status.](https://img.shields.io/docsrs/fearless_simd.svg)](https://docs.rs/fearless_simd)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip, #simd channel.](https://img.shields.io/badge/Linebender-%23simd-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/514230-simd)
[![GitHub Actions CI status.](https://img.shields.io/github/actions/workflow/status/linebender/fearless_simd/ci.yml?logo=github&label=CI)](https://github.com/linebender/fearless_simd/actions)
[![Dependency staleness status.](https://deps.rs/repo/github/linebender/fearless_simd/status.svg)](https://deps.rs/repo/github/linebender/fearless_simd)

</div>

> [!CAUTION]
> Fearless SIMD is in extremely early experimental development. As such, there are no stability
> guarantees, APIs are incomplete, and architectures have missing implementations. Fearless SIMD is
> being developed in conjunction with the [Vello Sparse
> Strips](https://github.com/linebender/vello/) renderer.

## Motivation

This crate proposes an experimental way to use SIMD intrinsics reasonably safely.
The blog post [A plan for SIMD] contains the high level motivations, goal, and summary for Fearless SIMD.

## History

A [much earlier version][fearless_simd 0.1.1] of this crate experimented with an approach that tried to accomplish safety in safe Rust as of 2018, using types that witnessed the SIMD capability of the CPU. There is a blog post, [Towards fearless SIMD], that wrote up the experiment. That approach couldn't quite be made to work, but was an interesting exploration at the time. A practical development along roughly similar lines is the [pulp] crate.

For more discussion about this crate, see [Towards fearless SIMD, 7 years later]. A planned future direction is to autogenerate the the SIMD types and methods, rather than having to maintain a significant amount of boilerplate code.

## SIMD types

The SIMD types in this crate are a thin newtype around the corresponding array, for example `f32x4` is a newtype for `[f32; 4]`, and also contains a zero-sized token representing a witness to the CPU level. These types are in the crate root and have a number of methods, including loading and storing, that do not require intrinsics. The SIMD types are aligned, but this only affects storage.

## Levels

A central idea is "levels," which represent a set of target features. Each level has a corresponding module. Each module for a level has a number of submodules, one for each type (though with an underscore instead of `x` to avoid name collision), with a large number of free functions for SIMD operations that operate on that type.

On aarch64, the level is `neon`.

On x86-64, the planned supported level is `avx2`. This is actually short for the x86-64-v3 [microarchitecture level][x86-64 microarchitecture levels], which corresponds roughly to Haswell. The `avx512` level is also planned, which is x86-64-v4.

On wasm, the level is `simd128`.

## Credits

This crate was inspired by [pulp], [std::simd], among others in the Rust ecosystem, though makes many decisions differently. It benefited from conversations with Luca Versari, though he is not responsible for any of the mistakes or bad decisions.

The proc macro was strongly inspired by the [safe-arch-macro] in [rbrotli-enc].

## Minimum supported Rust Version (MSRV)

This version of Fearless SIMD has been verified to compile with **Rust 1.85** and later.

Future versions of Fearless SIMD might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

## Community

[![Linebender Zulip, #simd channel.](https://img.shields.io/badge/Linebender-%23simd-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/514230-simd)

Discussion of Fearless SIMD development happens in the [Linebender Zulip](https://xi.zulipchat.com/), specifically in [#simd](https://xi.zulipchat.com/#narrow/channel/514230-simd).
All public content can be read without logging in.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Contributions are welcome by pull request. The [Rust code of conduct] applies.
Please feel free to add your name to the [AUTHORS] file in any substantive pull request.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.

[Rust Code of Conduct]: https://www.rust-lang.org/policies/code-of-conduct
[AUTHORS]: ./AUTHORS

[A plan for SIMD]: https://linebender.org/blog/a-plan-for-simd/
[pulp]: https://crates.io/crates/pulp
[Towards fearless SIMD]: https://raphlinus.github.io/rust/simd/2018/10/19/fearless-simd.html
[fearless_simd 0.1.1]: https://crates.io/crates/fearless_simd/0.1.1
[x86-64 microarchitecture levels]: https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels
[std::simd]: https://doc.rust-lang.org/std/simd/index.html
[safe-arch-macro]: https://github.com/google/rbrotli-enc/blob/ce44d008ff1beff1eee843e808542d01951add45/safe-arch-macro/src/lib.rs
[rbrotli-enc]: https://github.com/google/rbrotli-enc
[Towards fearless SIMD, 7 years later]: https://linebender.org/blog/towards-fearless-simd/
