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

Full details of how to use Fearless SIMD can be found in the [Fearless SIMD package's README](./fearless_simd/README.md).

## Motivation

This crate proposes an experimental way to use SIMD intrinsics reasonably safely.
The blog post [A plan for SIMD] contains the high level motivations, goal, and summary for Fearless SIMD.

## Repository Structure

The only package which is published to crates.io from this repository is Fearless SIMD, which can be found in the `fearless_simd` folder.
This folder also contains the examples.
The other packages are as follows:

- `fearless_simd_gen`: A code generator, used to generate the low signal-to-noise parts of the Fearless SIMD crate.
- `fearless_simd_tests`: Tests of functionality in Fearless SIMD, to validate that all implementations give the same and correct results.
- `fearless_simd_dev_macros`: Procedural macros used in `fearless_simd_tests` to generate versions of each test for each SIMD level supported on the current machine.

## History

A [much earlier version][fearless_simd 0.1.1] of this crate experimented with an approach that tried to accomplish safety in safe Rust as of 2018, using types that witnessed the SIMD capability of the CPU. There is a blog post, [Towards fearless SIMD], that wrote up the experiment. That approach couldn't quite be made to work, but was an interesting exploration at the time. A practical development along roughly similar lines is the [pulp] crate.

For more discussion about this crate, see [Towards fearless SIMD, 7 years later].

## Credits

This crate was inspired by [`pulp`], [std::simd], among others in the Rust ecosystem, though makes many decisions differently.
It benefited from conversations with Luca Versari, though he is not responsible for any of the mistakes or bad decisions.

## Minimum supported Rust Version (MSRV)

This version of Fearless SIMD has been verified to compile with **Rust 1.89** and later.

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
[std::simd]: https://doc.rust-lang.org/std/simd/index.html
[Towards fearless SIMD, 7 years later]: https://linebender.org/blog/towards-fearless-simd/
