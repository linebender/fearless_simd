<div align="center">

# Fearless SIMD

**Safer and easier SIMD**

[![Latest published version.](https://img.shields.io/crates/v/fearless_simd.svg)](https://crates.io/crates/fearless_simd)
[![Documentation build status.](https://docs.rs/fearless_simd/badge.svg)](https://docs.rs/fearless_simd)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
\
[![Linebender Zulip, #simd channel.](https://img.shields.io/badge/Linebender-%23simd-blue?logo=Zulip)](https://xi.zulipchat.com/#narrow/channel/514230-simd)
[![GitHub Actions CI status.](https://github.com/linebender/fearless_simd/workflows/CI/badge.svg)](https://github.com/linebender/fearless_simd/actions)
[![Dependency staleness status.](https://deps.rs/repo/github/linebender/fearless_simd/status.svg)](https://deps.rs/repo/github/linebender/fearless_simd)

</div>

<!-- We use cargo-rdme to update the README with the contents of lib.rs.
To edit the following section, update it in lib.rs, then run:
cargo rdme --workspace-project=fearless_simd --heading-base-level=0
Full documentation at https://github.com/orium/cargo-rdme -->

<!-- Intra-doc links used in lib.rs should be evaluated here. 
See https://linebender.org/blog/doc-include/ for related discussion. -->

[libm]: https://crates.io/crates/libm
<!-- cargo-rdme start -->

A helper library to make SIMD more friendly.

# Feature Flags

The following crate [feature flags](https://doc.rust-lang.org/cargo/reference/features.html#dependency-features) are available:

- `std` (enabled by default): Get floating point functions from the standard library (likely using your targets libc).
- `libm`: Use floating point implementations from [libm].

At least one of `std` and `libm` is required; `std` overrides `libm`.

<!-- cargo-rdme end -->

## Minimum supported Rust Version (MSRV)

This version of Fearless SIMD has been verified to compile with **Rust 1.85** and later.

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
