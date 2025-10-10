<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

The latest published Fearless SIMD release is [0.2.0](#020-2025-08-26) which was released on 2025-08-26.
You can find its changes [documented below](#020-2025-08-26).

## [Unreleased]

## [0.2.0][] (2025-08-26)

This release has an [MSRV][] of 1.86.

## [0.3.0][] (2025-10-09)

### Added

- `SimdBase::witness` to fetch the `Simd` implementation associated with a
  generic vector. ([#76][] by [@Ralith][])
- `Select` is now available on native-width masks. ([#77][], [#83][] by [@Ralith][])
- `Simd::shrv_*` preforms a right shift with shift amount specified
  per-lane. ([#79][] by [@Ralith][])
- The `>>` operator is implemented for SIMD vectors. ([#79][] by [@Ralith][])
- Assignment operator implementations. ([#80][] by [@Ralith][])
- `SimdFrom` splatting is available on native-width vectors. ([#84][] by [@Ralith][])
- Left shift by u32. ([#86][] by [@Ralith][])
- Unary negation of signed integers. ([#91][] by [@Ralith][])


### Fixed

- `Simd` now requires consistent mask types for native-width
  vectors. ([#75][] by [@Ralith][])
- `Simd` now requires consistent `Bytes` types for native-width vectors,
  enabling `Bytes::bitcast` in generic code. ([#81][] by [@Ralith][])
- Scalar fallback now uses wrapping integer addition. ([#85][] by [@Ralith][])

### Changed

- Breaking: `a.madd(b, c)` and `a.msub(b, c)` now correspond to `a *
  b + c` and `a * b - c` for consistency with `mul_add` in
  std. ([#88][] by [@Ralith][])
- Constructors for static SIMD levels are now `const` ([#93][] by [@Ralith][])

## [0.2.0][] (2025-08-26)

There has been a complete rewrite of Fearless SIMD.
For some details of the ideas used, see our blog post [*Towards fearless SIMD, 7 years later*](https://linebender.org/blog/towards-fearless-simd/).

The repository has also been moved into the Linebender organisation.

## [0.1.1][] (2018-11-05)

No changelog was kept for this release.

## [0.1.0][] (2018-10-19)

This is the initial release.
No changelog was kept for this release.

[@Ralith]: https://github.com/Ralith

[#75]: https://github.com/linebender/fearless_simd/pull/75
[#76]: https://github.com/linebender/fearless_simd/pull/76
[#77]: https://github.com/linebender/fearless_simd/pull/77
[#79]: https://github.com/linebender/fearless_simd/pull/79
[#80]: https://github.com/linebender/fearless_simd/pull/80
[#81]: https://github.com/linebender/fearless_simd/pull/81
[#83]: https://github.com/linebender/fearless_simd/pull/83
[#84]: https://github.com/linebender/fearless_simd/pull/84
[#85]: https://github.com/linebender/fearless_simd/pull/85
[#86]: https://github.com/linebender/fearless_simd/pull/86
[#88]: https://github.com/linebender/fearless_simd/pull/88
[#91]: https://github.com/linebender/fearless_simd/pull/91
[#93]: https://github.com/linebender/fearless_simd/pull/93

[Unreleased]: https://github.com/linebender/fearless_simd/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/linebender/fearless_simd/compare/e54304c66fc3e42d9604ddc7775b3345b589ce1a...v0.2.0
[0.1.1]: https://github.com/linebender/fearless_simd/compare/d683506b50721d35745cfc098527e007f1cb3425...e54304c66fc3e42d9604ddc7775b3345b589ce1a
[0.1.0]: https://github.com/linebender/fearless_simd/commit/d683506b50721d35745cfc098527e007f1cb3425

[MSRV]: README.md#minimum-supported-rust-version-msrv
