<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

# Changelog

The latest published Fearless SIMD release is [0.3.0](#030-2025-10-14) which was released on 2025-10-14.
You can find its changes [documented below](#030-2025-10-14).

## [Unreleased]

This release has an [MSRV][] of 1.89.

### Added

- All vector types now implement `Index` and `IndexMut`. ([#112][] by [@Ralith][])

### Changed

- Breaking change: `Level::fallback` has been removed, replaced with `Level::baseline`. ([#105][] by [@DJMcNab][])
  This corresponds with a change to avoid compiling in support for the fallback level on compilation targets which don't
  require it; this is most impactful for binary size on WASM, Apple Silicon Macs or Android.
  A consequence of this is that the available variants on `Level` are now dependent on the target features you are compiling with.
  The fallback level can be restored with the `force_support_fallback` cargo feature. We don't expect this to be necessary outside
  of tests.

### Removed

- Breaking change: The (deprecated) `simd_dispatch!` macro. ([#105][] by [@DJMcNab][])

## [0.3.0][] (2025-10-14)

This release has an [MSRV][] of 1.86.

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
- A simpler `dispatch` macro to replace `simd_dispatch`. ([#96][], [#99][] by [@Ralith][], [@DJMcNab][])

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
  Previously, `madd` was `a + b * c`, and `msub` was `a - b * c`.
  Therefore, if you previously had `a.madd(b, c)`, that's now written as `b.madd(c, a)`.
  And if you had `a.msub(b, c)`, that's now written `b.madd(-c, a)`.
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
[@DJMcNab]: https://github.com/DJMcNab

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
[#96]: https://github.com/linebender/fearless_simd/pull/96
[#99]: https://github.com/linebender/fearless_simd/pull/99
[#105]: https://github.com/linebender/fearless_simd/pull/105

[Unreleased]: https://github.com/linebender/fearless_simd/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/linebender/fearless_simd/compare/v0.3.0...v0.2.0
[0.2.0]: https://github.com/linebender/fearless_simd/compare/e54304c66fc3e42d9604ddc7775b3345b589ce1a...v0.2.0
[0.1.1]: https://github.com/linebender/fearless_simd/compare/d683506b50721d35745cfc098527e007f1cb3425...e54304c66fc3e42d9604ddc7775b3345b589ce1a
[0.1.0]: https://github.com/linebender/fearless_simd/commit/d683506b50721d35745cfc098527e007f1cb3425

[MSRV]: README.md#minimum-supported-rust-version-msrv
