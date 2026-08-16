<!-- Instructions

This changelog follows the patterns described here: <https://keepachangelog.com/en/>.

Subheadings to categorize changes are `added, changed, deprecated, removed, fixed, security`.

-->

The latest published Fearless SIMD release is [0.7.0](#070-2026-08-11) which was released on 2026-08-11.
You can find its changes [documented below](#070-2026-08-11).

## [Unreleased]

### Added

- - Added `mul_add_precise` and `mul_sub_precise` for floating-point vectors. They guarantee the infinite-precision product-plus-add rounded once, including on SIMD levels without hardware fused multiply-add instructions. They are not susceptible to the [bug](https://github.com/rust-lang/compiler-builtins/issues/1262) in Rust standard library, `std::simd` and musl libc that causes incorrect rounding for subnormal results. SSE4.2 gets SIMD emulation of these operations for better performance. ([#323][], [#324][] by [@Shnatsel][])
- Added `TryFrom` bounds to `SimdIntElement`, allowing attempted conversion from all primitive integer types.

## [0.7.0][] (2026-08-11)

This release has an [MSRV][] of 1.89.

### Added

- Added `i64x2`, `i64x4`, `i64x8`, `u64x2`, `u64x4`, and `u64x8` vector types, the native-width `i64s` and `u64s` associated types, and 64-bit integer operations across all backends. ([#253][], [#310][] by [@Shnatsel][])
- Added an `Sse2` level. This is the new baseline for i686-* and x86_64-* targets, replacing `Fallback`. It is detected at runtime on Tier-2 i586-* targets. ([#270][] by [@Shnatsel][])
- Added `shift_elements_left`, `shift_elements_right`, `rotate_elements_left`, and `rotate_elements_right` to non-mask vectors. Shifts accept a padding element and fill the entire vector when the offset is at least its lane count; rotations wrap the offset. ([#274][] by [@Shnatsel][])
- Added full-vector `swizzle_dyn` and `swizzle_dyn_precise` byte swizzles. `swizzle_dyn` permits implementation-defined results for out-of-range indices, while `swizzle_dyn_precise` always returns zero for them. ([#276][], [#304][] by [@Shnatsel][])
- Added the `SimdElement::BITS` constant, exposing the bit width of a vector's lane type to generic code. ([#296][] by [@danderson][])
- Added trait bounds on `SimdElement`, and introduced the `SimdIntElement` and `SimdFloatElement` subtraits. These allow generic code to access many math and utility operations on the elements of SIMD vector types. ([#302][] by [@danderson][])
- Added the `SimdWiden` and `SimdNarrow` traits, providing widening, narrowing, and saturating narrowing operations for all integer and floating-point vector types. ([#300][] by [@Shnatsel][])
- The four-way interleaved load and store operations are now exposed on the vector types they operate on and are available to generic code through `SimdInterleaved`. ([#321][] by [@Shnatsel][])

### Changed

- Breaking change: the `load_interleaved_128_*` and `store_interleaved_128_*` methods, which exchanged one 512-bit vector, have been replaced by `load_four_interleaved_*` and `store_four_interleaved_*`. The new methods exchange an array of four 128-bit vectors directly and are available for all non-mask scalar types. ([#298][] by [@Shnatsel][])
- Breaking change: the `new_unchecked()` function on SIMD level tokens such as `Avx2` has been renamed to `assume_supported()` and is now safe to call from contexts that already contain the appropriate `#[target_feature]` annotations. Functions without such annotations can still call `assume_supported()` with an `unsafe` block. ([#293][] by [@Shnatsel][])
- Breaking change: The `fxsr` CPU feature is now required for all x86 SIMD levels. It is present in hardware on all SIMD-capable CPUs, but it is possible to disable it in some emulators combined with a custom Rust target specification. ([#270][] by [@Shnatsel][])
- Breaking change: Operations shared by integer and floating-point vectors have moved from `SimdInt`/`SimdFloat` to `SimdBase`, so code generic over any non-mask vector can use `Add`, `Sub`, `Mul`, comparisons, zip/unzip, and interleave/deinterleave operations. ([#308][] by [@Shnatsel][])
- Breaking change: `min`, `max`, `min_precise`, and `max_precise` have moved from `SimdInt`/`SimdFloat` to `SimdBase`, allowing generic code to use them across integer and floating-point vectors. ([#313][] by [@Shnatsel][])
- On x86_64 targets with static SSE2 support, `Level::baseline()` now returns `Sse2` instead of `Fallback`. ([#270][] by [@Shnatsel][])
- The scalar `Fallback` backend and `Level::Fallback` variant are no longer compiled when the target has a better ambient SIMD baseline (e.g. SSE2 on x86, NEON on Aarch64). The `force_support_fallback` feature continues to make them available for testing. `disable_dispatch_sse2` no longer disables SSE2 if it is the baseline level. ([#320][] by [@Shnatsel][])
- Runtime CPU feature detection performed by `Level::new()` is now cached on x86. ([#278][] by [@Shnatsel][])
- The result of integer shifts by an amount greater than or equal to the element width is now explicitly documented as platform-dependent. Scalar fallback shifts use wrapping shift amounts instead of potentially panicking in debug builds. ([#283][] by [@Shnatsel][])
- Full-vector 8-bit shifts on x86 have been optimized, including a 2.4x faster left-shift formulation and faster signed and unsigned right shifts. ([#291][] by [@Shnatsel][])
- All native-width non-mask vector types now share `u8s` as their byte representation, enabling `Bytes::bitcast` between arbitrary lane types in code generic over `Simd`. The byte representation of any `SimdBase` type is now also guaranteed to be an idempotent, same-token `u8` SIMD vector, so it can be manipulated directly in generic code. ([#284][] by [@Shnatsel][])
- Generic bounds now encode existing relationships between masks, vectors, blocks, elements, and split/combined vector types. This enables mask selection, by-value element access, recursive block use, and reversible split/combine operations without additional bounds. ([#285][] by [@Shnatsel][])
- `SimdBase::Array` now guarantees `Copy` (and therefore `Clone`), `Debug`, by-value `IntoIterator`, `AsRef`, `AsMut`, and conversion from its vector type, while `SimdBase` guarantees construction from its associated array through `SimdFrom`. ([#285][] by [@Shnatsel][])
- `Simd` and `SimdBase` now require `Debug`, exposing implementations already provided by all tokens and vector types to generic code. ([#309][] by [@Shnatsel][])
- The `Simd::vectorize` documentation now explains when to use it and includes an end-to-end example. ([#312][] by [@Shnatsel][])
- Generated code and metadata have been substantially reduced, cutting x86 build time by roughly one third. ([#292][], [#317][], [#318][] by [@Shnatsel][])

### Removed

- Breaking change: removed the low-level `reinterpret_f32_*`, `reinterpret_f64_*`, `reinterpret_i32_*`, `reinterpret_u32_*`, `reinterpret_u8_*`, `cvt_to_bytes_*`, and `cvt_from_bytes_*` methods. Use `Bytes::bitcast` for arbitrary same-width bit reinterpretation, or `Bytes::to_bytes` and `Bytes::from_bytes` for direct byte-vector conversions. ([#284][] by [@Shnatsel][])
- Breaking change: removed the `WithSimd` trait, which only delegated to the `dispatch!` macro. Use `dispatch!` directly instead. ([#306][] by [@Shnatsel][])

### Fixed

- Integer negation in the scalar fallback now wraps for the minimum signed value, matching the SIMD backends instead of potentially panicking in debug builds. ([#253][] by [@Shnatsel][])
- Fixed x86 8-bit left shifts: overflowing `u8` lanes now wrap instead of saturating, and `i8` lanes now match Rust's signed shift semantics. ([#288][], [#290][] by [@danderson][])

## [0.6.0][] (2026-07-10)

This release has an [MSRV][] of 1.89.

### Added

- Added Ice Lake-class AVX-512 support with a generated `Avx512` level and 512-bit native-width vector types. ([#231][] by [@Shnatsel][])
- Added controls for x86 automatic multiversioning using `cfg` flags, letting final binary builds decide which levels `dispatch!` uses. ([#258][], [#264][] by [@Shnatsel][])
- Added the `swizzle_dyn_within_blocks` method on SIMD vector types for dynamic byte-indexed swizzles within each native-width block. ([#266][] by [@Shnatsel][])

### Changed

- The MSRV is now Rust 1.89. ([#231][] by [@Shnatsel][])
- Documentation and examples have been expanded and cleaned up to show autovectorization, use of intrinsics, x86 multiversioning, and AVX-512-width inputs. ([#252][], [#258][], [#261][], [#264][] by [@Shnatsel][])
- `Level::Fallback` and x86 `Level::Sse4_2` variants now remain available whenever their token types exist. ([#264][] by [@Shnatsel][])

### Fixed

- `dispatch!` and `Level::dispatch` now correctly omit AVX2 code paths when the Ice Lake/AVX-512 feature set is available. ([#263][] by [@Shnatsel][])

## [0.5.0][] (2026-06-18)

This release has an [MSRV][] of 1.88.

### Added

- The `kernel!` macro, which creates safe wrappers around SIMD-level-specific kernels so platform intrinsics from `core::arch` or `std::arch` can be used safely when a token proves the required target features. ([#214][] by [@Shnatsel][])
- The `approximate_recip` method on floating-point SIMD vector types. It uses fast hardware reciprocal estimates where available and exact division otherwise. ([#204][] by [@tomcur][])
- `SimdMask::from_bitmask`, `SimdMask::to_bitmask`, `SimdMask::test`, and `SimdMask::set`, mirroring the `std::simd` mask API. ([#226][] by [@Shnatsel][])

### Changed

- Breaking change: the crate's SIMD extension traits are now sealed, so external crates can no longer implement them for their own types. ([#211][] by [@LaurenzV][])
- Breaking change: mask types now have opaque storage and use the new `SimdMask` trait instead of `SimdBase`. Masks no longer expose integer-vector APIs such as `Deref`, indexing, `Bytes`, public `SimdSplit`/`SimdCombine`, `slide`, `slide_within_blocks`, byte conversions, or scalar bit-operator overloads. ([#218][] by [@Shnatsel][])
- Generated SIMD loads, stores, reference casts, transmute-like conversions, helpers, const-generic functions, and intrinsic calls now use checked wrappers or `kernel!`, removing most `unsafe` from generated code. ([#232][], [#233][], [#234][], [#235][], [#236][], [#237][], [#238][], [#239][], [#244][], [#245][] by [@Shnatsel][])
- Documentation and examples have been expanded and cleaned up for SIMD level tokens, mask types, platform-specific intrinsics, custom transmute wrappers, README consistency, and docs.rs visibility for NEON and WebAssembly APIs. ([#213][], [#221][], [#222][], [#230][], [#240][], [#243][] by [@Shnatsel][], [#224][], [#225][] by [@DJMcNab][])

### Removed

- Breaking change: the `core_arch` wrapper module and the `safe_wrappers` feature have been removed. Use `kernel!` with `core::arch` or `std::arch` intrinsics instead. ([#216][] by [@Shnatsel][])

## [0.4.1][] (2026-05-16)

This release has an [MSRV][] of 1.88.

### Added

- The `interleave` and `deinterleave` methods on integer and floating-point SIMD vector types. ([#206][] by [@Shnatsel][])

### Fixed

- `Sse4_2` and `Avx2` now consistently use the x86-64-v2 and x86-64-v3 feature sets for detection, dispatch, and generated `target_feature` attributes. ([#208][] by [@Shnatsel][])

## [0.4.0][] (2026-02-13)

This release has an [MSRV][] of 1.88.

### Added

- All vector types now implement `Index` and `IndexMut`. ([#112][] by [@Ralith][])
- 256-bit vector types now use native AVX2 intrinsics on supported platforms. ([#115][] by [@valadaptive][])
- 8-bit integer multiplication is now implemented on x86. ([#115][] by [@valadaptive][])
- New native-width associated types: `f64s` and `mask64s`. ([#125][] by [@valadaptive][])
- The bitwise "not" operation on integer vector types. ([#130][] by [@valadaptive][])
- The `from_fn` method on vector types. ([#137][] by [@valadaptive][])
- The `load_interleaved` and `store_interleaved` operations now use native intrinsics on x86, instead of using the fallback implementations. ([#140][] by [@valadaptive][])
- Add support for `relaxed_simd` operations in WebAssembly. ([#143][] by [@valadaptive][])
- The `ceil` and `round_ties_even` operations on floating-point vector types. (Rust's `round` operation rounds away from zero in the case of ties. Many architectures do not natively implement that behavior, so it's omitted.) ([#145][] by [@valadaptive][])
- A `prelude` module, which exports all the traits in the library but not the types. ([#149][] by [@valadaptive][])
- The `any_true`, `all_true`, `any_false`, and `all_false` methods on mask types. ([#141][] by [@valadaptive][])
- Documentation for most traits, vector types, and operations. ([#154][] by [@valadaptive][])
- A "shift left by vector" operation, to go with the existing "shift right by vector". ([#155][] by [@valadaptive][])
- "Precise" float-to-integer conversions, which saturate out-of-bounds results and convert NaN to 0 across all platforms. ([#167][] by [@valadaptive][])
- Add the `slide` and `slide_within_blocks` methods for shifting elements within a vector. ([#164][] by [@valadaptive][])
- The `Level::is_fallback` method, which lets you check if the current SIMD level is the scalar fallback. This works even if `Level::Fallback` is not compiled in, always returning false in that case. ([#168][] by [@valadaptive][])
- Added `store_array` methods to store SIMD vectors back to memory explicitly using intrinsics. ([#181][] by [@LaurenzV][])

### Fixed
- Improved the performance for load/store operations of vectors. ([#185][] by [@valadaptive][])
- Integer equality comparisons now function properly on x86. Previously, they performed "greater than" comparisons.
  ([#115][] by [@valadaptive][])
- All float-to-integer and integer-to-float conversions are implemented properly on x86, including the precise versions. ([#134][] by [@valadaptive][])
- The floating-point `min_precise` and `max_precise` operations now behave the same way on x86 and WebAssembly as they do on AArch64, returning the non-NaN operand if one operand is NaN and the other is not. Previously, they returned the second operand if either was NaN. ([#136][] by [@valadaptive][])

### Changed

- Breaking change: The AVX2 level now requires all features from the x86-64-v3 baseline. ([#188][] by [@Shnatsel][])
- Breaking change: `Level::fallback` has been removed, replaced with `Level::baseline`. ([#105][] by [@DJMcNab][])
  This corresponds with a change to avoid compiling in support for the fallback level on compilation targets which don't
  require it; this is most impactful for binary size on WASM, Apple Silicon Macs or Android.
  A consequence of this is that the available variants on `Level` are now dependent on the target features you are compiling with.
  The fallback level can be restored with the `force_support_fallback` cargo feature. We don't expect this to be necessary outside
  of tests.
- Code generation for `select` and `unzip` operations on x86 has been improved. ([#115][] by [@valadaptive][])
- Breaking change: The native-width associated types (`f32s`, `u8s`, etc.) for the `Avx2` struct have been widened from 128-bit
  types (like `f32x4`) to 256-bit types (like `f32x8`). ([#123][] by [@valadaptive][])
- Breaking change: All the vector types' inherent methods have been removed. Any remaining functionality has been moved
  to trait methods. ([#149][] by [@valadaptive][])

  Some functionality is exposed under different names:
  - Instead of the `reinterpret` methods, use the `bitcast` method on the `Bytes` trait. (e.g. `foo.reinterpret_i32()`
    -> `foo.bitcast::<i32x4<_>>()`)
  - Instead of the `cvt` methods, use the `to_int` or `to_float` convenience methods on the `SimdFloat` and `SimdInt`
    traits (e.g. `foo.cvt_u32()` -> `foo.to_int::<u32x4<_>>()`)

  Some functionality (such as `split` or `combine`) is exposed under new traits. You may use the new `prelude` module to
  conveniently import all of the traits.
- Breaking change: The `madd` and `msub` methods have been renamed to `mul_add` and `mul_sub`, matching Rust's naming conventions.
  ([#158][] by [@Shnatsel][])
- Breaking change: the `val` field on SIMD vector types is now private, and vector types are no longer represented as arrays internally. To access a vector type's elements, you can use the `Into` or `Deref` traits to obtain an array, or the `as_slice`/`as_mut_slice` methods to obtain a slice. ([#159][] by [@valadaptive][])
- Breaking change: the `Element` type on the `SimdBase` trait is now an associated type instead of a type parameter. This should make it more pleasant to write code that's generic over different vector types. ([#170][] by [@valadaptive][])
- The `WasmSimd128` token type now wraps the new `crate::core_arch::wasm32::WasmSimd128` type. This doesn't expose any new functionality as WASM SIMD128 can only be enabled statically, but matches all the other backend tokens. ([#176][] by [@valadaptive][])
- Breaking change: the `SimdFrom::simd_from` method now takes the SIMD token as the first argument instead of the second. This matches the argument order of the `from_slice`, `splat`, and `from_fn` methods on `SimdBase`. ([#180][] by [@valadaptive][])

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
[@tomcur]: https://github.com/tomcur
[@valadaptive]: https://github.com/valadaptive
[@LaurenzV]: https://github.com/LaurenzV
[@Shnatsel]: https://github.com/Shnatsel
[@danderson]: https://github.com/danderson

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
[#112]: https://github.com/linebender/fearless_simd/pull/112
[#115]: https://github.com/linebender/fearless_simd/pull/115
[#123]: https://github.com/linebender/fearless_simd/pull/123
[#125]: https://github.com/linebender/fearless_simd/pull/125
[#130]: https://github.com/linebender/fearless_simd/pull/130
[#134]: https://github.com/linebender/fearless_simd/pull/134
[#136]: https://github.com/linebender/fearless_simd/pull/136
[#137]: https://github.com/linebender/fearless_simd/pull/137
[#140]: https://github.com/linebender/fearless_simd/pull/140
[#141]: https://github.com/linebender/fearless_simd/pull/141
[#143]: https://github.com/linebender/fearless_simd/pull/143
[#145]: https://github.com/linebender/fearless_simd/pull/145
[#149]: https://github.com/linebender/fearless_simd/pull/149
[#154]: https://github.com/linebender/fearless_simd/pull/154
[#155]: https://github.com/linebender/fearless_simd/pull/155
[#158]: https://github.com/linebender/fearless_simd/pull/158
[#159]: https://github.com/linebender/fearless_simd/pull/159
[#164]: https://github.com/linebender/fearless_simd/pull/164
[#167]: https://github.com/linebender/fearless_simd/pull/167
[#168]: https://github.com/linebender/fearless_simd/pull/168
[#170]: https://github.com/linebender/fearless_simd/pull/170
[#176]: https://github.com/linebender/fearless_simd/pull/176
[#180]: https://github.com/linebender/fearless_simd/pull/180
[#181]: https://github.com/linebender/fearless_simd/pull/181
[#185]: https://github.com/linebender/fearless_simd/pull/185
[#188]: https://github.com/linebender/fearless_simd/pull/188
[#204]: https://github.com/linebender/fearless_simd/pull/204
[#206]: https://github.com/linebender/fearless_simd/pull/206
[#208]: https://github.com/linebender/fearless_simd/pull/208
[#211]: https://github.com/linebender/fearless_simd/pull/211
[#213]: https://github.com/linebender/fearless_simd/pull/213
[#214]: https://github.com/linebender/fearless_simd/pull/214
[#215]: https://github.com/linebender/fearless_simd/pull/215
[#216]: https://github.com/linebender/fearless_simd/pull/216
[#218]: https://github.com/linebender/fearless_simd/pull/218
[#221]: https://github.com/linebender/fearless_simd/pull/221
[#222]: https://github.com/linebender/fearless_simd/pull/222
[#224]: https://github.com/linebender/fearless_simd/pull/224
[#225]: https://github.com/linebender/fearless_simd/pull/225
[#226]: https://github.com/linebender/fearless_simd/pull/226
[#230]: https://github.com/linebender/fearless_simd/pull/230
[#231]: https://github.com/linebender/fearless_simd/pull/231
[#232]: https://github.com/linebender/fearless_simd/pull/232
[#233]: https://github.com/linebender/fearless_simd/pull/233
[#234]: https://github.com/linebender/fearless_simd/pull/234
[#235]: https://github.com/linebender/fearless_simd/pull/235
[#236]: https://github.com/linebender/fearless_simd/pull/236
[#237]: https://github.com/linebender/fearless_simd/pull/237
[#238]: https://github.com/linebender/fearless_simd/pull/238
[#239]: https://github.com/linebender/fearless_simd/pull/239
[#240]: https://github.com/linebender/fearless_simd/pull/240
[#243]: https://github.com/linebender/fearless_simd/pull/243
[#244]: https://github.com/linebender/fearless_simd/pull/244
[#245]: https://github.com/linebender/fearless_simd/pull/245
[#252]: https://github.com/linebender/fearless_simd/pull/252
[#258]: https://github.com/linebender/fearless_simd/pull/258
[#261]: https://github.com/linebender/fearless_simd/pull/261
[#263]: https://github.com/linebender/fearless_simd/pull/263
[#264]: https://github.com/linebender/fearless_simd/pull/264
[#266]: https://github.com/linebender/fearless_simd/pull/266
[#253]: https://github.com/linebender/fearless_simd/pull/253
[#270]: https://github.com/linebender/fearless_simd/pull/270
[#274]: https://github.com/linebender/fearless_simd/pull/274
[#276]: https://github.com/linebender/fearless_simd/pull/276
[#278]: https://github.com/linebender/fearless_simd/pull/278
[#283]: https://github.com/linebender/fearless_simd/pull/283
[#284]: https://github.com/linebender/fearless_simd/pull/284
[#285]: https://github.com/linebender/fearless_simd/pull/285
[#288]: https://github.com/linebender/fearless_simd/pull/288
[#290]: https://github.com/linebender/fearless_simd/pull/290
[#291]: https://github.com/linebender/fearless_simd/pull/291
[#292]: https://github.com/linebender/fearless_simd/pull/292
[#293]: https://github.com/linebender/fearless_simd/pull/293
[#296]: https://github.com/linebender/fearless_simd/pull/296
[#298]: https://github.com/linebender/fearless_simd/pull/298
[#300]: https://github.com/linebender/fearless_simd/pull/300
[#302]: https://github.com/linebender/fearless_simd/pull/302
[#304]: https://github.com/linebender/fearless_simd/pull/304
[#306]: https://github.com/linebender/fearless_simd/pull/306
[#308]: https://github.com/linebender/fearless_simd/pull/308
[#309]: https://github.com/linebender/fearless_simd/pull/309
[#310]: https://github.com/linebender/fearless_simd/pull/310
[#312]: https://github.com/linebender/fearless_simd/pull/312
[#313]: https://github.com/linebender/fearless_simd/pull/313
[#317]: https://github.com/linebender/fearless_simd/pull/317
[#318]: https://github.com/linebender/fearless_simd/pull/318
[#320]: https://github.com/linebender/fearless_simd/pull/320
[#321]: https://github.com/linebender/fearless_simd/pull/321
[#323]: https://github.com/linebender/fearless_simd/pull/323
[#324]: https://github.com/linebender/fearless_simd/pull/324

[Unreleased]: https://github.com/linebender/fearless_simd/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/linebender/fearless_simd/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/linebender/fearless_simd/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/linebender/fearless_simd/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/linebender/fearless_simd/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/linebender/fearless_simd/compare/v0.4.0...v0.3.0
[0.3.0]: https://github.com/linebender/fearless_simd/compare/v0.3.0...v0.2.0
[0.2.0]: https://github.com/linebender/fearless_simd/compare/e54304c66fc3e42d9604ddc7775b3345b589ce1a...v0.2.0
[0.1.1]: https://github.com/linebender/fearless_simd/compare/d683506b50721d35745cfc098527e007f1cb3425...e54304c66fc3e42d9604ddc7775b3345b589ce1a
[0.1.0]: https://github.com/linebender/fearless_simd/commit/d683506b50721d35745cfc098527e007f1cb3425

[MSRV]: README.md#minimum-supported-rust-version-msrv
