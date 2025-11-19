// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![expect(
    missing_docs,
    clippy::missing_assert_message,
    reason = "TODO: https://github.com/linebender/fearless_simd/issues/40"
)]

use fearless_simd::*;

mod harness;

// Ensure that we can cast between generic native-width vectors
#[expect(dead_code, reason = "Compile only test")]
fn generic_cast<S: Simd>(x: S::f32s) -> S::u32s {
    x.to_int()
}

#[allow(clippy::allow_attributes, reason = "Only needed in some cfgs.")]
#[allow(
    unused_variables,
    reason = "The constructed `Level` is only used in some cfgs."
)]
#[allow(
    dead_code,
    reason = "The `UNSUPPORTED_LEVEL_MESSAGE` is only used in some cfgs."
)]
#[test]
fn supports_highest_level() {
    const UNSUPPORTED_LEVEL_MESSAGE: &str = "This means that some of the other tests in this run may be false positives, that is, they have been marked as succeeding even though they would actually fail if they could run.\n\
        When these tests are run on CI, any false positives should be caught.\n\
        However, please open a thread in the #simd channel on the Linebender Zulip if you see this message.\n\
        That would allow us to know whether it's worth us setting up the tests to run on an emulated system (such as using QEMU).";

    let level = Level::new();

    // When running tests locally, ensure that every SIMD level to be tested is actually supported. The tests themselves
    // will return early and pass if run with an unsupported SIMD level.
    //
    // We skip this on CI because some runners may not support all SIMD levels--in particular, the macOS x86_64 runner
    // doesn't support AVX2.
    if std::env::var_os("CI").is_none() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert!(
            level.as_avx2().is_some(),
            "This machine does not support every `Level` supported by Fearless SIMD (currently AVX2 and below).\n{UNSUPPORTED_LEVEL_MESSAGE}",
        );

        #[cfg(target_arch = "aarch64")]
        assert!(
            level.as_neon().is_some(),
            "This machine does not support every `Level` supported by Fearless SIMD (currently NEON and below).\n{UNSUPPORTED_LEVEL_MESSAGE}",
        );
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    assert!(
        level.as_wasm_simd128().is_some(),
        "This environment does not support WASM SIMD128. This should never happen, since it should always be supported if the `simd128` feature is enabled."
    );
}
